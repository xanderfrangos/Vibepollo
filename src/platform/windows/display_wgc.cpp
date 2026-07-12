/**
 * @file src/platform/windows/display_wgc.cpp
 * @brief Windows Game Capture (WGC) IPC display implementation with shared session helper and DXGI fallback.
 */

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <winsock2.h>
#include <dxgi1_2.h>
#include <optional>
#include <wrl/client.h>

// local includes
#include "src/config.h"
#include "ipc/ipc_session.h"
#include "ipc/misc_utils.h"
#include "src/logging.h"
#include "src/platform/windows/display.h"
#include "src/platform/windows/display_vram.h"
#include "src/platform/windows/lsfg_framegen.h"
#include "src/platform/windows/misc.h"
#include "src/utility.h"

// platform includes
#include <winrt/base.h>

namespace platf::dxgi {
  namespace {
    struct wgc_dxgi_fallback_state_t {
      bool secure_desktop_active;
      bool recent_desktop_switch;
    };

    std::atomic<uint64_t> g_wgc_snapshot_copies {0};
    std::atomic<uint64_t> g_wgc_slow_snapshot_locks {0};
    std::atomic<uint64_t> g_wgc_slow_snapshot_copies {0};

    class adapter_luid_override_guard {
    public:
      explicit adapter_luid_override_guard(const std::optional<LUID> &luid) {
        previous_ = get_dxgi_adapter_luid_override();
        if (luid.has_value()) {
          set_dxgi_adapter_luid_override(luid);
        }
      }

      ~adapter_luid_override_guard() {
        set_dxgi_adapter_luid_override(previous_);
      }

    private:
      std::optional<LUID> previous_;
    };

    std::optional<wgc_dxgi_fallback_state_t> get_wgc_dxgi_fallback_state() {
      wgc_dxgi_fallback_state_t state {
        .secure_desktop_active = platf::dxgi::is_secure_desktop_active(),
        .recent_desktop_switch = recent_wgc_desktop_switch_grace_active()
      };

      if (!state.secure_desktop_active && !state.recent_desktop_switch) {
        return std::nullopt;
      }

      return state;
    }

    void log_wgc_dxgi_fallback_reason(const char *path_name, const wgc_dxgi_fallback_state_t &state) {
      if (state.secure_desktop_active && state.recent_desktop_switch) {
        BOOST_LOG(debug) << "Secure desktop detected and the desktop-switch grace window is active; "
                         << "using DXGI fallback for WGC capture (" << path_name << ")";
      } else if (state.secure_desktop_active) {
        BOOST_LOG(debug) << "Secure desktop detected, using DXGI fallback for WGC capture (" << path_name << ")";
      } else {
        BOOST_LOG(debug) << "Recent desktop switch grace window active, using DXGI fallback for WGC capture ("
                         << path_name << ")";
      }
    }

    bool is_wgc_constant_mode() {
      return config::video.capture == "wgcc";
    }

    std::chrono::milliseconds effective_wgc_timeout(std::chrono::milliseconds timeout, int client_framerate, bool fills_every_slot = false) {
      if (timeout.count() != 0) {
        return timeout;
      }

      // WGC's instantaneous publish cadence is vsync-stepped on the source
      // monitor (frame_qpc_delta tracks 240/N for an N integer), so the
      // window between our sleep_target wake and the next helper publish
      // can be up to one full helper interarrival. When DWM is composing
      // at 240/2 = 120 fps the gap is ~8.3 ms, but when it bounces between
      // 240 and 120 the average interarrival drops to ~6 ms and a 4 ms
      // grace times out frequently -- captured rate falls visibly below
      // the helper publish rate even though the helper is producing more
      // than enough frames. 6 ms covers the high-rate regime while still
      // staying under the 8.33 ms 120 Hz cadence with ~2 ms of slack for
      // the snapshot's CPU work.
      auto grace = std::chrono::milliseconds(6);

      if ((is_wgc_constant_mode() || fills_every_slot) && client_framerate > 0) {
        // In constant mode a timed-out zero-timeout snapshot forwards the
        // cached frame, so the pacing group must survive the full grace: the
        // snapshot returns at slot + grace, and if that lands past the next
        // slot the capture loop invalidates the pacing group and re-anchors
        // through a 200 ms blocking snapshot. Above ~166 fps the client frame
        // interval is shorter than the 6 ms grace, so an idle screen busts
        // the group on every forwarded frame and capture collapses to the
        // compositor's publish rate (vibepollo#267). Clamp the grace to leave
        // ~2 ms of the slot for wait/scheduling overhead.
        const auto frame_interval = std::chrono::nanoseconds(std::chrono::seconds(1)) / client_framerate;
        const auto clamped = std::chrono::duration_cast<std::chrono::milliseconds>(frame_interval - std::chrono::milliseconds(2));
        grace = std::clamp(clamped, std::chrono::milliseconds(1), grace);
      }

      return grace;
    }

    // The helper detects UAC/lock transitions via a desktop-switch hook, but it runs
    // with the user's token and its notification can be missed entirely (hook not yet
    // installed, pipe down, event lost). Without it the stream freezes on the last
    // pre-UAC frame forever. As a safety net, once frame delivery stalls, probe the
    // input desktop from this process; when the secure desktop is active, force a
    // reinit so the factory swaps to the DXGI fallback.
    bool wgc_stall_requires_dxgi_fallback(std::chrono::steady_clock::time_point &stall_start, std::chrono::steady_clock::time_point &last_probe) {
      constexpr auto stall_before_first_probe = std::chrono::milliseconds(200);
      constexpr auto probe_interval = std::chrono::milliseconds(250);

      const auto now = std::chrono::steady_clock::now();
      if (stall_start.time_since_epoch().count() == 0) {
        stall_start = now;
        last_probe = {};
      }

      if (now - stall_start < stall_before_first_probe) {
        return false;
      }

      if (last_probe.time_since_epoch().count() != 0 && now - last_probe < probe_interval) {
        return false;
      }

      last_probe = now;
      return is_secure_desktop_active();
    }

    capture_e forward_cached_wgc_frame(std::shared_ptr<platf::img_t> cached_frame, std::shared_ptr<platf::img_t> &img_out) {
      if (!cached_frame) {
        return capture_e::timeout;
      }

      const auto now = std::chrono::steady_clock::now();
      cached_frame->frame_timestamp = now;
      cached_frame->host_processing_timestamp = now;
      cached_frame->capture_pacing_timestamp = now;
      img_out = std::move(cached_frame);
      return capture_e::ok;
    }

    // 100% at 1920x1080 down to 50% at 3840x2160, linearly interpolated by total
    // pixel count and clamped at both ends so lower/higher resolutions still get
    // a usable flow scale instead of extrapolating past the configured range.
    int auto_lsfg_flow_scale_percent(int width, int height) {
      constexpr double k_min_pixels = 1920.0 * 1080.0;  // 100% flow scale
      constexpr double k_max_pixels = 3840.0 * 2160.0;  // 50% flow scale
      const double pixels = static_cast<double>(width) * static_cast<double>(height);
      if (pixels <= k_min_pixels) {
        return 100;
      }
      if (pixels >= k_max_pixels) {
        return 50;
      }
      const double t = (pixels - k_min_pixels) / (k_max_pixels - k_min_pixels);
      return static_cast<int>(std::lround(100.0 - t * 50.0));
    }

    int resolve_lsfg_base_flow_percent(int client_width, int client_height) {
      if (config::video.lsfg.auto_flow_scale && client_width > 0 && client_height > 0) {
        return auto_lsfg_flow_scale_percent(client_width, client_height);
      }
      return std::clamp(config::video.lsfg.flow_scale, 25, 100);
    }
  }  // namespace

  display_wgc_ipc_vram_t::display_wgc_ipc_vram_t() = default;

  display_wgc_ipc_vram_t::~display_wgc_ipc_vram_t() {
    if (_frame_locked && _ipc_session) {
      _ipc_session->release();
      _frame_locked = false;
    }
  }

  int display_wgc_ipc_vram_t::init(const ::video::config_t &config, const std::string &display_name) {
    _config = config;
    _display_name = display_name;

    if (display_base_t::init(config, display_name, true /* skip_dd_test: WGC doesn't use Desktop Duplication */)) {
      return -1;
    }

    capture_format = DXGI_FORMAT_UNKNOWN;  // Start with unknown format (prevents race condition/crash on first frame)

    // Host-side LSFG frame generation: the interpolator itself is created lazily on
    // the first frame (the capture format isn't known yet), but the pacing decision
    // has to be made now. Generated frames fill pacing slots the source can't, so
    // the capture loop may pace above the host display's refresh rate.
    _lsfg_requested = config::video.lsfg.enabled;
    if (_lsfg_requested) {
      pacing_allow_above_refresh = true;
    }

    const bool advanced_color_capture = is_hdr();

    // Create session
    _ipc_session = std::make_unique<ipc_session_t>();
    if (_ipc_session->init(config, display_name, device.get(), advanced_color_capture)) {
      return -1;
    }

    return 0;
  }

  capture_e display_wgc_ipc_vram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    if (!_ipc_session) {
      return capture_e::error;
    }

    // We return capture::reinit for most scenarios because the logic in picking which mode to capture is all handled in the factory function.
    if (_ipc_session->should_swap_to_dxgi()) {
      return capture_e::reinit;
    }

    // Generally this only becomes true if the helper process has crashed or is otherwise not responding.
    if (_ipc_session->should_reinit()) {
      return capture_e::reinit;
    }

    _ipc_session->initialize_if_needed();
    if (!_ipc_session->is_initialized()) {
      BOOST_LOG(warning) << "WGC IPC helper failed to initialize; requesting capture reinit.";
      return capture_e::reinit;
    }

    // Hot-apply lsfg_capture_framegen toggling mid-stream. pacing_allow_above_refresh
    // feeds display_base_t::capture()'s client_frame_rate_adjusted, which is computed
    // once at the top of that function's long-running loop -- flipping the flag alone
    // wouldn't change the active pacing ceiling until the loop restarts, so request a
    // reinit, the same mechanism video.cpp already uses for other mid-stream settings
    // changes (e.g. runtime bitrate). Other lsfg_* settings don't touch pacing and are
    // hot-applied below without a reinit.
    if (const bool lsfg_enabled_now = config::video.lsfg.enabled; lsfg_enabled_now != _lsfg_requested) {
      BOOST_LOG(info) << "LSFG: capture frame generation " << (lsfg_enabled_now ? "enabled" : "disabled") << " mid-stream; requesting capture reinit";
      _lsfg_requested = lsfg_enabled_now;
      pacing_allow_above_refresh = lsfg_enabled_now;
      return capture_e::reinit;
    }

    const bool lsfg_wanted = _lsfg_requested && !_lsfg_failed;
    timeout = effective_wgc_timeout(timeout, _config.framerate, lsfg_wanted);

    auto capture_status = _ipc_session->wait_for_frame(timeout);
    const bool have_new_frame = capture_status == capture_e::ok;
    if (!have_new_frame) {
      if (capture_status != capture_e::timeout) {
        return capture_status;
      }
      if (wgc_stall_requires_dxgi_fallback(_wgc_stall_start, _last_secure_desktop_probe)) {
        BOOST_LOG(info) << "WGC frames stalled while the secure desktop is active; falling back to DXGI capture";
        note_wgc_desktop_switch();
        return capture_e::reinit;
      }
      if (_lsfg_variants.empty() || !_lsfg_variants[_lsfg_active_variant]->has_frame()) {
        if (is_wgc_constant_mode()) {
          return forward_cached_wgc_frame(_last_cached_frame, img_out);
        }
        return capture_status;
      }
      // LSFG keeps filling pacing slots between source frames; fall through.
    } else {
      _wgc_stall_start = {};

      // Peek the shared texture's descriptor without taking the keyed mutex.
      // The descriptor is fixed at session setup; reading it here lets us know
      // the capture format on the very first frame so complete_img() can run
      // before we ever touch the shared IPC mutex.
      D3D11_TEXTURE2D_DESC desc;
      if (!_ipc_session->peek_shared_texture_desc(desc)) {
        return capture_e::reinit;
      }

      if (capture_format == DXGI_FORMAT_UNKNOWN) {
        capture_format = desc.Format;
        BOOST_LOG(info) << "Capture format [" << dxgi_format_to_string(capture_format) << ']';
      }

      // Display enumeration can race with mode changes and produce mismatched
      // image pool and desktop texture sizes. Detect that early.
      if (desc.Width != width_before_rotation || desc.Height != height_before_rotation) {
        BOOST_LOG(info) << "Capture size changed ["sv << width_before_rotation << 'x' << height_before_rotation << " -> "sv << desc.Width << 'x' << desc.Height << ']';
        return capture_e::reinit;
      }

      // The capture format can change on the fly; if so, reinit to refresh the
      // image pool and detection state.
      if (capture_format != desc.Format) {
        BOOST_LOG(info) << "Capture format changed ["sv << dxgi_format_to_string(capture_format) << " -> "sv << dxgi_format_to_string(desc.Format) << ']';
        return capture_e::reinit;
      }

      // Hot-apply mid-stream: rebuilding the prebuilt quality set occurs only when
      // the user's base settings change. Adaptive changes below select an existing
      // member of the set at an output boundary and never allocate in that path.
      if (!_lsfg_variants.empty()) {
        const auto desired_flow_scale = resolve_lsfg_base_flow_percent(_config.width, _config.height) / 100.0f;
        const auto desired_performance_mode = config::video.lsfg.performance_mode;
        if (desired_flow_scale != _lsfg_flow_scale || desired_performance_mode != _lsfg_performance_mode) {
          BOOST_LOG(info) << "LSFG: base pipeline options changed mid-stream; rebuilding adaptive variant set";
          _lsfg_variants.clear();
          _lsfg_active_variant = 0;
          _lsfg_over_budget_samples = 0;
          _lsfg_recovery_samples = 0;
          _lsfg_last_generated_gpu_ms = -1.0;
        } else {
          for (auto &variant : _lsfg_variants) {
            variant->update_live_options(config::video.lsfg.max_multiplier);
          }
        }
      }

      // Create the user's configured pipeline followed by exactly two lower-cost
      // variants. For a base 85% quality profile this yields 60% quality, then
      // 60% performance. If the base already uses performance shaders, the final
      // step instead uses 50% performance.
      if (lsfg_wanted && _lsfg_variants.empty()) {
        lsfg_framegen_t::options_t base_options;
        const int base_flow_percent = resolve_lsfg_base_flow_percent(_config.width, _config.height);
        base_options.flow_scale = base_flow_percent / 100.0f;
        base_options.max_multiplier = config::video.lsfg.max_multiplier;
        base_options.performance_mode = config::video.lsfg.performance_mode;
        if (_config.framerate > 0) {
          base_options.target_fps = _config.framerate;
        } else if (_config.framerateX100 > 0) {
          base_options.target_fps = _config.framerateX100 / 100.0;
        }

        std::array<lsfg_framegen_t::options_t, 3> variant_options {};
        variant_options[0] = base_options;
        variant_options[1] = base_options;
        variant_options[1].flow_scale = std::max(25, base_flow_percent - 25) / 100.0f;
        variant_options[2] = variant_options[1];
        if (base_options.performance_mode) {
          variant_options[2].flow_scale = std::max(25, base_flow_percent - 35) / 100.0f;
        } else {
          variant_options[2].performance_mode = true;
        }

        std::vector<std::unique_ptr<lsfg_framegen_t>> variants;
        variants.reserve(variant_options.size());
        for (const auto &options : variant_options) {
          auto variant = lsfg_framegen_t::create(device.get(), device_ctx.get(), desc.Width, desc.Height, capture_format, options);
          if (!variant) {
            break;
          }
          variants.push_back(std::move(variant));
        }
        if (!variants.empty()) {
          _lsfg_variants = std::move(variants);
          _lsfg_active_variant = 0;
          _lsfg_over_budget_samples = 0;
          _lsfg_recovery_samples = 0;
          _lsfg_last_generated_gpu_ms = -1.0;
          _lsfg_flow_scale = base_options.flow_scale;
          _lsfg_performance_mode = base_options.performance_mode;
          if (_lsfg_variants.size() == variant_options.size()) {
            BOOST_LOG(info) << "LSFG: prebuilt " << _lsfg_variants.size() << " adaptive quality variants";
          } else {
            BOOST_LOG(warning) << "LSFG: only " << _lsfg_variants.size()
                               << " of " << variant_options.size()
                               << " adaptive quality variants could be allocated";
          }
        } else {
          _lsfg_failed = true;
        }
      }
    }

    if (!_lsfg_variants.empty()) {
      return snapshot_lsfg(pull_free_image_cb, img_out, have_new_frame);
    }

    // Pull a free image from the pool before touching the shared IPC keyed
    // mutex. The encoder image pool can block under pressure; holding the
    // shared mutex during that wait stalls the WGC helper producer.
    std::shared_ptr<platf::img_t> img;
    if (!pull_free_image_cb(img)) {
      return capture_e::interrupted;
    }

    auto d3d_img = std::static_pointer_cast<img_d3d_t>(img);
    if (complete_img(d3d_img.get(), false)) {
      return capture_e::error;
    }

    // Acquire the encoder image's keyed mutex BEFORE the shared IPC mutex.
    // The encoder may still be sampling the previous frame; if we held the
    // shared IPC mutex during this wait, the helper's delivery thread would
    // block, its WGC frame pool would back up, and the compositor would drop
    // frames at the source. Taking the encoder mutex first keeps the shared
    // IPC mutex critical section bounded to the GPU-copy submission only.
    const auto capture_mutex_wait_start = std::chrono::steady_clock::now();
    HRESULT status = d3d_img->capture_mutex->AcquireSync(0, 3000);
    const auto capture_mutex_wait = std::chrono::steady_clock::now() - capture_mutex_wait_start;
    if (status == WAIT_ABANDONED) {
      BOOST_LOG(error) << "Capture texture keyed mutex was abandoned; continuing with lock held";
    } else if (status != S_OK) {
      BOOST_LOG(error) << "Failed to lock capture texture [0x"sv << util::hex(status).to_string_view() << ']';
      return capture_e::error;
    }

    auto release_capture_mutex = util::fail_guard([&]() {
      const HRESULT release_status = d3d_img->capture_mutex->ReleaseSync(0);
      if (FAILED(release_status)) {
        BOOST_LOG(warning) << "Failed to release capture texture mutex [0x"sv << util::hex(release_status).to_string_view() << ']';
      }
    });

    texture2d_t src;
    uint64_t frame_qpc = 0;
    winrt::com_ptr<ID3D11Texture2D> gpu_tex;
    capture_status = _ipc_session->lock_frame(gpu_tex, frame_qpc);
    if (capture_status != capture_e::ok) {
      return capture_status;
    }
    gpu_tex.copy_to(&src);
    _frame_locked = true;

    const auto host_processing_timestamp = std::chrono::steady_clock::now();
    auto frame_timestamp = host_processing_timestamp - qpc_time_difference(qpc_counter(), frame_qpc);

    // The IPC texture is a single mutable helper-owned surface. Snapshot it into
    // this pool-owned texture so queued encoder frames remain stable.
    const auto copy_start = std::chrono::steady_clock::now();
    device_ctx->CopyResource(d3d_img->capture_texture.get(), src.get());
    const auto copy_submit = std::chrono::steady_clock::now() - copy_start;
    d3d_img->blank = false;

    // Release the shared IPC mutex immediately after queueing the copy. The
    // GPU work is fenced through the keyed-mutex / encoder pipeline, so the
    // helper is free to publish the next frame as soon as we drop this mutex.
    _ipc_session->release();
    _frame_locked = false;

    const auto copy_count = g_wgc_snapshot_copies.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto capture_mutex_wait_ms = std::chrono::duration<double, std::milli>(capture_mutex_wait).count();
    const auto copy_submit_ms = std::chrono::duration<double, std::milli>(copy_submit).count();
    const bool slow_lock = capture_mutex_wait_ms > 1.0;
    const bool slow_copy = copy_submit_ms > 1.0;
    if (slow_lock) {
      g_wgc_slow_snapshot_locks.fetch_add(1, std::memory_order_relaxed);
    }
    if (slow_copy) {
      g_wgc_slow_snapshot_copies.fetch_add(1, std::memory_order_relaxed);
    }
    if (copy_count == 1 || copy_count % 600 == 0 || slow_lock || slow_copy) {
      BOOST_LOG(debug) << "WGC snapshot copy timing: frame=" << copy_count
                       << " capture_mutex_wait_ms=" << capture_mutex_wait_ms
                       << " copy_submit_ms=" << copy_submit_ms
                       << " slow_locks=" << g_wgc_slow_snapshot_locks.load(std::memory_order_relaxed)
                       << " slow_copies=" << g_wgc_slow_snapshot_copies.load(std::memory_order_relaxed);
    }

    img->frame_timestamp = frame_timestamp;
    img->host_processing_timestamp = host_processing_timestamp;
    // Keep WGC's QPC-derived timestamp for RTP/client accounting, but do not
    // use compositor timestamp jitter as the capture-loop sleep anchor.
    img->capture_pacing_timestamp = host_processing_timestamp;
    img_out = img;
    _last_cached_frame = img;

    return capture_e::ok;
  }

  capture_e display_wgc_ipc_vram_t::snapshot_lsfg(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, bool have_new_frame) {
    if (_lsfg_variants.empty() || _lsfg_active_variant >= _lsfg_variants.size()) {
      return capture_e::error;
    }

    // Leave one quarter of every requested output interval for the source copy,
    // optical-flow pre-pass, encoder and scheduler. Neither query path flushes
    // nor waits for the GPU, so adaptation cannot introduce a capture hitch.
    auto &timed_variant = *_lsfg_variants[_lsfg_active_variant];
    const auto gpu_time = timed_variant.poll_generated_gpu_time();
    const auto frame_budget = timed_variant.target_frame_duration() * 3 / 4;
    if (gpu_time) {
      _lsfg_last_generated_gpu_ms = std::chrono::duration<double, std::milli>(*gpu_time).count();
    }
    const auto now = std::chrono::steady_clock::now();
    const bool timing_backlogged = timed_variant.generated_gpu_timing_backlogged();
    const bool gpu_work_overdue = timed_variant.gpu_work_overdue(now, frame_budget);
    const bool adaptive_quality = config::video.lsfg.adaptive_quality;

    if (!adaptive_quality) {
      _lsfg_active_variant = 0;
      _lsfg_over_budget_samples = 0;
      _lsfg_recovery_samples = 0;
    } else if (gpu_work_overdue || timing_backlogged || (gpu_time && *gpu_time >= frame_budget)) {
      _lsfg_recovery_samples = 0;
      if (++_lsfg_over_budget_samples >= 4 && _lsfg_active_variant + 1 < _lsfg_variants.size()) {
        ++_lsfg_active_variant;
        _lsfg_over_budget_samples = 0;
        BOOST_LOG(warning) << "LSFG: generated-frame GPU budget exceeded; selecting adaptive quality step "
                           << _lsfg_active_variant;
      }
    } else if (gpu_time) {
      _lsfg_over_budget_samples = 0;
      // Quality recovery is deliberately much slower than degradation. Requiring
      // 240 comfortably-under-budget generated frames avoids oscillation when a
      // game alternates between light and heavy scenes.
      if (_lsfg_active_variant > 0 && *gpu_time * 2 <= frame_budget) {
        if (++_lsfg_recovery_samples >= 240) {
          --_lsfg_active_variant;
          _lsfg_recovery_samples = 0;
          BOOST_LOG(info) << "LSFG: sustained GPU headroom; restoring adaptive quality step "
                          << _lsfg_active_variant;
        }
      } else {
        _lsfg_recovery_samples = 0;
      }
    }

    if (now - _lsfg_adaptation_log_at >= 10s) {
      BOOST_LOG(debug) << "LSFG adaptive quality: enabled=" << (adaptive_quality ? "yes" : "no")
                       << " step=" << _lsfg_active_variant
                       << " variants=" << _lsfg_variants.size()
                       << " tail_gpu_ms=" << _lsfg_last_generated_gpu_ms
                       << " pending_fences=" << timed_variant.pending_gpu_work_fences()
                       << " work_overdue=" << (gpu_work_overdue ? "yes" : "no")
                       << " timing_backlogged=" << (timing_backlogged ? "yes" : "no")
                       << " over_budget_samples=" << _lsfg_over_budget_samples;
      _lsfg_adaptation_log_at = now;
    }

    auto &lsfg = *_lsfg_variants[_lsfg_active_variant];

    // Stage the fresh helper frame while its keyed mutex is held, then release it
    // immediately. Interpolated slots commit the capture only after rendering so
    // the optical-flow history still matches the source pair used by this slot.
    uint64_t staged_frame_qpc = 0;
    bool staged_capture = false;
    if (have_new_frame) {
      winrt::com_ptr<ID3D11Texture2D> gpu_tex;
      auto capture_status = _ipc_session->lock_frame(gpu_tex, staged_frame_qpc);
      if (capture_status != capture_e::ok) {
        return capture_status;
      }
      _frame_locked = true;
      // Each prebuilt variant receives the same source stream. Keeping their
      // optical-flow histories warm makes a later quality change a selection,
      // rather than a resource build or cold-start on the capture thread.
      for (auto &variant : _lsfg_variants) {
        variant->stage_capture(gpu_tex.get());
      }
      _ipc_session->release();
      _frame_locked = false;
      staged_capture = true;
    }

    auto commit_staged_capture = [&]() {
      if (staged_capture) {
        for (auto &variant : _lsfg_variants) {
          variant->commit_capture(staged_frame_qpc);
        }
        staged_capture = false;
      }
    };

    if (staged_capture && !lsfg.defer_capture_commit()) {
      commit_staged_capture();
    }

    if (!lsfg.has_frame()) {
      commit_staged_capture();
      return capture_e::timeout;
    }

    const auto output_slot = pacing_slot_timestamp.value_or(std::chrono::steady_clock::now());
    float phase = 0.0f;
    const bool generate = lsfg.want_generated(output_slot, phase);
    // Advance every inactive variant's presentation clock at this same slot.
    // This preserves phase continuity when an adaptive selection is made.
    for (std::size_t i = 0; i < _lsfg_variants.size(); ++i) {
      if (i != _lsfg_active_variant) {
        float unused_phase = 0.0f;
        _lsfg_variants[i]->want_generated(output_slot, unused_phase);
      }
    }

    // Requested FPS is a ceiling, not a floor: with nothing new to show, return
    // no_new_content (routine pacing slack, distinct from a timeout stall) rather
    // than pushing a duplicate, skipping the image-pool/mutex/copy work below.
    // passthrough_pending still forces one push after a run of generated frames so
    // the client settles on the true captured frame instead of a stale extrapolation.
    const bool passthrough_pending = lsfg.has_new_passthrough_frame();
    // A fresh source frame can be intentionally held while LSFG's delayed
    // presentation timeline is still showing the preceding pair. Do not let
    // arrival alone force that frame through early, or fractional conversions
    // such as 33 -> 60 reintroduce the cadence jitter the timeline removes.
    if (!generate && !passthrough_pending) {
      commit_staged_capture();
      if (adaptive_quality) {
        lsfg.submit_gpu_work_fence();
      }
      return capture_e::no_new_content;
    }

    std::shared_ptr<platf::img_t> img;
    if (!pull_free_image_cb(img)) {
      commit_staged_capture();
      return capture_e::interrupted;
    }

    auto d3d_img = std::static_pointer_cast<img_d3d_t>(img);
    if (complete_img(d3d_img.get(), false)) {
      commit_staged_capture();
      return capture_e::error;
    }

    HRESULT status = d3d_img->capture_mutex->AcquireSync(0, 3000);
    if (status == WAIT_ABANDONED) {
      BOOST_LOG(error) << "Capture texture keyed mutex was abandoned; continuing with lock held";
    } else if (status != S_OK) {
      BOOST_LOG(error) << "Failed to lock capture texture [0x"sv << util::hex(status).to_string_view() << ']';
      commit_staged_capture();
      return capture_e::error;
    }

    auto release_capture_mutex = util::fail_guard([&]() {
      const HRESULT release_status = d3d_img->capture_mutex->ReleaseSync(0);
      if (FAILED(release_status)) {
        BOOST_LOG(warning) << "Failed to release capture texture mutex [0x"sv << util::hex(release_status).to_string_view() << ']';
      }
    });

    bool wrote_generated = false;
    if (generate) {
      wrote_generated = lsfg.render_generated(phase, d3d_img->capture_rt.get(), d3d_img->width, d3d_img->height);
    }
    if (wrote_generated) {
      lsfg.mark_generated_shown();
    } else {
      device_ctx->CopyResource(d3d_img->capture_texture.get(), lsfg.passthrough_texture());
      lsfg.mark_passthrough_shown();
    }
    d3d_img->blank = false;

    img->frame_timestamp = output_slot;
    img->host_processing_timestamp = output_slot;
    img->capture_pacing_timestamp = output_slot;
    img_out = img;
    _last_cached_frame = img;

    commit_staged_capture();
    if (adaptive_quality) {
      // This comes after staged source commits, so its completion represents the
      // full LSFG GPU workload for this slot, not only the generated-frame tail.
      lsfg.submit_gpu_work_fence();
    }
    return capture_e::ok;
  }

  capture_e display_wgc_ipc_vram_t::acquire_next_frame(std::chrono::milliseconds timeout, texture2d_t &src, uint64_t &frame_qpc, bool cursor_visible) {
    if (!_ipc_session) {
      return capture_e::error;
    }

    winrt::com_ptr<ID3D11Texture2D> gpu_tex;
    auto status = _ipc_session->acquire(effective_wgc_timeout(timeout, _config.framerate), gpu_tex, frame_qpc);

    if (status != capture_e::ok) {
      return status;
    }

    gpu_tex.copy_to(&src);
    _frame_locked = true;

    return capture_e::ok;
  }

  capture_e display_wgc_ipc_vram_t::release_snapshot() {
    if (_ipc_session && _frame_locked) {
      _ipc_session->release();
      _frame_locked = false;
    }
    return capture_e::ok;
  }

  int display_wgc_ipc_vram_t::dummy_img(platf::img_t *img_base) {
    // Use the base class implementation which creates a blank GPU texture directly,
    // avoiding Desktop Duplication which may fail on headless/disconnected sessions.
    return complete_img(img_base, true);
  }

  std::shared_ptr<display_t> display_wgc_ipc_vram_t::create(const ::video::config_t &config, const std::string &display_name) {
    if (auto fallback_state = get_wgc_dxgi_fallback_state()) {
      log_wgc_dxgi_fallback_reason("VRAM", *fallback_state);
      adapter_luid_override_guard guard(get_last_wgc_adapter_luid());
      auto disp = std::make_shared<temp_dxgi_vram_t>();
      if (!disp->init(config, display_name)) {
        return disp;
      }
    } else {
      // Secure desktop not active, use WGC IPC
      BOOST_LOG(debug) << "Using WGC IPC implementation (VRAM)";
      auto disp = std::make_shared<display_wgc_ipc_vram_t>();
      if (!disp->init(config, display_name)) {
        return disp;
      }
    }

    return nullptr;
  }

  display_wgc_ipc_ram_t::display_wgc_ipc_ram_t() = default;

  display_wgc_ipc_ram_t::~display_wgc_ipc_ram_t() = default;

  int display_wgc_ipc_ram_t::init(const ::video::config_t &config, const std::string &display_name) {
    // Save config for later use
    _config = config;
    _display_name = display_name;

    // Initialize the base display class
    if (display_base_t::init(config, display_name, true /* skip_dd_test: WGC doesn't use Desktop Duplication */)) {
      return -1;
    }

    // Initialize capture format to unknown - will be determined from first frame
    capture_format = DXGI_FORMAT_UNKNOWN;

    if (config::video.lsfg.enabled) {
      BOOST_LOG(info) << "LSFG capture frame generation requires a hardware encoder; continuing without it";
    }

    // Note: WGC captures at monitor native resolution, not the requested config resolution.
    // The display helper handles resolution changes before capture starts if needed.
    // We use the dimensions set by display_base_t::init() which reflect the actual monitor size.

    const bool advanced_color_capture = is_hdr();

    // Create session
    _ipc_session = std::make_unique<ipc_session_t>();
    if (_ipc_session->init(config, display_name, device.get(), advanced_color_capture)) {
      return -1;
    }

    return 0;
  }

  capture_e display_wgc_ipc_ram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    if (!_ipc_session) {
      return capture_e::error;
    }

    if (_ipc_session->should_swap_to_dxgi()) {
      return capture_e::reinit;
    }

    // If the helper process crashed or was terminated forcefully by the user, we will re-initialize it.
    if (_ipc_session->should_reinit()) {
      return capture_e::reinit;
    }

    _ipc_session->initialize_if_needed();
    if (!_ipc_session->is_initialized()) {
      BOOST_LOG(warning) << "WGC IPC helper failed to initialize; requesting capture reinit.";
      return capture_e::reinit;
    }

    winrt::com_ptr<ID3D11Texture2D> gpu_tex;
    uint64_t frame_qpc = 0;
    timeout = effective_wgc_timeout(timeout, _config.framerate);
    auto status = _ipc_session->acquire(timeout, gpu_tex, frame_qpc);

    if (status != capture_e::ok) {
      if (status == capture_e::timeout) {
        if (wgc_stall_requires_dxgi_fallback(_wgc_stall_start, _last_secure_desktop_probe)) {
          BOOST_LOG(info) << "WGC frames stalled while the secure desktop is active; falling back to DXGI capture";
          note_wgc_desktop_switch();
          return capture_e::reinit;
        }
        if (is_wgc_constant_mode()) {
          return forward_cached_wgc_frame(_last_cached_frame, img_out);
        }
      }
      // For the default mode just return the capture status on timeouts.
      return status;
    }
    _wgc_stall_start = {};

    // Get description of the captured texture
    D3D11_TEXTURE2D_DESC desc;
    gpu_tex->GetDesc(&desc);

    // If we don't know the capture format yet, grab it from this texture
    if (capture_format == DXGI_FORMAT_UNKNOWN) {
      capture_format = desc.Format;
      BOOST_LOG(info) << "Capture format [" << dxgi_format_to_string(capture_format) << ']';
    }

    // Check for size changes - use width_before_rotation/height_before_rotation since WGC
    // captures textures in unrotated physical pixel dimensions, same as VRAM path
    if (desc.Width != width_before_rotation || desc.Height != height_before_rotation) {
      BOOST_LOG(info) << "Capture size changed [" << width_before_rotation << 'x' << height_before_rotation << " -> " << desc.Width << 'x' << desc.Height << ']';
      _ipc_session->release();
      return capture_e::reinit;
    }

    // Check for format changes
    if (capture_format != desc.Format) {
      BOOST_LOG(info) << "Capture format changed [" << dxgi_format_to_string(capture_format) << " -> " << dxgi_format_to_string(desc.Format) << ']';
      _ipc_session->release();
      return capture_e::reinit;
    }

    // Create or recreate staging texture if needed
    // Use unrotated dimensions to match the captured texture size
    if (!texture ||
        width_before_rotation != _last_width ||
        height_before_rotation != _last_height ||
        capture_format != _last_format) {
      D3D11_TEXTURE2D_DESC t {};
      t.Width = width_before_rotation;
      t.Height = height_before_rotation;
      t.Format = capture_format;
      t.ArraySize = 1;
      t.MipLevels = 1;
      t.SampleDesc = {1, 0};
      t.Usage = D3D11_USAGE_STAGING;
      t.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

      auto hr = device->CreateTexture2D(&t, nullptr, &texture);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "[display_wgc_ipc_ram_t] Failed to create staging texture: " << hr;
        _ipc_session->release();
        return capture_e::error;
      }

      _last_width = width_before_rotation;
      _last_height = height_before_rotation;
      _last_format = capture_format;

      BOOST_LOG(info) << "[display_wgc_ipc_ram_t] Created staging texture: "
                      << width_before_rotation << "x" << height_before_rotation << ", format: " << capture_format;
    }

    // Copy from shared texture to staging texture (queues GPU work)
    device_ctx->CopyResource(texture.get(), gpu_tex.get());

    // CRITICAL: Release the keyed mutex BEFORE blocking on Map()
    // The helper needs the mutex to write the next frame while we're reading this one
    _ipc_session->release();

    // Get a free image from the pool
    if (!pull_free_image_cb(img_out)) {
      return capture_e::interrupted;
    }

    auto img = img_out.get();

    // If we don't know the final capture format yet, encode a dummy image
    if (capture_format == DXGI_FORMAT_UNKNOWN) {
      if (dummy_img(img)) {
        return capture_e::error;
      }
    } else {
      // Map the staging texture for CPU access (blocks until GPU copy completes)
      auto hr = device_ctx->Map(texture.get(), 0, D3D11_MAP_READ, 0, &img_info);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "[display_wgc_ipc_ram_t] Failed to map staging texture: " << hr;
        return capture_e::error;
      }

      // Now that we know the capture format, we can finish creating the image
      if (complete_img(img, false)) {
        device_ctx->Unmap(texture.get(), 0);
        img_info.pData = nullptr;
        return capture_e::error;
      }

      // Copy data - use height_before_rotation since WGC captures unrotated texture
      std::copy_n((std::uint8_t *) img_info.pData, height_before_rotation * img_info.RowPitch, img->data);

      // Unmap the staging texture to allow GPU access again
      device_ctx->Unmap(texture.get(), 0);
      img_info.pData = nullptr;
    }

    // Set frame timestamp
    const auto host_processing_timestamp = std::chrono::steady_clock::now();
    auto frame_timestamp = host_processing_timestamp - qpc_time_difference(qpc_counter(), frame_qpc);
    img->frame_timestamp = frame_timestamp;
    img->host_processing_timestamp = host_processing_timestamp;
    img->capture_pacing_timestamp = host_processing_timestamp;
    _last_cached_frame = img_out;

    return capture_e::ok;
  }

  capture_e display_wgc_ipc_ram_t::release_snapshot() {
    // Not used in RAM path since we handle everything in snapshot()
    return capture_e::ok;
  }

  int display_wgc_ipc_ram_t::dummy_img(platf::img_t *img_base) {
    // Use the base class implementation directly,
    // avoiding Desktop Duplication which may fail on headless/disconnected sessions.
    return display_ram_t::dummy_img(img_base);
  }

  std::shared_ptr<display_t> display_wgc_ipc_ram_t::create(const ::video::config_t &config, const std::string &display_name) {
    if (auto fallback_state = get_wgc_dxgi_fallback_state()) {
      log_wgc_dxgi_fallback_reason("RAM", *fallback_state);
      adapter_luid_override_guard guard(get_last_wgc_adapter_luid());
      auto disp = std::make_shared<temp_dxgi_ram_t>();
      if (!disp->init(config, display_name)) {
        return disp;
      }
    } else {
      // Secure desktop not active, use WGC IPC
      BOOST_LOG(debug) << "Using WGC IPC implementation (RAM)";
      auto disp = std::make_shared<display_wgc_ipc_ram_t>();
      if (!disp->init(config, display_name)) {
        return disp;
      }
    }

    return nullptr;
  }

  capture_e temp_dxgi_vram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    // Check periodically if secure desktop is still active
    if (auto now = std::chrono::steady_clock::now(); now - _last_check_time >= CHECK_INTERVAL) {
      _last_check_time = now;
      const bool secure_desktop_active = platf::dxgi::is_secure_desktop_active();
      if (!secure_desktop_active && !recent_wgc_desktop_switch_grace_active()) {
        BOOST_LOG(debug) << "DXGI Capture is no longer necessary, swapping back to WGC!";
        return capture_e::reinit;
      }
    }

    // Call parent DXGI duplication implementation
    return display_ddup_vram_t::snapshot(pull_free_image_cb, img_out, timeout, cursor_visible);
  }

  capture_e temp_dxgi_ram_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    // Check periodically if secure desktop is still active
    if (auto now = std::chrono::steady_clock::now(); now - _last_check_time >= CHECK_INTERVAL) {
      _last_check_time = now;
      const bool secure_desktop_active = platf::dxgi::is_secure_desktop_active();
      if (!secure_desktop_active && !recent_wgc_desktop_switch_grace_active()) {
        BOOST_LOG(debug) << "DXGI Capture is no longer necessary, swapping back to WGC!";
        return capture_e::reinit;
      }
    }

    // Call parent DXGI duplication implementation
    return display_ddup_ram_t::snapshot(pull_free_image_cb, img_out, timeout, cursor_visible);
  }

}  // namespace platf::dxgi
