/**
 * @file src/platform/windows/lsfg_framegen.cpp
 * @brief Host-side Lossless Scaling frame generation (LSFG) for the Windows capture pipeline.
 *
 * D3D11 port of the LSFG frame-generation pipeline from lsfg-vk
 * (https://github.com/PancakeTAS/lsfg-vk, GPL-3.0-or-later).
 *
 * The compute shaders are loaded at runtime as DXBC blobs from the RT_RCDATA
 * resources of the user's own Lossless Scaling installation: ids 255-279 are the
 * "quality" shader set, 280-302 the "performance" set (id+23 for ids 257-279;
 * mipmap/generate at 255/256 are shared between both -- see build_pipeline()).
 * Register assignment is b0.., s0.., t0.., u0.. in the same order lsfg-vk binds
 * descriptors (buffers, samplers, sampled images, storage images).
 */
// platform includes
#include <winsock2.h>

#include <d3d11.h>
#include <d3dcompiler.h>

// standard includes
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// local includes
#include "src/config.h"
#include "src/logging.h"
#include "src/platform/windows/lossless_scaling_paths.h"
#include "src/platform/windows/lsfg_framegen.h"
#include "src/platform/windows/misc.h"
#include "src/utility.h"

// platform includes (after local so winsock ordering stays intact)
#include <winrt/base.h>

namespace platf::dxgi {

  using namespace std::chrono_literals;

  namespace {

    using winrt::com_ptr;

    struct ext_t {
      std::uint32_t w;
      std::uint32_t h;
    };

    ext_t shift_ext(ext_t e, std::uint32_t s) {
      return {std::max(e.w >> s, 1u), std::max(e.h >> s, 1u)};
    }

    ext_t add_shift(ext_t e, std::uint32_t a, std::uint32_t s) {
      return {(e.w + a) >> s, (e.h + a) >> s};
    }

    ext_t g8(ext_t e) {
      return add_shift(e, 7, 3);
    }

    struct tex_t {
      com_ptr<ID3D11Texture2D> tex;
      com_ptr<ID3D11ShaderResourceView> srv;
      com_ptr<ID3D11UnorderedAccessView> uav;
      ext_t extent {};
    };

    bool make_tex2d(ID3D11Device *device, ext_t extent, DXGI_FORMAT format, bool uav, const void *init, std::uint32_t init_pitch, tex_t &out) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = extent.w;
      desc.Height = extent.h;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = format;
      desc.SampleDesc = {1, 0};
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0);

      D3D11_SUBRESOURCE_DATA init_data {};
      init_data.pSysMem = init;
      init_data.SysMemPitch = init_pitch;

      out = {};
      out.extent = extent;
      auto status = device->CreateTexture2D(&desc, init ? &init_data : nullptr, out.tex.put());
      if (FAILED(status)) {
        BOOST_LOG(error) << "LSFG: failed to create texture [0x" << util::hex(status).to_string_view() << ']';
        return false;
      }
      status = device->CreateShaderResourceView(out.tex.get(), nullptr, out.srv.put());
      if (FAILED(status)) {
        BOOST_LOG(error) << "LSFG: failed to create SRV [0x" << util::hex(status).to_string_view() << ']';
        return false;
      }
      if (uav) {
        status = device->CreateUnorderedAccessView(out.tex.get(), nullptr, out.uav.put());
        if (FAILED(status)) {
          BOOST_LOG(error) << "LSFG: failed to create UAV [0x" << util::hex(status).to_string_view() << ']';
          return false;
        }
      }
      return true;
    }

    // mirrors lsfg-vk's ConstantBuffer (48 bytes)
    struct cb_data_t {
      std::uint32_t input_offset[2];
      std::uint32_t first_iter;
      std::uint32_t first_iter_s;
      std::uint32_t advanced_color_kind;
      std::uint32_t hdr_support;
      float resolution_inv_scale;
      float timestamp;
      float ui_threshold;
      std::uint32_t pad[3];
    };

    static_assert(sizeof(cb_data_t) == 48, "LSFG constant buffer must stay 48 bytes");

    cb_data_t make_cb_data(float timestamp, float inv_flow, bool hdr) {
      cb_data_t data {};
      data.hdr_support = hdr ? 1u : 0u;
      data.resolution_inv_scale = inv_flow;
      data.timestamp = timestamp;
      data.ui_threshold = 0.5f;
      return data;
    }

    /// Fixed interpolation phase for output `index` of `total` (multiplier mode).
    float phase_of(std::size_t index, std::size_t total) {
      return static_cast<float>(index + 1) / static_cast<float>(total + 1);
    }

    com_ptr<ID3D11Buffer> make_cb(ID3D11Device *device, float timestamp, float inv_flow, bool hdr, bool updatable) {
      const auto data = make_cb_data(timestamp, inv_flow, hdr);
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(cb_data_t);
      // DEFAULT (not IMMUTABLE) when the phase changes per frame in adaptive
      // mode, so UpdateSubresource can rewrite the timestamp.
      desc.Usage = updatable ? D3D11_USAGE_DEFAULT : D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA init {};
      init.pSysMem = &data;
      com_ptr<ID3D11Buffer> buf;
      auto status = device->CreateBuffer(&desc, &init, buf.put());
      if (FAILED(status)) {
        BOOST_LOG(error) << "LSFG: failed to create constant buffer [0x" << util::hex(status).to_string_view() << ']';
      }
      return buf;
    }

    com_ptr<ID3D11SamplerState> make_sampler(ID3D11Device *device, D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE address, D3D11_COMPARISON_FUNC compare, bool border_white) {
      const float b = border_white ? 1.0f : 0.0f;
      D3D11_SAMPLER_DESC desc {};
      desc.Filter = filter;
      desc.AddressU = address;
      desc.AddressV = address;
      desc.AddressW = address;
      desc.ComparisonFunc = compare;
      desc.BorderColor[0] = b;
      desc.BorderColor[1] = b;
      desc.BorderColor[2] = b;
      desc.BorderColor[3] = b;
      desc.MaxLOD = D3D11_FLOAT32_MAX;
      com_ptr<ID3D11SamplerState> sampler;
      auto status = device->CreateSamplerState(&desc, sampler.put());
      if (FAILED(status)) {
        BOOST_LOG(error) << "LSFG: failed to create sampler [0x" << util::hex(status).to_string_view() << ']';
      }
      return sampler;
    }

    /// Load an RT_RCDATA resource from a DLL loaded as a data file.
    bool load_resource(HMODULE module, std::uint16_t id, std::vector<std::uint8_t> &out) {
      auto res = FindResourceW(module, MAKEINTRESOURCEW(id), MAKEINTRESOURCEW(10) /* RT_RCDATA */);
      if (!res) {
        return false;
      }
      const auto size = SizeofResource(module, res);
      auto handle = LoadResource(module, res);
      if (!handle || size == 0) {
        return false;
      }
      auto *ptr = LockResource(handle);
      if (!ptr) {
        return false;
      }
      out.assign(static_cast<const std::uint8_t *>(ptr), static_cast<const std::uint8_t *>(ptr) + size);
      return true;
    }

    /// One compute dispatch: shader + fully resolved bindings.
    struct disp_t {
      com_ptr<ID3D11ComputeShader> cs;
      std::vector<com_ptr<ID3D11ShaderResourceView>> srvs;
      std::vector<com_ptr<ID3D11UnorderedAccessView>> uavs;
      std::vector<com_ptr<ID3D11SamplerState>> samplers;
      com_ptr<ID3D11Buffer> cb;
      ext_t groups {};
    };

    /// One logical pass; `variants[fidx % variants.size()]` is the dispatch list for a frame.
    struct step_t {
      std::vector<std::vector<disp_t>> variants;
    };

    void exec_disp(ID3D11DeviceContext *ctx, const disp_t &d) {
      std::array<ID3D11ShaderResourceView *, 16> srvs {};
      std::array<ID3D11UnorderedAccessView *, 8> uavs {};
      std::array<ID3D11SamplerState *, 4> samplers {};
      const auto srv_count = static_cast<UINT>(d.srvs.size());
      const auto uav_count = static_cast<UINT>(d.uavs.size());
      const auto sampler_count = static_cast<UINT>(d.samplers.size());
      for (UINT i = 0; i < srv_count; ++i) {
        srvs[i] = d.srvs[i].get();
      }
      for (UINT i = 0; i < uav_count; ++i) {
        uavs[i] = d.uavs[i].get();
      }
      for (UINT i = 0; i < sampler_count; ++i) {
        samplers[i] = d.samplers[i].get();
      }

      ctx->CSSetShader(d.cs.get(), nullptr, 0);
      if (d.cb) {
        ID3D11Buffer *cb = d.cb.get();
        ctx->CSSetConstantBuffers(0, 1, &cb);
      }
      if (sampler_count) {
        ctx->CSSetSamplers(0, sampler_count, samplers.data());
      }
      ctx->CSSetShaderResources(0, srv_count, srvs.data());
      ctx->CSSetUnorderedAccessViews(0, uav_count, uavs.data(), nullptr);
      ctx->Dispatch(d.groups.w, d.groups.h, 1);
      // unbind so the next dispatch can rebind these resources the other way
      std::array<ID3D11ShaderResourceView *, 16> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 8> null_uavs {};
      ctx->CSSetShaderResources(0, srv_count, null_srvs.data());
      ctx->CSSetUnorderedAccessViews(0, uav_count, null_uavs.data(), nullptr);
    }

    bool compile_shader(const char *src, std::size_t len, const char *entry, const char *target, std::vector<std::uint8_t> &out) {
      com_ptr<ID3DBlob> blob;
      com_ptr<ID3DBlob> errors;
      auto status = D3DCompile(src, len, nullptr, nullptr, nullptr, entry, target, 0, 0, blob.put(), errors.put());
      if (FAILED(status)) {
        if (errors) {
          BOOST_LOG(error) << "LSFG: shader compile error: " << std::string_view(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize());
        }
        return false;
      }
      out.assign(
        static_cast<const std::uint8_t *>(blob->GetBufferPointer()),
        static_cast<const std::uint8_t *>(blob->GetBufferPointer()) + blob->GetBufferSize()
      );
      return true;
    }

    // Fullscreen-triangle blit used to write the generated frame (RGBA8/RGBA16F
    // UAV output) into the encoder-shared capture texture, which may have a
    // different channel order (e.g. BGRA8) that CopyResource cannot bridge.
    constexpr char k_blit_shader_src[] = R"(
Texture2D tex : register(t0);
SamplerState samp : register(s0);

struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VSOut vs_main(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uv;
    return o;
}

float4 ps_main(VSOut i) : SV_Target {
    return float4(tex.Sample(samp, i.uv).rgb, 1.0);
}
)";

    class blitter_t {
    public:
      bool init(ID3D11Device *device) {
        std::vector<std::uint8_t> vs_code, ps_code;
        if (!compile_shader(k_blit_shader_src, sizeof(k_blit_shader_src) - 1, "vs_main", "vs_5_0", vs_code) ||
            !compile_shader(k_blit_shader_src, sizeof(k_blit_shader_src) - 1, "ps_main", "ps_5_0", ps_code)) {
          return false;
        }
        if (FAILED(device->CreateVertexShader(vs_code.data(), vs_code.size(), nullptr, _vs.put())) ||
            FAILED(device->CreatePixelShader(ps_code.data(), ps_code.size(), nullptr, _ps.put()))) {
          BOOST_LOG(error) << "LSFG: failed to create blit shaders";
          return false;
        }
        _sampler = make_sampler(device, D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_COMPARISON_ALWAYS, false);
        return static_cast<bool>(_sampler);
      }

      void blit(ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *src, ID3D11RenderTargetView *rtv, std::uint32_t width, std::uint32_t height) {
        D3D11_VIEWPORT viewport {};
        viewport.Width = static_cast<float>(width);
        viewport.Height = static_cast<float>(height);
        viewport.MaxDepth = 1.0f;

        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
        ctx->RSSetState(nullptr);
        ctx->RSSetViewports(1, &viewport);
        ctx->VSSetShader(_vs.get(), nullptr, 0);
        ctx->PSSetShader(_ps.get(), nullptr, 0);
        ID3D11SamplerState *sampler = _sampler.get();
        ctx->PSSetSamplers(0, 1, &sampler);
        ctx->PSSetShaderResources(0, 1, &src);
        ctx->Draw(3, 0);
        ID3D11ShaderResourceView *null_srv = nullptr;
        ctx->PSSetShaderResources(0, 1, &null_srv);
        ID3D11RenderTargetView *null_rtv = nullptr;
        ctx->OMSetRenderTargets(1, &null_rtv, nullptr);
      }

    private:
      com_ptr<ID3D11VertexShader> _vs;
      com_ptr<ID3D11PixelShader> _ps;
      com_ptr<ID3D11SamplerState> _sampler;
    };

  }  // namespace

  struct lsfg_framegen_t::impl_t {
    com_ptr<ID3D11Device> device;
    com_ptr<ID3D11DeviceContext> ctx;
    HMODULE lossless_module = nullptr;

    ext_t extent {};
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    bool hdr = false;
    float inv_flow = 1.0f;
    options_t options {};

    // Real-frame pool: distinct frame N lands in sources[N % pool_size]. pool_size is
    // 2 for queue_frames == 0 (today's ping-pong); options.queue_frames + 2 otherwise,
    // sized so a frame's slot can't be overwritten before its turn as the active pair.
    std::vector<tex_t> sources;
    std::size_t pool_size = 2;
    // render-owned copy of the latest captured frame (hashed for dedup, and
    // shown during pass-through so the stream never freezes on stale content)
    tex_t latest;

    std::vector<step_t> prepass;
    std::vector<std::vector<step_t>> passes;
    std::vector<tex_t> outputs;
    std::vector<com_ptr<ID3D11Buffer>> cbp;  // per-pass constant buffers, updatable for adaptive

    blitter_t blitter;

    // adaptive timing state (queue_frames == 0: extrapolate from the newest arrival)
    std::size_t frames_seen = 0;  // count of DISTINCT source frames consumed
    bool has_capture = false;
    std::uint64_t last_distinct_qpc = 0;
    std::chrono::steady_clock::time_point last_arrival {};
    std::chrono::nanoseconds src_interval = 16ms;  // EMA between distinct frames
    std::chrono::nanoseconds frame_dur = 16ms;  // one client-requested frame interval
    std::size_t consecutive_dirty_skips = 0;  // guards against an unreliable frame_dirty signal
    std::uint64_t commit_calls = 0;  // diagnostics: total commit_capture() calls
    std::uint64_t dirty_skip_total = 0;  // diagnostics: calls where frame_dirty was false

    // TEMPORARY diagnostics (added 2026-07-09): pacing-quality instrumentation for
    // the floor-slot bucketing / QPC-anchoring fixes in want_generated(). Logged
    // periodically by record_phase_tick(); safe to remove once real-hardware pacing
    // is confirmed good.
    bool diag_have_last_anchor = false;
    std::chrono::steady_clock::time_point diag_last_anchor {};
    std::uint64_t diag_ticks_total = 0;  // every want_generated() call that reached slot logic
    std::uint64_t diag_generated_count = 0;  // ticks that returned a generated phase
    std::uint64_t diag_passthrough_tick_count = 0;  // ticks that landed on the real-frame slot
    std::uint64_t diag_duplicate_slot_count = 0;  // consecutive ticks in the same interval, same slot
    long diag_last_slot_in_interval = -1;
    std::uint64_t diag_interval_ticks = 0;  // ticks seen so far in the current interval
    bool diag_interval_showed_real = false;
    std::uint64_t diag_intervals_closed = 0;
    std::uint64_t diag_single_tick_intervals = 0;  // intervals that only got one tick total
    std::uint64_t diag_intervals_real_shown = 0;  // intervals where the real frame slot was hit
    float diag_first_tick_elapsed_min = 1e9f;
    float diag_first_tick_elapsed_max = 0.0f;
    bool diag_have_last_log = false;
    std::chrono::steady_clock::time_point diag_last_log {};

    // True whenever latest_texture() may return something the caller hasn't shown yet
    // (a real arrival landed, or queued mode advanced the active pair) as a genuine
    // pass-through -- as opposed to a generated/extrapolated approximation of it. Lets
    // the caller settle on the true final frame once generation stops being due, instead
    // of leaving the last generated guess on screen forever. See mark_passthrough_shown().
    bool passthrough_dirty = true;

    // Queued-mode state (queue_frames >= 1: interpolate a confirmed pair). See
    // impl_t::try_advance_active_pair() for the mechanism.
    std::vector<std::uint64_t> slot_qpc;  // QPC of the real frame currently held in sources[i]
    bool buffered_ready = false;  // true once the first active pair has been established
    std::size_t active_idx = 0;  // 0-based arrival index of the active pair's newer frame
    std::chrono::steady_clock::time_point active_anchor {};
    std::chrono::nanoseconds active_interval = 16ms;

    // Optical-flow pre-pass output (mipmaps -> alpha0/alpha1 -> beta0/beta1). a1_out/
    // b1_imgs are a CONTINUOUSLY rolling temporal history the shaders expect fed one
    // real arrival at a time with no gaps (alpha1's per-level slot rotation, beta0's
    // 3-way temporal blend) -- so the pre-pass always runs on every arrival regardless
    // of queue_frames (see commit_capture()). In extrapolate mode gamma/delta/generate
    // read a1_out/b1_imgs directly, same as always. In queued mode they instead read
    // active_a1_out/active_b1_imgs, a separate stable working copy that's only
    // refreshed (via CopyResource from held_a1_out/held_b1_imgs) when the active pair
    // changes -- so it survives newer arrivals continuing to advance the rolling
    // history underneath it without needing to touch that history at all. held_a1_out/
    // held_b1_imgs is a pool_size-deep ring of full snapshots, one per pool slot,
    // refreshed every arrival right after its own pre-pass run.
    std::array<ext_t, 7> a1_extent {};
    std::array<std::vector<std::vector<tex_t>>, 7> a1_out;
    std::vector<tex_t> b1_imgs;
    std::array<std::vector<std::vector<tex_t>>, 7> active_a1_out;  // queue_frames >= 1 only
    std::vector<tex_t> active_b1_imgs;  // queue_frames >= 1 only
    std::vector<std::array<std::vector<std::vector<tex_t>>, 7>> held_a1_out;  // [pool slot]
    std::vector<std::vector<tex_t>> held_b1_imgs;  // [pool slot]

    ~impl_t() {
      if (lossless_module) {
        FreeLibrary(lossless_module);
      }
    }

    bool build_pipeline();
    void run_steps(const std::vector<step_t> &steps, std::size_t fidx);
    void try_advance_active_pair();
    void start_active_pair();
    void snapshot_flow_state(std::size_t slot);
    void restore_active_flow_state(std::size_t slot);
    void recompute_frame_dur();
    void record_phase_tick(std::chrono::steady_clock::time_point now, std::chrono::steady_clock::time_point anchor, float elapsed_s, long slot, long steps_i);
  };

  namespace {

    /// Helper used during pipeline construction.
    struct builder_t {
      ID3D11Device *device;
      std::map<std::uint16_t, com_ptr<ID3D11ComputeShader>> shaders;
      com_ptr<ID3D11SamplerState> bnb;  // linear, border black
      com_ptr<ID3D11SamplerState> bnw;  // linear, border white
      com_ptr<ID3D11SamplerState> eab;  // comparison linear, clamp, always
      bool ok = true;

      tex_t tex(ext_t extent, DXGI_FORMAT format) {
        tex_t t;
        if (!make_tex2d(device, extent, format, true, nullptr, 0, t)) {
          ok = false;
        }
        return t;
      }

      std::vector<tex_t> texes(std::size_t n, ext_t extent, DXGI_FORMAT format) {
        std::vector<tex_t> result;
        result.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
          result.push_back(tex(extent, format));
        }
        return result;
      }

      disp_t disp(std::uint16_t id, const std::vector<const tex_t *> &srcs, const std::vector<const tex_t *> &dsts, const std::vector<ID3D11SamplerState *> &samplers, ID3D11Buffer *cb, ext_t groups) {
        disp_t d;
        d.cs = shaders[id];
        d.srvs.reserve(srcs.size());
        for (const auto *t : srcs) {
          d.srvs.push_back(t->srv);
        }
        d.uavs.reserve(dsts.size());
        for (const auto *t : dsts) {
          if (!t->uav) {
            ok = false;
          }
          d.uavs.push_back(t->uav);
        }
        d.samplers.reserve(samplers.size());
        for (auto *s : samplers) {
          com_ptr<ID3D11SamplerState> sp;
          sp.copy_from(s);
          d.samplers.push_back(std::move(sp));
        }
        if (cb) {
          d.cb.copy_from(cb);
        }
        d.groups = groups;
        return d;
      }
    };

    std::vector<const tex_t *> refs(const std::vector<tex_t> &v) {
      std::vector<const tex_t *> result;
      result.reserve(v.size());
      for (const auto &t : v) {
        result.push_back(&t);
      }
      return result;
    }

    using a1_out_t = std::array<std::vector<std::vector<tex_t>>, 7>;

    /// Build a fresh set of plain (CopyResource-only) textures matching a1_out's shape.
    /// @param group Textures per temporal slot: 2*m, i.e. 4 quality / 2 performance.
    a1_out_t make_a1_out_like(builder_t &b, const std::array<ext_t, 7> &a1_extent, std::size_t group) {
      a1_out_t out;
      for (std::size_t lvl = 0; lvl < 7; ++lvl) {
        const std::size_t slots = (lvl == 0) ? 3 : 2;
        for (std::size_t s = 0; s < slots; ++s) {
          out[lvl].push_back(b.texes(group, a1_extent[lvl], DXGI_FORMAT_R8G8B8A8_UNORM));
        }
      }
      return out;
    }

    /// Build a fresh set of plain textures matching b1_imgs' shape (6 R8 pyramid levels).
    std::vector<tex_t> make_b1_imgs_like(builder_t &b, ext_t be) {
      std::vector<tex_t> out;
      for (std::uint32_t i = 0; i < 6; ++i) {
        out.push_back(b.tex(shift_ext(be, i), DXGI_FORMAT_R8_UNORM));
      }
      return out;
    }

    void copy_a1_out(ID3D11DeviceContext *ctx, const a1_out_t &src, a1_out_t &dst) {
      for (std::size_t lvl = 0; lvl < 7; ++lvl) {
        for (std::size_t s = 0; s < src[lvl].size() && s < dst[lvl].size(); ++s) {
          for (std::size_t c = 0; c < src[lvl][s].size() && c < dst[lvl][s].size(); ++c) {
            ctx->CopyResource(dst[lvl][s][c].tex.get(), src[lvl][s][c].tex.get());
          }
        }
      }
    }

    void copy_b1_imgs(ID3D11DeviceContext *ctx, const std::vector<tex_t> &src, std::vector<tex_t> &dst) {
      for (std::size_t i = 0; i < src.size() && i < dst.size(); ++i) {
        ctx->CopyResource(dst[i].tex.get(), src[i].tex.get());
      }
    }

  }  // namespace

  bool lsfg_framegen_t::impl_t::build_pipeline() {
    builder_t b {device.get()};

    // Resource ids 255 (mipmap) and 256 (generate) are shared between the "quality" and
    // "performance" shader sets. Ids 257-279 are quality; performance's counterparts are
    // the same shader roles at id+23 (ids 280-302), lighter/faster (fewer temporal-history
    // texture bindings each) -- ground-truth verified against a real Lossless.dll via
    // lsfg-fs's dumprsrc/reflect tools: 280-302 is a same-sized (23 entries) DXBC block
    // immediately following 279, with matching numthreads/bind-signature structure and
    // uniformly smaller sizes per id. Loaded here under the LOGICAL (quality) id either
    // way, so every b.disp(257..279, ...) call below stays mode-agnostic.
    for (std::uint16_t id = 255; id <= 279; ++id) {
      const std::uint16_t resource_id = (options.performance_mode && id >= 257) ? static_cast<std::uint16_t>(id + 23) : id;
      std::vector<std::uint8_t> bytes;
      if (!load_resource(lossless_module, resource_id, bytes)) {
        BOOST_LOG(error) << "LSFG: Lossless.dll is missing shader resource " << resource_id << " (unsupported Lossless Scaling version?)";
        return false;
      }
      com_ptr<ID3D11ComputeShader> cs;
      auto status = device->CreateComputeShader(bytes.data(), bytes.size(), nullptr, cs.put());
      if (FAILED(status)) {
        BOOST_LOG(error) << "LSFG: failed to create compute shader " << resource_id << " [0x" << util::hex(status).to_string_view() << ']';
        return false;
      }
      b.shaders.emplace(id, std::move(cs));
    }

    b.bnb = make_sampler(device.get(), D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_COMPARISON_NEVER, false);
    b.bnw = make_sampler(device.get(), D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_COMPARISON_NEVER, true);
    b.eab = make_sampler(device.get(), D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_COMPARISON_ALWAYS, false);
    if (!b.bnb || !b.bnw || !b.eab) {
      return false;
    }
    ID3D11SamplerState *bnb = b.bnb.get();
    ID3D11SamplerState *bnw = b.bnw.get();
    ID3D11SamplerState *eab = b.eab.get();

    // adaptive mode drives a single reusable generation tail
    const std::size_t count = 1;
    const ext_t source_extent = extent;

    // Performance mode doesn't just swap shader bytecode (the id+23 remap above):
    // the perf shaders declare HALF the per-group texture bindings, so the pipeline
    // structure must shrink to match -- lsfg-vk models this as `m = perf ? 1 : 2`
    // multiplying alpha0's temp/output groups, alpha1's per-slot outputs, gamma1's
    // and delta1's temp rings, and delta0's second output group (plus delta1's
    // second half binding only the first m entries of its rings). Feeding perf
    // bytecode quality-sized groups misaligns every SRV/UAV slot after the first
    // few and silently produces garbage flow (looks like interpolation not applying
    // at all). Group counts NOT multiplied by m (beta0/beta1 temps, gamma0/delta0
    // first outputs, the R8 pyramids) are fixed in lsfg-vk for both modes.
    const std::size_t m = options.performance_mode ? 1 : 2;
    const ext_t flow_extent {
      static_cast<std::uint32_t>(static_cast<float>(source_extent.w) / inv_flow),
      static_cast<std::uint32_t>(static_cast<float>(source_extent.h) / inv_flow)
    };

    auto cb0 = make_cb(device.get(), phase_of(0, 1), inv_flow, hdr, false);
    if (!cb0) {
      return false;
    }
    cbp.clear();
    for (std::size_t p = 0; p < count; ++p) {
      auto cb = make_cb(device.get(), phase_of(p, count), inv_flow, hdr, true);
      if (!cb) {
        return false;
      }
      cbp.push_back(std::move(cb));
    }

    const std::array<std::uint8_t, 4 * 4 * 4> black_pixels {};
    tex_t black;
    if (!make_tex2d(device.get(), {4, 4}, DXGI_FORMAT_R8G8B8A8_UNORM, false, black_pixels.data(), 4 * 4, black)) {
      return false;
    }

    const auto rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
    const auto r8 = DXGI_FORMAT_R8_UNORM;
    const auto rgba16f = DXGI_FORMAT_R16G16B16A16_FLOAT;

    prepass.clear();
    passes.clear();
    outputs.clear();
    for (auto &lvl : a1_out) {
      lvl.clear();
    }
    b1_imgs.clear();
    for (auto &lvl : active_a1_out) {
      lvl.clear();
    }
    active_b1_imgs.clear();
    held_a1_out.clear();
    held_b1_imgs.clear();

    // --- mipmaps: source -> 7 R8 mips of the flow grid ---
    std::vector<tex_t> mips;
    for (std::uint32_t i = 0; i < 7; ++i) {
      mips.push_back(b.tex(shift_ext(flow_extent, i), r8));
    }
    {
      step_t step;
      for (std::size_t s = 0; s < pool_size; ++s) {
        step.variants.push_back({b.disp(255, {&sources[s]}, refs(mips), {bnb}, cb0.get(), add_shift(flow_extent, 63, 6))});
      }
      prepass.push_back(std::move(step));
    }

    // --- alpha0 + alpha1 per mip level, rendered level 6 down to 0 ---
    // alpha1 keeps temporal history: 3 slots for level 0, 2 otherwise. a1_out/a1_extent
    // are impl_t members (not locals): queued mode needs them outside build_pipeline()
    // to size/copy the held-pair flow-state snapshots (see impl_t docs above).
    for (std::size_t lvl = 0; lvl < 7; ++lvl) {
      const auto e = mips[lvl].extent;
      const auto half = add_shift(e, 1, 1);
      const auto quarter = add_shift(half, 1, 1);
      a1_extent[lvl] = quarter;
      const std::size_t slots = (lvl == 0) ? 3 : 2;
      for (std::size_t s = 0; s < slots; ++s) {
        a1_out[lvl].push_back(b.texes(2 * m, quarter, rgba8));
      }
    }
    for (int lvl = 6; lvl >= 0; --lvl) {
      const auto e = mips[lvl].extent;
      const auto half = add_shift(e, 1, 1);
      const auto quarter = a1_extent[lvl];
      auto t0 = b.texes(m, half, rgba8);
      auto t1 = b.texes(m, half, rgba8);
      auto out = b.texes(2 * m, quarter, rgba8);
      {
        step_t step;
        step.variants.push_back({
          b.disp(267, {&mips[lvl]}, refs(t0), {bnb}, nullptr, g8(half)),
          b.disp(268, refs(t0), refs(t1), {bnb}, nullptr, g8(half)),
          b.disp(269, refs(t1), refs(out), {bnb}, nullptr, g8(quarter)),
        });
        prepass.push_back(std::move(step));
      }
      {
        step_t step;
        for (std::size_t slot = 0; slot < a1_out[lvl].size(); ++slot) {
          step.variants.push_back({b.disp(270, refs(out), refs(a1_out[lvl][slot]), {bnb}, nullptr, g8(quarter))});
        }
        prepass.push_back(std::move(step));
      }
    }

    // --- beta0: rotates over alpha1 level 0's three temporal slots ---
    const auto be = a1_extent[0];
    auto b0_out = b.texes(2, be, rgba8);
    {
      step_t step;
      for (std::size_t i = 0; i < 3; ++i) {
        std::vector<const tex_t *> srcs;
        srcs.reserve(12);
        for (const auto *t : refs(a1_out[0][(i + 1) % 3])) {
          srcs.push_back(t);
        }
        for (const auto *t : refs(a1_out[0][(i + 2) % 3])) {
          srcs.push_back(t);
        }
        for (const auto *t : refs(a1_out[0][i])) {
          srcs.push_back(t);
        }
        step.variants.push_back({b.disp(275, srcs, refs(b0_out), {bnw}, nullptr, g8(be))});
      }
      prepass.push_back(std::move(step));
    }

    // --- beta1: blur chain -> 6 R8 pyramid levels --- (b1_imgs is an impl_t member; see a1_out above)
    auto b1_t0 = b.texes(2, be, rgba8);
    auto b1_t1 = b.texes(2, be, rgba8);
    for (std::uint32_t i = 0; i < 6; ++i) {
      b1_imgs.push_back(b.tex(shift_ext(be, i), r8));
    }
    {
      step_t step;
      step.variants.push_back({
        b.disp(276, refs(b0_out), refs(b1_t0), {bnb}, nullptr, g8(be)),
        b.disp(277, refs(b1_t0), refs(b1_t1), {bnb}, nullptr, g8(be)),
        b.disp(278, refs(b1_t1), refs(b1_t0), {bnb}, nullptr, g8(be)),
        b.disp(279, refs(b1_t0), refs(b1_imgs), {bnb}, cb0.get(), add_shift(be, 31, 5)),
      });
      prepass.push_back(std::move(step));
    }

    // --- queued mode only: a stable "active pair" working copy of a1_out/b1_imgs
    // (refreshed via CopyResource, not compute-written, when the active pair changes),
    // plus a pool_size-deep ring of full snapshots taken after every arrival's
    // pre-pass run so an arbitrary already-arrived pair's flow data survives newer
    // arrivals continuing to advance the rolling live history. See impl_t docs above.
    if (options.queue_frames >= 1) {
      active_a1_out = make_a1_out_like(b, a1_extent, 2 * m);
      active_b1_imgs = make_b1_imgs_like(b, be);
      held_a1_out.reserve(pool_size);
      held_b1_imgs.reserve(pool_size);
      for (std::size_t slot = 0; slot < pool_size; ++slot) {
        held_a1_out.push_back(make_a1_out_like(b, a1_extent, 2 * m));
        held_b1_imgs.push_back(make_b1_imgs_like(b, be));
      }
    }

    // --- per generated frame: gamma / delta pyramids + generate ---
    for (std::size_t p = 0; p < count; ++p) {
      ID3D11Buffer *cb = cbp[p].get();
      // Extrapolate mode reads the live, continuously-rolling a1_out/b1_imgs directly.
      // Queued mode reads the separate "active pair" working copy instead, refreshed
      // by start_active_pair() -- see the impl_t member docs for why.
      auto &gen_a1_out = (options.queue_frames >= 1) ? active_a1_out : a1_out;
      auto &gen_b1_imgs = (options.queue_frames >= 1) ? active_b1_imgs : b1_imgs;
      std::vector<step_t> steps;
      std::vector<tex_t> g1_img;
      std::vector<tex_t> d1_img0;
      std::vector<tex_t> d1_img1;
      g1_img.reserve(7);
      d1_img0.reserve(3);
      d1_img1.reserve(3);
      for (std::size_t j = 0; j < 7; ++j) {
        const std::size_t lvl = 6 - j;
        const auto e = a1_extent[lvl];
        const auto &srcs = gen_a1_out[lvl];  // 2 temporal slots at levels 1-6

        // gamma0
        auto g0_out = b.texes(3, e, rgba8);
        const tex_t *g_add0 = (j == 0) ? &black : &g1_img[j - 1];
        {
          step_t step;
          for (std::size_t i = 0; i < 2; ++i) {
            std::vector<const tex_t *> sv;
            sv.reserve(9);
            for (const auto *t : refs(srcs[(i + 1) % 2])) {
              sv.push_back(t);
            }
            for (const auto *t : refs(srcs[i])) {
              sv.push_back(t);
            }
            sv.push_back(g_add0);
            step.variants.push_back({b.disp(257, sv, refs(g0_out), {bnw, eab}, cb, g8(e))});
          }
          steps.push_back(std::move(step));
        }

        // gamma1
        auto g1_t0 = b.texes(2 * m, e, rgba8);
        auto g1_t1 = b.texes(2 * m, e, rgba8);
        auto img = b.tex(e, rgba16f);
        const std::size_t bidx = (j == 0) ? 5 : 6 - j;
        {
          std::vector<const tex_t *> last_srcs = refs(g1_t0);
          last_srcs.push_back(g_add0);
          last_srcs.push_back(&gen_b1_imgs[bidx]);
          step_t step;
          step.variants.push_back({
            b.disp(259, refs(g0_out), refs(g1_t0), {bnb}, nullptr, g8(e)),
            b.disp(260, refs(g1_t0), refs(g1_t1), {bnb}, nullptr, g8(e)),
            b.disp(261, refs(g1_t1), refs(g1_t0), {bnb}, nullptr, g8(e)),
            b.disp(262, last_srcs, {&img}, {bnb, eab}, cb, g8(e)),
          });
          steps.push_back(std::move(step));
        }
        g1_img.push_back(std::move(img));

        // delta passes for j = 4, 5, 6
        if (j < 4) {
          continue;
        }
        const std::size_t k = j - 4;
        auto d0_out0 = b.texes(3, e, rgba8);
        auto d0_out1 = b.texes(m, e, rgba8);
        const tex_t *d_add0 = (k == 0) ? &black : &d1_img0[k - 1];
        const tex_t *d_add1 = &g1_img[j - 1];
        {
          step_t step;
          for (std::size_t i = 0; i < 2; ++i) {
            std::vector<const tex_t *> sv0;
            sv0.reserve(9);
            for (const auto *t : refs(srcs[(i + 1) % 2])) {
              sv0.push_back(t);
            }
            for (const auto *t : refs(srcs[i])) {
              sv0.push_back(t);
            }
            sv0.push_back(d_add0);
            std::vector<const tex_t *> sv1;
            sv1.reserve(10);
            for (const auto *t : refs(srcs[(i + 1) % 2])) {
              sv1.push_back(t);
            }
            for (const auto *t : refs(srcs[i])) {
              sv1.push_back(t);
            }
            sv1.push_back(d_add1);
            sv1.push_back(d_add0);
            step.variants.push_back({
              b.disp(257, sv0, refs(d0_out0), {bnw, eab}, cb, g8(e)),
              b.disp(258, sv1, refs(d0_out1), {bnw, eab}, cb, g8(e)),
            });
          }
          steps.push_back(std::move(step));
        }

        auto d1_t0 = b.texes(2 * m, e, rgba8);
        auto d1_t1 = b.texes(2 * m, e, rgba8);
        auto img0 = b.tex(e, rgba16f);
        auto img1 = b.tex(e, rgba16f);
        const tex_t *d_add2 = (k == 0) ? &black : &d1_img1[k - 1];
        {
          std::vector<const tex_t *> s4 = refs(d1_t0);
          s4.push_back(d_add0);
          s4.push_back(&gen_b1_imgs[6 - j]);
          // delta1's second half binds only the first m ring entries (lsfg-vk's
          // `.storages(this->tempImages0, 0, m)` subrange).
          std::vector<const tex_t *> t0h;
          std::vector<const tex_t *> t1h;
          for (std::size_t h = 0; h < m; ++h) {
            t0h.push_back(&d1_t0[h]);
            t1h.push_back(&d1_t1[h]);
          }
          std::vector<const tex_t *> s7 = t0h;
          s7.push_back(d_add2);
          step_t step;
          step.variants.push_back({
            b.disp(263, refs(d0_out0), refs(d1_t0), {bnb}, nullptr, g8(e)),
            b.disp(264, refs(d1_t0), refs(d1_t1), {bnb}, nullptr, g8(e)),
            b.disp(265, refs(d1_t1), refs(d1_t0), {bnb}, nullptr, g8(e)),
            b.disp(266, s4, {&img0}, {bnb, eab}, cb, g8(e)),
            b.disp(271, refs(d0_out1), t0h, {bnb}, nullptr, g8(e)),
            b.disp(272, t0h, t1h, {bnb}, nullptr, g8(e)),
            b.disp(273, t1h, t0h, {bnb}, nullptr, g8(e)),
            b.disp(274, s7, {&img1}, {bnb, eab}, cb, g8(e)),
          });
          steps.push_back(std::move(step));
        }
        d1_img0.push_back(std::move(img0));
        d1_img1.push_back(std::move(img1));
      }

      // generate: warp prev+curr with the flow pyramids into the output.
      // RGBA16F for HDR streams; RGBA8 (mandatory UAV store support) otherwise.
      auto out = b.tex(source_extent, hdr ? rgba16f : rgba8);
      {
        step_t step;
        for (std::size_t v = 0; v < pool_size; ++v) {
          // variant v: curr = sources[v] (the pair's newer/active frame), prev = the
          // slot immediately behind it in the ring. Identical to the old fixed
          // i==0/i==1 two-variant scheme when pool_size == 2.
          const tex_t *s1 = &sources[v];
          const tex_t *s0 = &sources[(v + pool_size - 1) % pool_size];
          step.variants.push_back({b.disp(
            256,
            {s0, s1, &g1_img[6], &d1_img0[2], &d1_img1[2]},
            {&out},
            {bnb, eab},
            cb,
            add_shift(source_extent, 15, 4)
          )});
        }
        steps.push_back(std::move(step));
      }
      outputs.push_back(std::move(out));
      passes.push_back(std::move(steps));
    }

    return b.ok;
  }

  void lsfg_framegen_t::impl_t::run_steps(const std::vector<step_t> &steps, std::size_t fidx) {
    for (const auto &step : steps) {
      for (const auto &d : step.variants[fidx % step.variants.size()]) {
        exec_disp(ctx.get(), d);
      }
    }
    ctx->CSSetShader(nullptr, nullptr, 0);
  }

  lsfg_framegen_t::lsfg_framegen_t():
      _impl(std::make_unique<impl_t>()) {
  }

  lsfg_framegen_t::~lsfg_framegen_t() = default;

  std::optional<std::filesystem::path> lsfg_framegen_t::find_lossless_dll() {
    // explicit override for development / unusual installs
    if (const auto *env = std::getenv("LSFG_DLL_PATH")) {
      std::filesystem::path p {env};
      std::error_code ec;
      if (std::filesystem::is_regular_file(p, ec)) {
        return p;
      }
    }

    std::optional<std::filesystem::path> configured;
    if (!config::lossless_scaling.exe_path.empty()) {
      configured = std::filesystem::path(config::lossless_scaling.exe_path);
    }

    const auto candidates = lossless_paths::discover_lossless_candidates(
      configured,
      std::nullopt,
      lossless_paths::default_steam_lossless_path()
    );
    for (const auto &exe : candidates) {
      auto dll = exe.parent_path() / L"Lossless.dll";
      std::error_code ec;
      if (std::filesystem::is_regular_file(dll, ec)) {
        return dll;
      }
    }
    return std::nullopt;
  }

  std::unique_ptr<lsfg_framegen_t> lsfg_framegen_t::create(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    std::uint32_t width,
    std::uint32_t height,
    DXGI_FORMAT capture_format,
    const options_t &options
  ) {
    if (!device || !ctx || width == 0 || height == 0) {
      return nullptr;
    }

    bool hdr = false;
    switch (capture_format) {
      case DXGI_FORMAT_B8G8R8A8_UNORM:
      case DXGI_FORMAT_R8G8B8A8_UNORM:
        break;
      case DXGI_FORMAT_R16G16B16A16_FLOAT:
        hdr = true;
        break;
      default:
        BOOST_LOG(warning) << "LSFG: unsupported capture format for frame generation; passing frames through unmodified";
        return nullptr;
    }

    const auto dll_path = find_lossless_dll();
    if (!dll_path) {
      BOOST_LOG(warning) << "LSFG: Lossless.dll not found (install Lossless Scaling from Steam or set the executable path); frame generation disabled";
      return nullptr;
    }

    std::unique_ptr<lsfg_framegen_t> self {new lsfg_framegen_t()};
    auto &impl = *self->_impl;
    impl.device.copy_from(device);
    impl.ctx.copy_from(ctx);
    impl.extent = {width, height};
    impl.format = capture_format;
    impl.hdr = hdr;
    impl.options = options;
    impl.options.flow_scale = std::clamp(options.flow_scale, 0.25f, 1.0f);
    impl.options.max_multiplier = std::clamp(options.max_multiplier, 2, 20);
    impl.options.queue_frames = std::clamp(options.queue_frames, 0, 2);
    impl.options.target_fps_cutoff_percent = std::clamp(options.target_fps_cutoff_percent, 50, 100);
    impl.inv_flow = 1.0f / impl.options.flow_scale;
    impl.recompute_frame_dur();
    impl.pool_size = static_cast<std::size_t>(impl.options.queue_frames) + 2;

    impl.lossless_module = LoadLibraryExW(dll_path->c_str(), nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (!impl.lossless_module) {
      BOOST_LOG(warning) << "LSFG: failed to load Lossless.dll as a resource module; frame generation disabled";
      return nullptr;
    }

    impl.sources.resize(impl.pool_size);
    impl.slot_qpc.assign(impl.pool_size, 0);
    for (auto &src : impl.sources) {
      if (!make_tex2d(device, impl.extent, capture_format, false, nullptr, 0, src)) {
        return nullptr;
      }
    }
    if (!make_tex2d(device, impl.extent, capture_format, false, nullptr, 0, impl.latest)) {
      return nullptr;
    }

    if (!impl.blitter.init(device)) {
      return nullptr;
    }

    if (!impl.build_pipeline()) {
      BOOST_LOG(warning) << "LSFG: failed to build the frame-generation pipeline; frame generation disabled";
      return nullptr;
    }

    BOOST_LOG(info) << "LSFG: adaptive frame generation active [" << width << 'x' << height
                    << ", target " << options.target_fps << " fps"
                    << (impl.options.target_fps_cutoff_percent < 100 ? " (internally pacing to " + std::to_string(options.target_fps * impl.options.target_fps_cutoff_percent / 100.0) + " fps, " + std::to_string(impl.options.target_fps_cutoff_percent) + "% cutoff)" : std::string())
                    << ", flow scale " << impl.options.flow_scale
                    << ", cap " << impl.options.max_multiplier << "x, queue "
                    << impl.options.queue_frames << " frame(s), "
                    << (impl.options.performance_mode ? "performance" : "quality") << " shaders, "
                    << (hdr ? "HDR" : "SDR") << ", shaders from " << dll_path->string() << ']';
    return self;
  }

  void lsfg_framegen_t::stage_capture(ID3D11Texture2D *src) {
    auto &impl = *_impl;
    impl.ctx->CopyResource(impl.latest.tex.get(), src);
    impl.has_capture = true;
  }

  void lsfg_framegen_t::commit_capture(std::uint64_t frame_qpc, bool frame_dirty) {
    auto &impl = *_impl;
    if (!impl.has_capture) {
      return;
    }

    // TEMPORARY DIAGNOSTIC OVERRIDE: force every frame through as dirty, disabling
    // the DirtyRegionMode skip below without removing it, to isolate whether a
    // stale/partial deploy (not all LSFG binaries updated) was the actual cause of
    // a prior "LSFG not interpolating" report, independent of DirtyRegionMode
    // reliability. Flip back to false once confirmed either way.
    constexpr bool kForceDirtyBypass = true;
    if (kForceDirtyBypass) {
      frame_dirty = true;
    }

    // Skip source rotation and the interval estimate for a frame WGC reported as a
    // compositor-only republish (DirtyRegionMode, Windows 11 24H2+): otherwise a static
    // screen or idle recomposite would perturb src_interval and mistime the adaptive
    // phase. On older Windows builds frame_dirty is always true (unchanged behavior).
    //
    // Bounded to one skip in a row: DirtyRegionMode is compositor recomposition
    // tracking, which is unverified on the virtual/headless displays this project
    // primarily targets. If it ever misreports every frame as non-dirty, an
    // unbounded skip would permanently starve frames_seen below 2 and silently
    // disable generation entirely (pass-through only, higher output fps but no
    // interpolation) -- far worse than occasionally trusting a stale interval.
    constexpr std::size_t kMaxConsecutiveDirtySkips = 1;
    ++impl.commit_calls;
    if (!frame_dirty) {
      ++impl.dirty_skip_total;
    }
    if (impl.commit_calls % 300 == 0) {
      BOOST_LOG(info) << "LSFG: commit_capture diagnostics: calls=" << impl.commit_calls
                      << " dirty_skips=" << impl.dirty_skip_total
                      << " (" << (100.0 * static_cast<double>(impl.dirty_skip_total) / static_cast<double>(impl.commit_calls)) << "%)"
                      << " frames_seen=" << impl.frames_seen
                      << " src_interval_ms=" << std::chrono::duration<double, std::milli>(impl.src_interval).count();
    }
    if (!frame_dirty && impl.consecutive_dirty_skips < kMaxConsecutiveDirtySkips) {
      ++impl.consecutive_dirty_skips;
      return;
    }
    impl.consecutive_dirty_skips = 0;
    impl.passthrough_dirty = true;

    const std::size_t new_idx = impl.frames_seen;  // 0-based index of this arrival
    impl.ctx->CopyResource(impl.sources[new_idx % impl.pool_size].tex.get(), impl.latest.tex.get());
    if (impl.last_distinct_qpc != 0 && frame_qpc > impl.last_distinct_qpc) {
      const auto dt = platf::qpc_time_difference(static_cast<int64_t>(frame_qpc), static_cast<int64_t>(impl.last_distinct_qpc));
      if (dt > 0ns && dt < 1s) {
        impl.src_interval = std::chrono::duration_cast<std::chrono::nanoseconds>(impl.src_interval * 0.85 + dt * 0.15);
      }
    }
    if (frame_qpc != 0) {
      impl.last_distinct_qpc = frame_qpc;
    }

    if (impl.options.queue_frames <= 0) {
      // Extrapolate mode (today's behavior, unchanged): run the flow pre-pass
      // immediately for the newest arrival and reset the wall-clock phase anchor.
      //
      // Anchor to the frame's own QPC timestamp (the compositor's capture clock),
      // not steady_clock::now() at commit time: WGC delivery jitter, the up-to-~6ms
      // effective_wgc_timeout() blocking wait, and IPC handoff all add variable delay
      // between when the frame was actually captured and when commit_capture() runs
      // for it. Re-deriving the phase-slot grid from that jittery instant every
      // interval can misclassify a metronome tick into the wrong slot; QPC is the
      // same clock WGC stamped the frame with, so this locks the grid to the
      // source's true cadence instead. Falls back to steady_clock::now() if the
      // backend couldn't supply a QPC timestamp.
      impl.last_arrival = frame_qpc != 0
        ? std::chrono::steady_clock::now() - platf::qpc_time_difference(platf::qpc_counter(), static_cast<int64_t>(frame_qpc))
        : std::chrono::steady_clock::now();
      if (impl.frames_seen > 0) {
        // compute optical flow once per distinct frame; generation tails reuse it
        // for every phase requested until the next distinct frame arrives
        impl.run_steps(impl.prepass, impl.frames_seen);
      }
      ++impl.frames_seen;
      return;
    }

    // Queued mode: the flow pre-pass still runs on every single arrival, exactly
    // like extrapolate mode -- the shaders' alpha1/beta0 temporal history is a
    // continuously rolling window with no concept of "skip this one," so it must
    // never see a gap regardless of whether this arrival ends up chosen as an
    // active pair's endpoint. Immediately after, snapshot the live a1_out/b1_imgs
    // into this slot's held copy so a delayed pair's flow data survives later
    // arrivals continuing to advance that same rolling history underneath it.
    impl.slot_qpc[new_idx % impl.pool_size] = frame_qpc;
    if (impl.frames_seen > 0) {
      impl.run_steps(impl.prepass, new_idx);
    }
    impl.snapshot_flow_state(new_idx % impl.pool_size);
    ++impl.frames_seen;
    impl.try_advance_active_pair();
  }

  void lsfg_framegen_t::impl_t::try_advance_active_pair() {
    const std::size_t newest_idx = frames_seen - 1;  // 0-based index of the newest arrival
    const auto lookahead = static_cast<std::size_t>(options.queue_frames);

    if (!buffered_ready) {
      // Bootstrap: need a first pair (0, 1) plus `lookahead` frames of confirmation
      // beyond it before we can trust its measured interval and start presenting it.
      if (frames_seen < 2 + lookahead) {
        return;
      }
      active_idx = 1;
      start_active_pair();
      buffered_ready = true;
      return;
    }

    // Already presenting a pair: advance the instant the next one is confirmed
    // (arrival-count based), NOT gated on real elapsed time vs. active_interval.
    //
    // active_interval is deliberately a single unsmoothed measurement (it needs to
    // be the *exact* interval for that specific pair, that's the whole point of
    // queued mode) -- which makes it as noisy as any single game frame-time sample.
    // Gating advancement on "has that long elapsed yet" let one overestimated
    // interval stall active_idx while new real arrivals kept landing in the source
    // pool regardless (commit_capture() always writes sources[new_idx % pool_size]
    // unconditionally) -- pool_size is sized assuming active_idx never lags
    // newest_idx - lookahead by more than one arrival, so stalling it let a new
    // arrival's slot wrap around and overwrite a texture the still-active pair was
    // being read from mid-use. That was visible as periodic artifacting, with a
    // stutter when the gate finally released and jumped to catch up.
    //
    // want_generated()'s own elapsed/active_interval phase clamp is what paces the
    // *visual* reveal of generated frames within a pair; it doesn't need this
    // function to also gate on time, only to keep the pool from overflowing.
    if (newest_idx < active_idx + 1 + lookahead) {
      return;  // next pair isn't confirmed yet -- want_generated() holds
    }

    active_idx = newest_idx - lookahead;
    start_active_pair();
  }

  void lsfg_framegen_t::impl_t::start_active_pair() {
    const auto curr_qpc = slot_qpc[active_idx % pool_size];
    const auto prev_qpc = slot_qpc[(active_idx - 1) % pool_size];
    // Same QPC-anchoring rationale as extrapolate mode's last_arrival (see
    // commit_capture()): locks the phase-slot grid to the pair's actual capture
    // cadence rather than the jittery instant try_advance_active_pair() happened to
    // run at.
    active_anchor = curr_qpc != 0
      ? std::chrono::steady_clock::now() - platf::qpc_time_difference(platf::qpc_counter(), static_cast<int64_t>(curr_qpc))
      : std::chrono::steady_clock::now();
    if (curr_qpc != 0 && prev_qpc != 0 && curr_qpc > prev_qpc) {
      const auto dt = platf::qpc_time_difference(static_cast<int64_t>(curr_qpc), static_cast<int64_t>(prev_qpc));
      if (dt > 0ns) {
        // Same runaway-gap guard as extrapolate mode's max_multiplier cap: a pair
        // separated by a long stall is held, not slow-motion-crossfaded into.
        active_interval = std::min(dt, frame_dur * options.max_multiplier);
      }
    }
    // The flow pre-pass for exactly this (prev, curr) pair already ran continuously
    // back when this arrival came in (commit_capture()); pull its saved snapshot
    // into the stable "active" working copy gamma/delta/generate read from, rather
    // than re-running the pre-pass now (which would compute flow for whatever the
    // live rolling history currently holds -- almost certainly a later pair by now).
    restore_active_flow_state(active_idx % pool_size);
  }

  void lsfg_framegen_t::impl_t::snapshot_flow_state(std::size_t slot) {
    copy_a1_out(ctx.get(), a1_out, held_a1_out[slot]);
    copy_b1_imgs(ctx.get(), b1_imgs, held_b1_imgs[slot]);
  }

  void lsfg_framegen_t::impl_t::restore_active_flow_state(std::size_t slot) {
    copy_a1_out(ctx.get(), held_a1_out[slot], active_a1_out);
    copy_b1_imgs(ctx.get(), held_b1_imgs[slot], active_b1_imgs);
  }

  bool lsfg_framegen_t::want_generated(std::chrono::steady_clock::time_point now, float &phase_out) {
    auto &impl = *_impl;

    if (impl.options.queue_frames >= 1) {
      // Queued mode: the active pair's interval is exact (measured, not estimated),
      // so no min_gen_ratio/quantization noise-reduction is needed here -- that was
      // only ever compensating for src_interval being a guess. Still wall-clock
      // elapsed/interval, for the same call-cadence-robustness reason as below.
      if (!impl.buffered_ready) {
        return false;
      }
      const auto elapsed = std::chrono::duration<float>(now - impl.active_anchor).count();
      const auto interval_s = std::max(std::chrono::duration<float>(impl.active_interval).count(), 1e-4f);
      const auto frame_dur_s = std::max(std::chrono::duration<float>(impl.frame_dur).count(), 1e-4f);
      const long steps = std::max<long>(1, std::lround(interval_s / frame_dur_s));

      // Bucket by tick number within the interval, not by rounding a continuous
      // fraction -- see the extrapolate-mode branch below for why flooring avoids
      // duplicate/skipped slots that rounding produces. active_interval is exact here
      // (queued mode's whole point), so this is purely about matching lsfg-vk's
      // fixed-phase cadence and avoiding rounding-boundary flips, not about denominator
      // noise (extrapolate mode's other reason for quantizing).
      long slot = static_cast<long>(std::floor(elapsed / frame_dur_s)) + 1;
      slot = std::clamp<long>(slot, 1, steps);
      impl.record_phase_tick(now, impl.active_anchor, elapsed, slot, steps);
      if (slot >= steps) {
        return false;  // holding at the active pair's newer frame until the next pair is ready
      }

      phase_out = static_cast<float>(slot) / static_cast<float>(steps);
      return true;
    }

    if (impl.frames_seen < 2) {
      return false;
    }

    // Only generate if a source interval spans enough present slots to fit an
    // in-between frame; below this the source is already near the target fps.
    const double min_gen_ratio = 1.5;
    if (std::chrono::duration<double>(impl.src_interval).count() < std::chrono::duration<double>(impl.frame_dur).count() * min_gen_ratio) {
      return false;
    }

    // Cap how far the phase stretches: never interpolate as though the gap
    // since the last real frame were more than `max_multiplier` present
    // intervals. Without this a stalled source would extrapolate across the
    // entire gap. Past the capped interval the latest real frame is held.
    //
    // NOTE: this must stay wall-clock based, not pacing-tick-counted. The capture
    // loop's frame-pacing "metronome" (display_base.cpp) can call snapshot() --
    // and therefore this function -- twice in a single loop iteration when a
    // pacing group busts (0ms continuation attempt fails, falls through to a
    // fresh 200ms snapshot immediately, no sleep in between). A tick counter
    // desyncs from real elapsed time every time that happens, which is routine
    // per that code's own bust-mix diagnostics -- tried this 2026-07-09, it
    // produced visibly worse judder ("rewinding") than the wall-clock version.
    const auto capped_interval = std::min(impl.src_interval, impl.frame_dur * impl.options.max_multiplier);

    // How many present slots fit in the (capped) source interval. src_interval is a
    // noisy EMA of real inter-frame gaps (game frame-time variance, WGC delivery
    // jitter); rounding it to a whole tick count here, instead of dividing by it
    // directly below, is what keeps the bucket count itself from jittering tick to
    // tick whenever the source:target ratio sits close to a small integer (e.g. 60fps
    // source into a 116fps stream, ratio ~1.93).
    std::int64_t steps = 1;
    if (const auto frame_dur_ns = impl.frame_dur.count(); frame_dur_ns > 0) {
      steps = std::max<std::int64_t>(1, (capped_interval.count() + frame_dur_ns / 2) / frame_dur_ns);
    }
    const long steps_i = static_cast<long>(steps);

    const auto elapsed = std::chrono::duration<float>(now - impl.last_arrival).count();
    const auto frame_dur_s = std::max(std::chrono::duration<float>(impl.frame_dur).count(), 1e-4f);

    // Bucket by tick number within the interval (floor(elapsed / frame_dur) + 1) --
    // NOT by rounding a continuous fraction (t * steps) to the nearest slot. Ticks
    // land ~frame_dur apart by construction (frame_dur IS the pacing metronome's
    // grid), so consecutive ticks fall in consecutive integer buckets under flooring.
    // Round-to-nearest instead has its boundaries at HALF a slot width; a small shift
    // in the tick-vs-arrival offset (ordinary drift, since target fps is essentially
    // never an integer multiple of source fps) can flip two consecutive ticks into
    // the SAME rounded bucket, which duplicates one warp phase and skips the other
    // slot entirely for that interval -- sometimes the final real-frame slot, which
    // is exactly the kind of miss that makes generation look worse, not better, as
    // source fps rises toward target (fewer, wider buckets -- lsfg-vk instead avoids
    // this class of problem altogether by presenting a FIXED frame count per real
    // arrival, decoupled from any independent tick clock).
    long slot = static_cast<long>(std::floor(elapsed / frame_dur_s)) + 1;
    slot = std::clamp<long>(slot, 1, steps_i);
    impl.record_phase_tick(now, impl.last_arrival, elapsed, slot, steps_i);
    if (slot >= steps_i) {
      return false;
    }

    phase_out = static_cast<float>(slot) / static_cast<float>(steps_i);
    return true;
  }

  bool lsfg_framegen_t::render_generated(float phase, ID3D11RenderTargetView *rtv, std::uint32_t out_width, std::uint32_t out_height) {
    auto &impl = *_impl;
    const bool queued = impl.options.queue_frames >= 1;
    if ((queued ? !impl.buffered_ready : impl.frames_seen < 2) || impl.passes.empty() || !rtv) {
      return false;
    }

    // adaptive mode: rewrite the interpolation phase, then run the tail
    const auto data = make_cb_data(phase, impl.inv_flow, impl.hdr);
    impl.ctx->UpdateSubresource(impl.cbp[0].get(), 0, nullptr, &data, 0, 0);

    // Queued mode: the pool-index variant for the pair the pre-pass was just run
    // against (start_active_pair()). Extrapolate mode: the newest arrival, as before.
    const std::size_t fidx = queued ? impl.active_idx : impl.frames_seen - 1;
    impl.run_steps(impl.passes[0], fidx);
    impl.blitter.blit(impl.ctx.get(), impl.outputs[0].srv.get(), rtv, out_width, out_height);
    return true;
  }

  void lsfg_framegen_t::impl_t::recompute_frame_dur() {
    if (options.target_fps > 0.0) {
      const double effective_target_fps = options.target_fps * (static_cast<double>(options.target_fps_cutoff_percent) / 100.0);
      frame_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / effective_target_fps));
    }
  }

  // TEMPORARY diagnostics (added 2026-07-09): see the diag_* member docs above.
  // Called from both want_generated() branches once per tick, right after the slot
  // for that tick is decided. `anchor` is impl.last_arrival (extrapolate) or
  // impl.active_anchor (queued) -- comparing it to the previous call's anchor is how
  // "a new source interval started" is detected, since both are only ever reassigned
  // on a genuine new arrival/pair activation.
  void lsfg_framegen_t::impl_t::record_phase_tick(std::chrono::steady_clock::time_point now, std::chrono::steady_clock::time_point anchor, float elapsed_s, long slot, long steps_i) {
    if (diag_have_last_anchor && anchor != diag_last_anchor) {
      ++diag_intervals_closed;
      if (diag_interval_ticks == 1) {
        ++diag_single_tick_intervals;
      }
      if (diag_interval_showed_real) {
        ++diag_intervals_real_shown;
      }
      diag_interval_ticks = 0;
      diag_interval_showed_real = false;
      diag_last_slot_in_interval = -1;
    }
    diag_last_anchor = anchor;
    diag_have_last_anchor = true;

    ++diag_ticks_total;
    if (diag_interval_ticks == 0) {
      diag_first_tick_elapsed_min = std::min(diag_first_tick_elapsed_min, elapsed_s);
      diag_first_tick_elapsed_max = std::max(diag_first_tick_elapsed_max, elapsed_s);
    }
    ++diag_interval_ticks;

    if (diag_last_slot_in_interval == slot) {
      ++diag_duplicate_slot_count;
    }
    diag_last_slot_in_interval = slot;

    const bool showed_real = slot >= steps_i;
    if (showed_real) {
      diag_interval_showed_real = true;
      ++diag_passthrough_tick_count;
    } else {
      ++diag_generated_count;
    }

    if (!diag_have_last_log) {
      diag_last_log = now;
      diag_have_last_log = true;
      return;
    }
    if (now - diag_last_log < 10s) {
      return;
    }
    diag_last_log = now;

    const double dup_pct = diag_ticks_total ? 100.0 * static_cast<double>(diag_duplicate_slot_count) / static_cast<double>(diag_ticks_total) : 0.0;
    const double single_tick_pct = diag_intervals_closed ? 100.0 * static_cast<double>(diag_single_tick_intervals) / static_cast<double>(diag_intervals_closed) : 0.0;
    const double real_shown_pct = diag_intervals_closed ? 100.0 * static_cast<double>(diag_intervals_real_shown) / static_cast<double>(diag_intervals_closed) : 0.0;
    BOOST_LOG(info) << "LSFG pacing diagnostics: ticks=" << diag_ticks_total
                    << " generated=" << diag_generated_count
                    << " passthrough_ticks=" << diag_passthrough_tick_count
                    << " intervals=" << diag_intervals_closed
                    << " dup_slot=" << diag_duplicate_slot_count << " (" << dup_pct << "%)"
                    << " single_tick_intervals=" << diag_single_tick_intervals << " (" << single_tick_pct << "%)"
                    << " real_shown_intervals=" << diag_intervals_real_shown << " (" << real_shown_pct << "%)"
                    << " first_tick_elapsed_ms=[" << (diag_first_tick_elapsed_min * 1000.0f) << ".." << (diag_first_tick_elapsed_max * 1000.0f) << "]";
  }

  void lsfg_framegen_t::update_live_options(int max_multiplier, int target_fps_cutoff_percent) {
    auto &impl = *_impl;
    impl.options.max_multiplier = std::clamp(max_multiplier, 2, 20);
    const auto clamped_cutoff = std::clamp(target_fps_cutoff_percent, 50, 100);
    if (clamped_cutoff != impl.options.target_fps_cutoff_percent) {
      impl.options.target_fps_cutoff_percent = clamped_cutoff;
      impl.recompute_frame_dur();
    }
  }

  ID3D11Texture2D *lsfg_framegen_t::latest_texture() const {
    auto &impl = *_impl;
    if (impl.options.queue_frames >= 1 && impl.buffered_ready) {
      // Show the active pair's confirmed newer frame while holding, not the
      // absolute newest raw capture -- jumping to it would skip ahead of the
      // deliberate queue delay and undo the point of buffering.
      return impl.sources[impl.active_idx % impl.pool_size].tex.get();
    }
    return impl.latest.tex.get();
  }

  bool lsfg_framegen_t::has_frame() const {
    return _impl->has_capture;
  }

  bool lsfg_framegen_t::has_new_passthrough_frame() const {
    return _impl->passthrough_dirty;
  }

  void lsfg_framegen_t::mark_passthrough_shown() {
    _impl->passthrough_dirty = false;
  }

}  // namespace platf::dxgi
