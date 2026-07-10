/**
 * @file src/platform/windows/lsfg_framegen.h
 * @brief Host-side Lossless Scaling frame generation (LSFG) for the Windows capture pipeline.
 *
 * D3D11 port of the LSFG frame-generation pipeline from lsfg-vk
 * (https://github.com/PancakeTAS/lsfg-vk, GPL-3.0-or-later).
 *
 * The compute shaders themselves are NOT part of Vibepollo: they are loaded at
 * runtime as DXBC blobs from the RT_RCDATA resources of the user's own
 * Lossless Scaling installation (Lossless.dll, purchased on Steam). If the
 * installation is missing the feature disables itself and capture proceeds
 * unmodified.
 */
#pragma once

#ifdef _WIN32

  #include <chrono>
  #include <cstdint>
  #include <filesystem>
  #include <memory>
  #include <optional>

  #include <d3d11.h>

namespace platf::dxgi {

  /**
   * @brief Adaptive-mode LSFG interpolator for a capture backend.
   *
   * Owns the ping-pong source frames, the optical-flow pre-pass and one
   * reusable generation tail driven by a dynamic interpolation phase
   * (LSFG "adaptive" mode). Captured frames are deduplicated with a
   * full-frame content hash so a compositor republish does not disturb
   * the source-interval estimate.
   *
   * All methods must be called from the capture thread that owns the
   * supplied immediate context.
   */
  class lsfg_framegen_t {
  public:
    struct options_t {
      float flow_scale = 1.0f;  ///< Optical-flow resolution scale, 0.25..1.0.
      int max_multiplier = 4;  ///< Adaptive phase cap (max output/input frame ratio honored).
      double target_fps = 60.0;  ///< Client-requested stream FPS (interpolation target).
      /// Real source frames to hold back before interpolating (0..2). 0 = extrapolate from the
      /// newest arrival using an estimated interval (lowest latency, today's default behavior).
      /// N>=1 delays presentation by N source-frame intervals, but interpolates the confirmed
      /// (already-arrived) pair using its *exact* measured interval instead of a guess -- trades
      /// latency for smoothness, mirroring Lossless Scaling's own queue-target setting.
      int queue_frames = 0;
      /// Use Lossless Scaling's "performance" optical-flow shader set instead of "quality".
      /// Lighter/faster (fewer temporal-history texture bindings per shader), lower visual
      /// fidelity. Same shader roles and resource layout, just less work per dispatch.
      bool performance_mode = false;
      /// Percent of target_fps the adaptive phase math internally aims for (50..100,
      /// default 100 = no margin). Values below 100 deliberately have the interpolator
      /// undershoot the requested rate, giving every pacing decision derived from
      /// frame_dur (the min-gen-ratio cutoff, the phase quantization grid, the
      /// near-end-of-window hold threshold) a bit more slack before hitting its edge
      /// case -- helps with jitter right at a borderline source:target ratio. Does not
      /// change target_fps itself (still reported/logged as requested).
      int target_fps_cutoff_percent = 100;
    };

    ~lsfg_framegen_t();

    /**
     * @brief Locate Lossless.dll from the configured Lossless Scaling install (or common installs).
     * @return Path to Lossless.dll, or std::nullopt when not found.
     */
    static std::optional<std::filesystem::path> find_lossless_dll();

    /**
     * @brief Create the interpolator and build the LSFG pipeline.
     * @param device Capture D3D11 device.
     * @param ctx Immediate context of @p device (capture thread only).
     * @param width Capture width in pixels.
     * @param height Capture height in pixels.
     * @param capture_format DXGI format of captured frames.
     * @return Interpolator instance, or nullptr when unavailable (reason logged).
     */
    static std::unique_ptr<lsfg_framegen_t> create(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      std::uint32_t width,
      std::uint32_t height,
      DXGI_FORMAT capture_format,
      const options_t &options
    );

    /**
     * @brief Queue a copy of the freshly captured frame into the internal latest-frame texture.
     * Safe to call while the producer's keyed mutex is held: this only issues CopyResource.
     * @param src Captured texture (same device, same size/format as configured).
     */
    void stage_capture(ID3D11Texture2D *src);

    /**
     * @brief On new content, rotate sources and run the optical-flow pre-pass.
     * Call after the producer mutex has been released.
     * @param frame_qpc QPC timestamp of the captured frame (0 if unknown).
     * @param frame_dirty Whether the capture backend reported this frame as a genuine content
     *   change (WGC DirtyRegionMode) rather than a compositor-only republish. Always true when
     *   the backend can't report it, which preserves the prior always-distinct behavior.
     */
    void commit_capture(std::uint64_t frame_qpc, bool frame_dirty);

    /**
     * @brief Decide what the current pacing slot should show.
     *
     * Must stay wall-clock based, not pacing-tick-counted: the capture loop's
     * frame-pacing "metronome" (display_base.cpp) can call this twice in a single
     * loop iteration when a pacing group busts, so counting calls as a proxy for
     * elapsed time desyncs from reality. Wall-clock elapsed/interval division is
     * self-correcting no matter how many times or how irregularly this is called.
     * @param now Current steady-clock time.
     * @param phase_out Set to the interpolation phase in (0,1) when returning true.
     * @return true when a generated frame should be produced at @p phase_out;
     *         false when the latest captured frame should be passed through.
     */
    bool want_generated(std::chrono::steady_clock::time_point now, float &phase_out);

    /**
     * @brief Run the generation tail at @p phase and blit the result into @p rtv.
     * @param phase Interpolation phase in (0,1).
     * @param rtv Render target view of the destination texture.
     * @param out_width Destination width in pixels.
     * @param out_height Destination height in pixels.
     * @return true on success.
     */
    bool render_generated(float phase, ID3D11RenderTargetView *rtv, std::uint32_t out_width, std::uint32_t out_height);

    /**
     * @brief Update the option(s) that don't require rebuilding the GPU pipeline.
     * Safe to call mid-stream with zero disruption. Everything else in options_t
     * (flow_scale, queue_frames, performance_mode) is baked into fixed-size textures
     * and shader/dispatch selection at create() time; changing those requires
     * destroying and recreating the whole lsfg_framegen_t instance instead.
     * @param max_multiplier New adaptive phase cap (clamped 2..20, as in create()).
     * @param target_fps_cutoff_percent New target-fps margin (clamped 50..100, as in create()).
     */
    void update_live_options(int max_multiplier, int target_fps_cutoff_percent);

    /**
     * @brief Frame to show when not actively generating: the newest raw capture in extrapolate
     * mode (queue_frames == 0), or the active buffered pair's newer frame once queued mode has
     * a confirmed pair (queue_frames >= 1) -- falling back to the newest raw capture until then.
     * Never null after the first stage_capture().
     */
    ID3D11Texture2D *latest_texture() const;

    /**
     * @brief Whether at least one frame has been captured (latest_texture() is valid).
     */
    bool has_frame() const;

    /**
     * @brief Whether latest_texture() may return content not yet shown as a genuine
     * pass-through (as opposed to a generated/extrapolated approximation of it).
     * Set on every real arrival (commit_capture()), cleared by mark_passthrough_shown().
     * Lets the caller settle on the true final frame once generation stops being due,
     * instead of leaving the last generated frame on screen indefinitely once the
     * source stops producing new content.
     */
    bool has_new_passthrough_frame() const;

    /**
     * @brief Record that the caller just displayed latest_texture() as a genuine
     * pass-through, clearing has_new_passthrough_frame() until the next real arrival
     * (or queued-mode active-pair advance).
     */
    void mark_passthrough_shown();

  private:
    lsfg_framegen_t();

    struct impl_t;
    std::unique_ptr<impl_t> _impl;
  };

}  // namespace platf::dxgi

#endif  // _WIN32
