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
   * (LSFG "adaptive" mode).
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
      /// Use Lossless Scaling's "performance" optical-flow shader set instead of "quality".
      /// Lighter/faster (fewer temporal-history texture bindings per shader), lower visual
      /// fidelity. Same shader roles and resource layout, just less work per dispatch.
      bool performance_mode = false;
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
     */
    void commit_capture(std::uint64_t frame_qpc);

    /**
     * @brief Whether a newly staged capture should be committed after the
     * current output slot, preserving the source pair used by that slot.
     */
    bool defer_capture_commit() const;

    /**
     * @brief Decide what the current pacing slot should show.
     *
     * Must stay wall-clock based, not pacing-tick-counted: the capture loop can
     * call this twice in one iteration when a pacing group busts, so counting
     * calls as a proxy for elapsed time desyncs from reality (tried and reverted).
     * @param now Current steady-clock time.
     * @param phase_out Set to the interpolation phase in (0,1) when returning true.
     * @return true when a generated frame should be produced at @p phase_out;
     *         false when the caller should either pass through the newest real
     *         frame or hold its current output (query has_new_passthrough_frame()
     *         to distinguish those cases).
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
     * @brief Return one completed GPU duration for a generated frame, without
     * waiting for the GPU. Timing is intentionally asynchronous so overload
     * detection never stalls capture.
     */
    std::optional<std::chrono::nanoseconds> poll_generated_gpu_time();

    /** @brief True when every non-blocking generated-frame timing slot is pending. */
    bool generated_gpu_timing_backlogged() const;

    /**
     * @brief Insert a non-blocking fence after all LSFG commands submitted for
     * the current slot. The caller invokes this after staged captures commit.
     */
    void submit_gpu_work_fence();

    /**
     * @brief Whether the oldest LSFG fence is still pending after @p budget.
     * Completed fences are retired without flushing or waiting for the GPU.
     */
    bool gpu_work_overdue(std::chrono::steady_clock::time_point now, std::chrono::nanoseconds budget);

    /** @brief Number of outstanding non-blocking LSFG work fences. */
    std::size_t pending_gpu_work_fences() const;

    /** @brief Duration of one requested output-frame slot. */
    std::chrono::nanoseconds target_frame_duration() const;

    /**
     * @brief Record that the generated frame selected by want_generated() was shown.
     */
    void mark_generated_shown();

    /**
     * @brief Update the option(s) that don't require rebuilding the GPU pipeline.
     * Safe to call mid-stream with zero disruption. Everything else in options_t
     * (flow_scale, performance_mode) is baked into fixed-size textures and
     * shader/dispatch selection at create() time; changing those requires
     * destroying and recreating the whole lsfg_framegen_t instance instead.
     * @param max_multiplier New adaptive phase cap (clamped 2..20, as in create()).
     */
    void update_live_options(int max_multiplier);

    /**
     * @brief Frame to show when not actively generating: the newest raw capture.
     * Never null after the first stage_capture().
     */
    ID3D11Texture2D *latest_texture() const;

    /**
     * @brief Source texture selected by the most recent output decision.
     * Usually the newest capture; at an exact timeline boundary it can be the
     * preceding source frame so the capture is not presented early.
     */
    ID3D11Texture2D *passthrough_texture() const;

    /**
     * @brief Whether at least one frame has been captured (latest_texture() is valid).
     */
    bool has_frame() const;

    /**
     * @brief Whether the current output decision permits a real-frame pass-through.
     * Set on every real arrival (commit_capture()), but suppressed when the delayed
     * output timeline has not reached that frame pair yet. Cleared by
     * mark_passthrough_shown(). This lets the caller settle on a true source frame
     * once it is due without presenting it early.
     */
    bool has_new_passthrough_frame() const;

    /**
     * @brief Record that the caller just displayed passthrough_texture() as a
     * genuine pass-through. This clears the pending newest-frame state only
     * when that newest frame was the one displayed.
     */
    void mark_passthrough_shown();

  private:
    lsfg_framegen_t();

    struct impl_t;
    std::unique_ptr<impl_t> _impl;
  };

}  // namespace platf::dxgi

#endif  // _WIN32
