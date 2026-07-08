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
     * @brief Hash + dedup the staged frame; on new content rotate sources and run the flow pre-pass.
     * Call after the producer mutex has been released.
     * @param frame_qpc QPC timestamp of the captured frame (0 if unknown).
     */
    void commit_capture(std::uint64_t frame_qpc);

    /**
     * @brief Decide what the current pacing slot should show.
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
     * @brief Latest captured frame (pass-through source). Never null after the first stage_capture().
     */
    ID3D11Texture2D *latest_texture() const;

    /**
     * @brief Whether at least one frame has been captured (latest_texture() is valid).
     */
    bool has_frame() const;

  private:
    lsfg_framegen_t();

    struct impl_t;
    std::unique_ptr<impl_t> _impl;
  };

}  // namespace platf::dxgi

#endif  // _WIN32
