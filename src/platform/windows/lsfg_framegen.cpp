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

    // Real-frame ping-pong pool: distinct frame N lands in sources[N % pool_size].
    std::vector<tex_t> sources;
    std::size_t pool_size = 2;
    // render-owned copy of the latest captured frame, shown during pass-through
    tex_t latest;

    std::vector<step_t> prepass;
    std::vector<std::vector<step_t>> passes;
    std::vector<tex_t> outputs;
    std::vector<com_ptr<ID3D11Buffer>> cbp;  // per-pass constant buffers, updatable for adaptive

    blitter_t blitter;

    // adaptive timing state: extrapolate phase from the newest arrival
    std::size_t frames_seen = 0;  // count of DISTINCT source frames consumed
    bool has_capture = false;
    std::uint64_t last_distinct_qpc = 0;
    std::chrono::steady_clock::time_point last_arrival {};
    std::chrono::nanoseconds src_interval = 16ms;  // EMA between distinct frames
    std::chrono::nanoseconds frame_dur = 16ms;  // one client-requested frame interval

    // Set on each real arrival, cleared once shown as a genuine pass-through: lets the
    // caller settle on the true final frame once generation stops being due, instead of
    // leaving the last generated approximation on screen forever. See mark_passthrough_shown().
    bool passthrough_dirty = true;

    // Optical-flow pre-pass output. a1_out/b1_imgs are a rolling temporal history the
    // shaders expect fed one real arrival at a time with no gaps (alpha1's per-level slot
    // rotation, beta0's 3-way temporal blend).
    std::array<ext_t, 7> a1_extent {};
    std::array<std::vector<std::vector<tex_t>>, 7> a1_out;
    std::vector<tex_t> b1_imgs;

    ~impl_t() {
      if (lossless_module) {
        FreeLibrary(lossless_module);
      }
    }

    bool build_pipeline();
    void run_steps(const std::vector<step_t> &steps, std::size_t fidx);
    void recompute_frame_dur();
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

    // Perf-mode shaders declare HALF the per-group texture bindings, so the groups
    // scaled by `m` below (matching lsfg-vk's `m = perf ? 1 : 2`) must shrink to
    // match; feeding them quality-sized groups misaligns every SRV/UAV slot and
    // produces garbage flow. Groups left at fixed counts match lsfg-vk for both modes.
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
    // alpha1 keeps temporal history: 3 slots for level 0, 2 otherwise.
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

    // --- beta1: blur chain -> 6 R8 pyramid levels ---
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

    // --- per generated frame: gamma / delta pyramids + generate ---
    for (std::size_t p = 0; p < count; ++p) {
      ID3D11Buffer *cb = cbp[p].get();
      auto &gen_a1_out = a1_out;
      auto &gen_b1_imgs = b1_imgs;
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

      // generate: RGBA16F for HDR streams, RGBA8 otherwise (mandatory UAV store support).
      auto out = b.tex(source_extent, hdr ? rgba16f : rgba8);
      {
        step_t step;
        for (std::size_t v = 0; v < pool_size; ++v) {
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
    impl.inv_flow = 1.0f / impl.options.flow_scale;
    impl.recompute_frame_dur();
    impl.pool_size = 2;

    impl.lossless_module = LoadLibraryExW(dll_path->c_str(), nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (!impl.lossless_module) {
      BOOST_LOG(warning) << "LSFG: failed to load Lossless.dll as a resource module; frame generation disabled";
      return nullptr;
    }

    impl.sources.resize(impl.pool_size);
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
                    << ", flow scale " << impl.options.flow_scale
                    << ", cap " << impl.options.max_multiplier << "x, "
                    << (impl.options.performance_mode ? "performance" : "quality") << " shaders, "
                    << (hdr ? "HDR" : "SDR") << ", shaders from " << dll_path->string() << ']';
    return self;
  }

  void lsfg_framegen_t::stage_capture(ID3D11Texture2D *src) {
    auto &impl = *_impl;
    impl.ctx->CopyResource(impl.latest.tex.get(), src);
    impl.has_capture = true;
  }

  void lsfg_framegen_t::commit_capture(std::uint64_t frame_qpc, bool /*frame_dirty*/) {
    auto &impl = *_impl;
    if (!impl.has_capture) {
      return;
    }

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

    // Anchor the phase-slot grid to the frame's own QPC capture timestamp, not
    // now(): WGC delivery jitter, the effective_wgc_timeout() wait, and IPC handoff
    // add variable delay between capture and this call, which would smear the grid
    // and misclassify pacing ticks. Falls back to now() if QPC is unavailable.
    impl.last_arrival = frame_qpc != 0
      ? std::chrono::steady_clock::now() - platf::qpc_time_difference(platf::qpc_counter(), static_cast<int64_t>(frame_qpc))
      : std::chrono::steady_clock::now();
    if (impl.frames_seen > 0) {
      // one flow pre-pass per distinct frame; generation tails reuse it every phase
      impl.run_steps(impl.prepass, impl.frames_seen);
    }
    ++impl.frames_seen;
  }

  bool lsfg_framegen_t::want_generated(std::chrono::steady_clock::time_point now, float &phase_out) {
    auto &impl = *_impl;

    if (impl.frames_seen < 2) {
      return false;
    }

    // Only generate if a source interval spans enough present slots to fit an
    // in-between frame; below this the source is already near the target fps.
    const double min_gen_ratio = 1.5;
    if (std::chrono::duration<double>(impl.src_interval).count() < std::chrono::duration<double>(impl.frame_dur).count() * min_gen_ratio) {
      return false;
    }

    // Cap the extrapolated gap at `max_multiplier` intervals so a stalled source
    // holds the latest real frame instead of extrapolating across the whole gap.
    //
    // Stays wall-clock based, never tick-counted: the capture loop can call this
    // twice per iteration on a pacing bust, which desyncs a call counter from real
    // time. Tried tick-counting and reverted (visibly worse "rewinding" judder).
    const auto capped_interval = std::min(impl.src_interval, impl.frame_dur * impl.options.max_multiplier);

    // Slots per interval, rounded to a whole tick count (not divided by src_interval
    // directly) so the noisy EMA can't make the bucket count jitter tick-to-tick at
    // ratios near a small integer (e.g. 60fps into 116fps).
    std::int64_t steps = 1;
    if (const auto frame_dur_ns = impl.frame_dur.count(); frame_dur_ns > 0) {
      steps = std::max<std::int64_t>(1, (capped_interval.count() + frame_dur_ns / 2) / frame_dur_ns);
    }
    const long steps_i = static_cast<long>(steps);

    const auto elapsed = std::chrono::duration<float>(now - impl.last_arrival).count();
    const auto frame_dur_s = std::max(std::chrono::duration<float>(impl.frame_dur).count(), 1e-4f);

    // Bucket by tick number (floor), not by rounding a continuous fraction to the
    // nearest slot: ticks land ~frame_dur apart, so flooring puts consecutive ticks
    // in consecutive buckets, whereas round-to-nearest (boundaries at half a slot)
    // can collapse two ticks into the same bucket on ordinary drift -- duplicating
    // one warp phase and dropping another slot, sometimes the real-frame slot.
    long slot = static_cast<long>(std::floor(elapsed / frame_dur_s)) + 1;
    slot = std::clamp<long>(slot, 1, steps_i);
    if (slot >= steps_i) {
      return false;
    }

    phase_out = static_cast<float>(slot) / static_cast<float>(steps_i);
    return true;
  }

  bool lsfg_framegen_t::render_generated(float phase, ID3D11RenderTargetView *rtv, std::uint32_t out_width, std::uint32_t out_height) {
    auto &impl = *_impl;
    if (impl.frames_seen < 2 || impl.passes.empty() || !rtv) {
      return false;
    }

    const auto data = make_cb_data(phase, impl.inv_flow, impl.hdr);
    impl.ctx->UpdateSubresource(impl.cbp[0].get(), 0, nullptr, &data, 0, 0);

    const std::size_t fidx = impl.frames_seen - 1;
    impl.run_steps(impl.passes[0], fidx);
    impl.blitter.blit(impl.ctx.get(), impl.outputs[0].srv.get(), rtv, out_width, out_height);
    return true;
  }

  void lsfg_framegen_t::impl_t::recompute_frame_dur() {
    if (options.target_fps > 0.0) {
      frame_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / options.target_fps));
    }
  }

  void lsfg_framegen_t::update_live_options(int max_multiplier) {
    auto &impl = *_impl;
    impl.options.max_multiplier = std::clamp(max_multiplier, 2, 20);
  }

  ID3D11Texture2D *lsfg_framegen_t::latest_texture() const {
    return _impl->latest.tex.get();
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
