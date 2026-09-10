/**
 * backend_interface.h
 *
 * Pure abstract interface for all execution backends (PyTorch, ONNX, T    virtual void execute(
        const std::vector<BackendTensorView>& inputs,
        std::vector<BackendTensorView>&       outputs,
        cudaStream_t                          stream
    ) = 0;lugin).
 * MD core must NOT include any framework-specific headers.
 */

#pragma once
#ifndef MLIP_BACKEND_INTERFACE_H
#define MLIP_BACKEND_INTERFACE_H

#include "../mlip_types.h"
#include "../runtime/model_specification.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlip {

// ─────────────────── lightweight tensor view ─────────────────────────────────
// Non-owning reference to a GPU or host buffer passed to/from backend::execute().
// Uses char[] instead of std::string so the struct is ABI-neutral
// (can be passed between TUs compiled with different _GLIBCXX_USE_CXX11_ABI).

struct BackendTensorView {
    void*      data       = nullptr;
    mlip::ScalarType dtype = mlip::ScalarType::Float64;
    int        ndim       = 0;
    int64_t    shape[4]   = {};
    int64_t    stride[4]  = {};
    int        device_id  = 0;   // -1 = host
    bool       requires_grad = false;
    char       name[64]   = {};  // for dict-style forward; empty for positional
};

// ─────────────────── backend model metadata ──────────────────────────────────

struct BackendInputMeta {
    char        name[64]  = {};
    int         index = -1;
    mlip::ScalarType  dtype = mlip::ScalarType::Float32;
};

struct BackendOutputMeta {
    char        name[64]  = {};
    int         index = -1;
    mlip::ScalarType  dtype = mlip::ScalarType::Float32;
};

struct BackendModelMetadata {
    std::vector<BackendInputMeta>  inputs;
    std::vector<BackendOutputMeta> outputs;
    bool supports_dict_input   = false;
    bool supports_positional   = true;
    bool supports_named_output = false;
    bool supports_tuple_output = true;
};

// ─────────────────── backend configuration ───────────────────────────────────
// All strings are fixed char[] to remain ABI-neutral across the ABI=0/ABI=1 boundary.

struct BackendConfiguration {
    char        model_path[512] = {};
    char        entrypoint[64]  = "forward";
    int         device_id    = 0;
    bool        use_fp32     = true;
    bool        share_cuda_stream = true;
    bool        allow_cpu_fallback = false;
    CallStyle   call_style   = CallStyle::Positional;
};

// ─────────────────── abstract backend ────────────────────────────────────────

class IMlipBackend {
public:
    virtual ~IMlipBackend() = default;

    /**
     * Load model and prepare runtime on the given device.
     * Throws std::runtime_error on failure.
     */
    virtual void initialize(
        const BackendConfiguration& config,
        const MlipExecutionContext& context
    ) = 0;

    /**
     * Introspect model input/output metadata (optional, may return empty).
     * Called once at init time for validation.
     */
    virtual BackendModelMetadata inspect_model() = 0;

    /**
     * Execute the model.
     * inputs  — views already prepared by the preprocessing pipeline.
     * outputs — writable views (allocated by MlipRuntime scratch/persistent buffers).
     * stream  — MD CUDA stream; backend must enqueue on this stream.
     * Does NOT call cudaDeviceSynchronize internally.
     */
    virtual void execute(
        const std::vector<BackendTensorView>& inputs,
        std::vector<BackendTensorView>&       outputs,
        cudaStream_t                          stream
    ) = 0;

    virtual void synchronize() = 0;

    virtual const char* name() const = 0;
};

// ─────────────────── backend registry ────────────────────────────────────────

class BackendRegistry {
public:
    using Creator = std::function<std::unique_ptr<IMlipBackend>()>;

    static BackendRegistry& instance();

    void register_backend(const std::string& name, Creator c) {
        if (creators_.count(name))
            throw std::runtime_error("[BackendRegistry] Duplicate: " + name);
        creators_[name] = std::move(c);
    }

    std::unique_ptr<IMlipBackend> create(const std::string& name) const {
        auto it = creators_.find(name);
        if (it == creators_.end())
            throw std::runtime_error("[BackendRegistry] Unknown backend '" + name +
                "'. Registered: " + list());
        return it->second();
    }

    bool has(const std::string& name) const { return creators_.count(name) > 0; }

    std::string list() const {
        std::string s;
        for (auto& [k, _] : creators_) { if (!s.empty()) s += ", "; s += k; }
        return s;
    }

private:
    BackendRegistry() = default;
    std::unordered_map<std::string, Creator> creators_;
};

struct BackendAutoRegister {
    BackendAutoRegister(const std::string& name, BackendRegistry::Creator c) {
        BackendRegistry::instance().register_backend(name, std::move(c));
    }
};

#define REGISTER_MLIP_BACKEND(name, ClassName)                                 \
    static mlip::BackendAutoRegister _mlip_backend_autoreg_##ClassName(        \
        name,                                                                   \
        []() -> std::unique_ptr<mlip::IMlipBackend> {                          \
            return std::make_unique<ClassName>();                               \
        })

} // namespace mlip
#endif // MLIP_BACKEND_INTERFACE_H
