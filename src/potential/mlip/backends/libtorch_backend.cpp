// Include mlip_types first. DO NOT "using namespace mlip" here —
// that would cause ::ScalarType vs c10::ScalarType ambiguity in torch headers.
#include "mlip/mlip_types.h"
#include "mlip/backends/libtorch_backend.h"

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#endif

#include <stdexcept>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// ForwardResult::Impl — holds the raw IValue output (opaque to callers)
// ---------------------------------------------------------------------------
struct LibTorchBackend::ForwardResult::Impl {
#ifdef USE_LIBTORCH
    torch::jit::IValue value;
#endif
};

// ---------------------------------------------------------------------------
// BackendImpl — Pimpl struct that owns the loaded TorchScript module
// ---------------------------------------------------------------------------
struct LibTorchBackend::BackendImpl {
#ifdef USE_LIBTORCH
    torch::jit::script::Module model;
    bool loaded = false;
#else
    bool loaded = false;
#endif
};

// ---------------------------------------------------------------------------
LibTorchBackend::LibTorchBackend()
    : impl_(std::make_unique<BackendImpl>())
{}

LibTorchBackend::~LibTorchBackend() = default;

// ---------------------------------------------------------------------------
void LibTorchBackend::load(const LibTorchConfig& cfg)
{
    cfg_ = cfg;
#ifdef USE_LIBTORCH
    try {
        std::cout << "[LibTorchBackend] Loading model: " << cfg.model_path
                  << "  device=cuda:" << cfg.device_id << "\n";

        torch::Device dev(torch::kCUDA, cfg.device_id);
        impl_->model = torch::jit::load(cfg.model_path, dev);
        impl_->model.eval();

        if (cfg.optimize_for_inference) {
            torch::jit::optimize_for_inference(impl_->model);
        }

        impl_->loaded = true;
        std::cout << "[LibTorchBackend] Model loaded successfully.\n";
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            std::string("[LibTorchBackend] Failed to load model: ") + e.what());
    }
#else
    (void)cfg;
    throw std::runtime_error(
        "[LibTorchBackend] Built without LibTorch (USE_LIBTORCH not defined).");
#endif
}

bool LibTorchBackend::is_loaded() const {
    return impl_->loaded;
}

// ---------------------------------------------------------------------------
LibTorchBackend::ForwardResult
LibTorchBackend::run_forward(void* inputs_ptr, cudaStream_t stream)
{
    ForwardResult res;
    res.impl = std::make_shared<ForwardResult::Impl>();
#ifdef USE_LIBTORCH
    if (!impl_->loaded)
        throw std::runtime_error("[LibTorchBackend] Model not loaded.");

    auto* inputs = static_cast<std::vector<torch::jit::IValue>*>(inputs_ptr);

    if (cfg_.share_stream && stream != nullptr) {
        at::cuda::CUDAStream torch_stream =
            at::cuda::getStreamFromExternal(
                static_cast<cudaStream_t>(stream),
                cfg_.device_id);
        at::cuda::CUDAStreamGuard guard(torch_stream);
        res.impl->value = impl_->model.forward(*inputs);
    } else {
        res.impl->value = impl_->model.forward(*inputs);
    }
#else
    (void)inputs_ptr; (void)stream;
    throw std::runtime_error(
        "[LibTorchBackend] Built without LibTorch.");
#endif
    return res;
}
