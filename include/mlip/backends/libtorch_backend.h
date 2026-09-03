/**
 * libtorch_backend.h
 *
 * LibTorch/TorchScript runtime backend.
 * Loads a .pt model, wraps MD raw GPU pointers as non-owning torch::Tensor views,
 * enqueues inference on the MD CUDA stream, and returns without synchronizing.
 *
 * Only libtorch_backend.cpp (and adapters that explicitly include torch headers)
 * may use torch:: types.  This header exposes NO at::Tensor / torch:: types.
 */

#pragma once
#ifndef LIBTORCH_BACKEND_H
#define LIBTORCH_BACKEND_H

#include "../../mlip/mlip_types.h"

#include <memory>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

// ---------------------------------------------------------------------------
struct LibTorchConfig {
    std::string model_path;
    int         device_id    = 0;
    bool        use_float32  = true;
    bool        share_stream = true;
    bool        optimize_for_inference = true;
};

// ---------------------------------------------------------------------------
// Opaque forward result — no torch:: types visible outside LibTorch TUs.
// ---------------------------------------------------------------------------
class LibTorchBackend {
public:
    // Opaque handle for the model output.  Internally holds torch::jit::IValue.
    struct ForwardResult {
        struct Impl;
        std::shared_ptr<Impl> impl;
    };

    LibTorchBackend();
    ~LibTorchBackend();

    void load(const LibTorchConfig& cfg);
    bool is_loaded() const;

    /// Run forward pass.
    /// inputs_ptr — pointer to std::vector<torch::jit::IValue> (adapter builds this)
    ForwardResult run_forward(void* inputs_ptr, cudaStream_t stream);

    int device_id() const { return cfg_.device_id; }

private:
    struct BackendImpl;
    std::unique_ptr<BackendImpl> impl_;
    LibTorchConfig cfg_;
};

#endif // LIBTORCH_BACKEND_H
