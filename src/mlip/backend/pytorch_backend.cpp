/**
 * // Include mlip_types BEFORE LibTorch to ensure mlip types are defined.
// DO NOT use "using namespace mlip" here — it would cause ScalarType ambiguity
// with c10::ScalarType when torch headers are processed.
#include "mlip/mlip_types.h"
#include "mlip/backend/pytorch_backend.h"

#ifdef USE_LIBTORCH
#  include <torch/torch.h>
#  include <torch/script.h>
#  include <ATen/cuda/CUDAContext.h>
#  include <c10/cuda/CUDAGuard.h>
#endifckend.cpp
 *
 * PyTorch / TorchScript backend implementation.
 * All torch:: code is confined to this translation unit.
 */

// Include mlip_types BEFORE LibTorch to define ::ScalarType before c10::ScalarType
#include "mlip/mlip_types.h"
#include "mlip/backend/pytorch_backend.h"

#ifdef USE_LIBTORCH
#  include <torch/torch.h>
#  include <torch/script.h>
#  include <ATen/cuda/CUDAContext.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <c10/cuda/CUDACachingAllocator.h>
#endif

#ifdef USE_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess) \
    throw std::runtime_error(std::string("[PyTorchBackend] CUDA: ")+cudaGetErrorString(_e)); } while(0)
#else
#  define CUDA_CHECK(e) (void)(e)
#endif

#include "mlip/backend/backend_interface.h"  // REGISTER_MLIP_BACKEND
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace mlip {

// Self-register is intentionally NOT done here because pytorch_backend.cpp is
// compiled with _GLIBCXX_USE_CXX11_ABI=0 (to match LibTorch), while the
// BackendRegistry singleton lives in ABI=1 TUs (execution_plan.cpp, etc.).
// Cross-ABI registration is performed by pytorch_backend_register.cpp (ABI=1)
// via the C-linkage factory below.

// ─────────────────────── Pimpl ───────────────────────────────────────────────

struct PyTorchBackend::Impl {
#ifdef USE_LIBTORCH
    torch::jit::script::Module model;
    bool loaded = false;
    c10::DeviceType device_type = c10::kCUDA;
    int device_id = 0;
    CallStyle call_style = CallStyle::Positional;
    // Keep last result alive so tv.data pointers remain valid until next execute()
    torch::jit::IValue last_result;
    // Keep contiguous output tensors alive across execute() → postprocessing
    std::vector<at::Tensor> output_tensors;
#endif
};

// ─────────────────────── helpers ─────────────────────────────────────────────

#ifdef USE_LIBTORCH
static c10::ScalarType to_torch_dtype(ScalarType dt) {
    switch (dt) {
        case ScalarType::Float32: return torch::kFloat32;
        case ScalarType::Float64: return torch::kFloat64;
        case ScalarType::Int32:   return torch::kInt32;
        case ScalarType::Int64:   return torch::kInt64;
        default: return torch::kFloat32;
    }
}

static ScalarType from_torch_dtype(c10::ScalarType dt) {
    if (dt == torch::kFloat32) return ScalarType::Float32;
    if (dt == torch::kFloat64) return ScalarType::Float64;
    if (dt == torch::kInt32)   return ScalarType::Int32;
    if (dt == torch::kInt64)   return ScalarType::Int64;
    return ScalarType::Float32;
}

/// Wrap a BackendTensorView as a non-owning torch::Tensor on the GPU.
/// Wrap a BackendTensorView as a torch::Tensor.
/// model_device_id < 0 means CPU model — data is on CUDA, so cudaMemcpy to CPU tensor.
/// model_device_id >= 0 means CUDA model — use from_blob (no copy).
static torch::Tensor wrap_tensor(const BackendTensorView& tv, int model_device_id) {
    if (!tv.data)
        throw std::runtime_error(std::string("[PyTorchBackend] Null data pointer in BackendTensorView '") +
                                  tv.name + "'");

    c10::ScalarType dtype = to_torch_dtype(tv.dtype);

    // Build shape
    std::vector<int64_t> shape(tv.ndim);
    size_t elems = 1;
    for (int i = 0; i < tv.ndim; ++i) { shape[i] = tv.shape[i]; elems *= shape[i]; }
    size_t bytes = elems * c10::elementSize(dtype);

    torch::Tensor t;
    if (model_device_id < 0) {
        // CPU model: check if source data is CUDA or CPU memory
        auto cpu_opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        t = torch::empty(shape, cpu_opts);

        cudaPointerAttributes attr;
        cudaError_t qerr = cudaPointerGetAttributes(&attr, tv.data);
        if (qerr == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            // Source is device (CUDA) memory → cudaMemcpy DtoH
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess)
                throw std::runtime_error(std::string("[PyTorchBackend] cudaDeviceSynchronize failed for '") +
                                          tv.name + "': " + cudaGetErrorString(sync_err));
            cudaError_t err = cudaMemcpy(t.data_ptr(), tv.data, bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("[PyTorchBackend] cudaMemcpy DtoH failed for '") +
                                          tv.name + "': " + cudaGetErrorString(err));
        } else {
            // Source is host (CPU) memory → plain memcpy
            std::memcpy(t.data_ptr(), tv.data, bytes);
        }
    } else {
        // CUDA model: zero-copy from_blob on the correct GPU
        torch::Device dev(torch::kCUDA, model_device_id);
        auto opts = torch::TensorOptions().dtype(dtype).device(dev);
        t = torch::from_blob(tv.data, shape, opts);
    }

    if (tv.requires_grad)
        t.set_requires_grad(true);

    return t;
}
#endif // USE_LIBTORCH

// ─────────────────────── constructor / destructor ────────────────────────────

PyTorchBackend::PyTorchBackend()
    : impl_(std::make_unique<Impl>())
{}

PyTorchBackend::~PyTorchBackend() = default;

// ─────────────────────── initialize ──────────────────────────────────────────

void PyTorchBackend::initialize(
    const BackendConfiguration& config,
    const MlipExecutionContext& context)
{
    cfg_ = config;
    ctx_ = context;

#ifdef USE_LIBTORCH
    // Determine target device from spec's model.device field (stored in config.device_id=-1 for CPU)
    const bool use_cpu = (config.device_id < 0);
    torch::Device dev = use_cpu ? torch::Device(torch::kCPU)
                                 : torch::Device(torch::kCUDA, config.device_id);

    std::cout << "[PyTorchBackend] Loading " << config.model_path
              << " on " << (use_cpu ? "cpu" : "cuda:" + std::to_string(config.device_id)) << "\n";

    try {
        impl_->model = torch::jit::load(config.model_path, dev);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            std::string("[PyTorchBackend] Failed to load '") + config.model_path + "': " + e.what());
    }

    impl_->model.eval();
    // Disable parameter gradients
    for (const auto& p : impl_->model.parameters())
        const_cast<at::Tensor&>(p).set_requires_grad(false);

    // NOTE: optimize_for_inference is intentionally skipped here.
    // MLIP models such as MACE and NequIP use torch.autograd.grad internally
    // to compute atomic forces from the energy. optimize_for_inference may fold
    // constant operations in a way that breaks the autograd graph (e.g., dtype
    // mismatches after constant-folding on CUDA). Running eval() + disabling
    // parameter gradients above is sufficient for inference performance.
    // torch::jit::optimize_for_inference(impl_->model);  // intentionally disabled

    impl_->loaded     = true;
    impl_->device_id  = config.device_id;
    impl_->call_style = config.call_style;

    if (!use_cpu) {
        // Release any temporary GPU memory allocated during model loading
        // (e.g. JIT kernel compilation buffers) back to the CUDA driver.
        c10::cuda::CUDACachingAllocator::emptyCache();

        // For GPU inference, limit CPU-side PyTorch threads to 1.
        // With 56 OMP threads, PyTorch's CPU dispatch overhead dominates
        // the hot path even when the model runs entirely on CUDA.
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
    }

    std::cout << "[PyTorchBackend] Model loaded.\n";
#else
    (void)config; (void)context;
    throw std::runtime_error(
        "[PyTorchBackend] Compiled without LibTorch (USE_LIBTORCH not set).");
#endif
}

// ─────────────────────── inspect_model ───────────────────────────────────────

BackendModelMetadata PyTorchBackend::inspect_model() {
    BackendModelMetadata meta;
    // TorchScript does not expose named I/O by default; return empty metadata.
    meta.supports_positional   = true;
    meta.supports_dict_input   = true;
    meta.supports_named_output = true;
    meta.supports_tuple_output = true;
    return meta;
}

// ─────────────────────── execute ─────────────────────────────────────────────

void PyTorchBackend::execute(
    const std::vector<BackendTensorView>& inputs,
    std::vector<BackendTensorView>&       outputs,
    cudaStream_t                          stream)
{
#ifdef USE_LIBTORCH
    if (!impl_->loaded)
        throw std::runtime_error("[PyTorchBackend] Model not loaded.");

    // Connect MD CUDA stream to PyTorch (only for CUDA model)
    std::optional<at::cuda::CUDAStreamGuard> guard;
    if (impl_->device_id >= 0 && cfg_.share_cuda_stream && stream) {
        at::cuda::CUDAStream ts =
            at::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream),
                                             impl_->device_id);
        guard.emplace(ts);
    }

    // Build forward inputs
    std::vector<torch::jit::IValue> jit_inputs;

    if (impl_->call_style == CallStyle::Dictionary) {
        // Build IDict<string, Tensor>
        c10::Dict<std::string, torch::Tensor> d;
        for (auto& tv : inputs) {
            if (tv.name[0] != '\0')
                d.insert(tv.name, wrap_tensor(tv, impl_->device_id));
        }
        jit_inputs.push_back(d);
    } else {
        // Positional
        for (auto& tv : inputs) {
            jit_inputs.push_back(wrap_tensor(tv, impl_->device_id));
        }
    }

    // Forward pass
    // NOTE: All supported models (DeePMD, MACE, NequIP) use torch.autograd.grad
    // internally to compute forces from energy. Do NOT use NoGradGuard here.
    try {
        impl_->last_result = impl_->model.forward(jit_inputs);
    } catch (const c10::Error& e) {
        throw std::runtime_error("[PyTorchBackend] Forward failed: " + std::string(e.what()));
    }
    torch::jit::IValue& result = impl_->last_result;
    impl_->output_tensors.clear();  // release previous step's tensors

    // Helper: ensure output tensor is on the correct device (CUDA) for the
    // MD postprocess ops which write into CUDA buffers.
    const int out_device = (impl_->device_id >= 0) ? impl_->device_id : 0;
    auto ensure_cuda = [&](torch::Tensor t) -> torch::Tensor {
        if (!t.is_contiguous()) t = t.contiguous();
        if (impl_->device_id < 0) {
            // CPU model — move outputs to CUDA so postprocess CUDA ops can use them
            t = t.to(torch::Device(torch::kCUDA, out_device));
        }
        // Upcast float32 → float64 so postprocess ops (copy_forces, copy_energy) get float64
        if (t.scalar_type() == torch::kFloat32) {
            t = t.to(torch::kFloat64);
        }
        return t;
    };

    // Extract outputs
    if (result.isGenericDict()) {
        auto d = result.toGenericDict();
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto& tv = outputs[i];
            if (tv.name[0] == '\0') continue;
            auto it = d.find(tv.name);
            if (it == d.end()) {
                std::cerr << "[PyTorchBackend] Warning: output '" << tv.name << "' not found in dict.\n";
                continue;
            }
            auto t = ensure_cuda(it->value().toTensor());
            tv.data = t.data_ptr();
            tv.dtype = from_torch_dtype(t.scalar_type());
            tv.ndim = t.dim();
            for (int d = 0; d < tv.ndim && d < 4; ++d) tv.shape[d] = t.size(d);
            tv.device_id = out_device;
            impl_->output_tensors.push_back(std::move(t));  // keep alive
        }
    } else if (result.isTuple()) {
        auto tup = result.toTuple();
        for (size_t i = 0; i < outputs.size() && i < tup->elements().size(); ++i) {
            auto& tv = outputs[i];
            auto t = ensure_cuda(tup->elements()[i].toTensor());
            tv.data = t.data_ptr();
            tv.dtype = from_torch_dtype(t.scalar_type());
            tv.ndim = t.dim();
            for (int d = 0; d < tv.ndim && d < 4; ++d) tv.shape[d] = t.size(d);
            tv.device_id = out_device;
            impl_->output_tensors.push_back(std::move(t));  // keep alive
        }
    } else if (result.isTensor() && !outputs.empty()) {
        auto t = ensure_cuda(result.toTensor());
        auto& tv = outputs[0];
        tv.data = t.data_ptr();
        tv.dtype = from_torch_dtype(t.scalar_type());
        tv.ndim = t.dim();
        for (int d = 0; d < tv.ndim && d < 4; ++d) tv.shape[d] = t.size(d);
        tv.device_id = out_device;
        impl_->output_tensors.push_back(std::move(t));  // keep alive
    }

#else
    (void)inputs; (void)outputs; (void)stream;
    throw std::runtime_error("[PyTorchBackend] Built without LibTorch.");
#endif
}

// ─────────────────────── synchronize ─────────────────────────────────────────

void PyTorchBackend::synchronize() {
#ifdef USE_CUDA
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

} // namespace mlip

// C-linkage factory: callable from ABI=1 TUs (pytorch_backend_register.cpp)
// without depending on mangled C++ names.
extern "C" mlip::IMlipBackend* mlip_create_pytorch_backend_c() {
    return new mlip::PyTorchBackend();
}
