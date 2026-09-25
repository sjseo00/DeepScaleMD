
#include "force_libtorch/mlip_types.h"
#include "force_libtorch/backend/pytorch_backend.h"

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

#include "force_libtorch/backend/backend_interface.h"  // REGISTER_MLIP_BACKEND
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace mlip {



struct PyTorchBackend::Impl {
#ifdef USE_LIBTORCH
    torch::jit::script::Module model;
    bool loaded = false;
    c10::DeviceType device_type = c10::kCUDA;
    int device_id = 0;
    CallStyle call_style = CallStyle::Positional;
    torch::jit::IValue last_result;
    std::vector<at::Tensor> output_tensors;
#endif
};


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

static torch::Tensor wrap_tensor(const BackendTensorView& tv, int model_device_id) {
    if (!tv.data)
        throw std::runtime_error(std::string("[PyTorchBackend] Null data pointer in BackendTensorView '") +
                                  tv.name + "'");

    c10::ScalarType dtype = to_torch_dtype(tv.dtype);

    std::vector<int64_t> shape(tv.ndim);
    size_t elems = 1;
    for (int i = 0; i < tv.ndim; ++i) { shape[i] = tv.shape[i]; elems *= shape[i]; }
    size_t bytes = elems * c10::elementSize(dtype);

    torch::Tensor t;
    if (model_device_id < 0) {
        auto cpu_opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        t = torch::empty(shape, cpu_opts);

        cudaPointerAttributes attr;
        cudaError_t qerr = cudaPointerGetAttributes(&attr, tv.data);
        if (qerr == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess)
                throw std::runtime_error(std::string("[PyTorchBackend] cudaDeviceSynchronize failed for '") +
                                          tv.name + "': " + cudaGetErrorString(sync_err));
            cudaError_t err = cudaMemcpy(t.data_ptr(), tv.data, bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("[PyTorchBackend] cudaMemcpy DtoH failed for '") +
                                          tv.name + "': " + cudaGetErrorString(err));
        } else {
            std::memcpy(t.data_ptr(), tv.data, bytes);
        }
    } else {
        torch::Device dev(torch::kCUDA, model_device_id);
        auto opts = torch::TensorOptions().dtype(dtype).device(dev);
        t = torch::from_blob(tv.data, shape, opts);
    }

    if (tv.requires_grad)
        t.set_requires_grad(true);

    return t;
}
#endif // USE_LIBTORCH


PyTorchBackend::PyTorchBackend()
    : impl_(std::make_unique<Impl>())
{}

PyTorchBackend::~PyTorchBackend() = default;


void PyTorchBackend::initialize(
    const BackendConfiguration& config,
    const MlipExecutionContext& context)
{
    cfg_ = config;
    ctx_ = context;

#ifdef USE_LIBTORCH
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
    for (const auto& p : impl_->model.parameters())
        const_cast<at::Tensor&>(p).set_requires_grad(false);


    impl_->loaded     = true;
    impl_->device_id  = config.device_id;
    impl_->call_style = config.call_style;

    if (!use_cpu) {
        c10::cuda::CUDACachingAllocator::emptyCache();

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


BackendModelMetadata PyTorchBackend::inspect_model() {
    BackendModelMetadata meta;
    meta.supports_positional   = true;
    meta.supports_dict_input   = true;
    meta.supports_named_output = true;
    meta.supports_tuple_output = true;
    return meta;
}


void PyTorchBackend::execute(
    const std::vector<BackendTensorView>& inputs,
    std::vector<BackendTensorView>&       outputs,
    cudaStream_t                          stream)
{
#ifdef USE_LIBTORCH
    if (!impl_->loaded)
        throw std::runtime_error("[PyTorchBackend] Model not loaded.");

    std::optional<at::cuda::CUDAStreamGuard> guard;
    if (impl_->device_id >= 0 && cfg_.share_cuda_stream && stream) {
        at::cuda::CUDAStream ts =
            at::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream),
                                             impl_->device_id);
        guard.emplace(ts);
    }

    std::vector<torch::jit::IValue> jit_inputs;

    if (impl_->call_style == CallStyle::Dictionary) {
        c10::Dict<std::string, torch::Tensor> d;
        for (auto& tv : inputs) {
            if (tv.name[0] != '\0')
                d.insert(tv.name, wrap_tensor(tv, impl_->device_id));
        }
        jit_inputs.push_back(d);
    } else {
        for (auto& tv : inputs) {
            jit_inputs.push_back(wrap_tensor(tv, impl_->device_id));
        }
    }

    try {
        impl_->last_result = impl_->model.forward(jit_inputs);
    } catch (const c10::Error& e) {
        throw std::runtime_error("[PyTorchBackend] Forward failed: " + std::string(e.what()));
    }
    torch::jit::IValue& result = impl_->last_result;
    impl_->output_tensors.clear();  // release previous step's tensors

    const int out_device = (impl_->device_id >= 0) ? impl_->device_id : 0;
    auto ensure_cuda = [&](torch::Tensor t) -> torch::Tensor {
        if (!t.is_contiguous()) t = t.contiguous();
        if (impl_->device_id < 0) {
            t = t.to(torch::Device(torch::kCUDA, out_device));
        }
        if (t.scalar_type() == torch::kFloat32) {
            t = t.to(torch::kFloat64);
        }
        return t;
    };

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


void PyTorchBackend::synchronize() {
#ifdef USE_CUDA
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

} // namespace mlip

extern "C" mlip::IMlipBackend* mlip_create_pytorch_backend_c() {
    return new mlip::PyTorchBackend();
}
