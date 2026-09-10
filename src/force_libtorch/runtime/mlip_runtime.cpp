/**
 * mlip_runtime.cpp
 *
 * Generic MLIP runtime implementation.
 */

#include "force_libtorch/runtime/mlip_runtime.h"
#include "force_libtorch/config/yaml_parser.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef USE_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess) \
    throw std::runtime_error(std::string("[MlipRuntime] CUDA: ")+cudaGetErrorString(_e)); } while(0)
#else
#  define CUDA_CHECK(e) (void)(e)
#endif

namespace mlip {

MlipRuntime::~MlipRuntime() {
    if (backend_) backend_->synchronize();
#ifdef USE_CUDA
    for (auto& [ptr, sz] : scratch_bufs_)
        if (ptr) cudaFree(ptr);
#endif
}

// ─────────────────────── initialization ──────────────────────────────────────

void MlipRuntime::initialize(const std::string& yaml_path,
                              const MlipExecutionContext& context)
{
    std::cout << "[MlipRuntime] Loading spec: " << yaml_path << "\n";
    MlipModelSpec spec = parse_model_spec(yaml_path);
    initialize(spec, context);
}

void MlipRuntime::initialize(const MlipModelSpec& spec,
                              const MlipExecutionContext& context)
{
    spec_ = spec;
    ctx_  = context;

    // Compile execution plan (validates registries, builds LUT, etc.)
    plan_ = compile_execution_plan(spec_, ctx_);

    // Load backend
    load_backend();

    // Populate capabilities from YAML declarations
    caps_.computes_total_energy  = spec_.cap_energy;
    caps_.computes_forces        = spec_.cap_forces;
    caps_.computes_virial        = spec_.cap_virial;
    caps_.computes_stress        = spec_.cap_stress;
    caps_.supports_cuda          = true;
    caps_.supports_float32       = (spec_.precision == "float32");
    caps_.supports_float64       = (spec_.precision == "float64");
    caps_.supports_external_neighbor_list = spec_.cap_external_neighbor_list;
    caps_.supports_external_cuda_stream   = spec_.cap_external_cuda_stream;
    caps_.requires_cell          = true;
    caps_.requires_pbc           = true;

    validate_capabilities();

    initialized_ = true;
    std::cout << "[MlipRuntime] Ready. Model='" << spec_.name
              << "' backend='" << (backend_ ? backend_->name() : "?")
              << "' rank=" << ctx_.mpi_rank << " device=" << ctx_.device_id << "\n";
}

void MlipRuntime::load_backend() {
    std::string bname;
    switch (spec_.backend.backend) {
        case BackendType::PyTorch:      bname = "pytorch"; break;
        case BackendType::ONNX:         bname = "onnx"; break;
        case BackendType::TensorRT:     bname = "tensorrt"; break;
        case BackendType::NativePlugin: bname = "native_plugin"; break;
        default: throw std::runtime_error("[MlipRuntime] Unknown backend type");
    }

    backend_ = BackendRegistry::instance().create(bname);

    BackendConfiguration bcfg;
    std::strncpy(bcfg.model_path, spec_.model_path.c_str(), sizeof(bcfg.model_path) - 1);
    std::strncpy(bcfg.entrypoint, spec_.backend.entrypoint.c_str(), sizeof(bcfg.entrypoint) - 1);
    // Use device_id=-1 for CPU models, GPU id otherwise
    bcfg.device_id           = (spec_.device == "cpu") ? -1 : ctx_.device_id;
    bcfg.use_fp32            = (spec_.precision == "float32");
    bcfg.share_cuda_stream   = spec_.backend.share_cuda_stream;
    bcfg.allow_cpu_fallback  = spec_.backend.allow_cpu_fallback;
    bcfg.call_style          = spec_.backend.call_style;

    backend_->initialize(bcfg, ctx_);
}

void MlipRuntime::validate_capabilities() {
    if (!caps_.computes_forces)
        throw std::runtime_error(
            "[MlipRuntime] YAML capabilities.forces is false — model cannot compute forces.");
    if (spec_.backend.allow_cpu_fallback == false) {
        if (!caps_.supports_cuda)
            throw std::runtime_error("[MlipRuntime] Model does not support CUDA.");
    }
}

// ─────────────────────── accessors ───────────────────────────────────────────

const NeighborRequirements& MlipRuntime::neighbor_requirements() const {
    return plan_.neighbor_requirements;
}

double MlipRuntime::cutoff() const {
    return spec_.neighbor.cutoff;
}

// ─────────────────────── hot path ────────────────────────────────────────────

void MlipRuntime::compute(const MlipInputView& input, MlipOutputView& output)
{
    if (!initialized_)
        throw std::runtime_error("[MlipRuntime] Not initialized.");

    // Reset per-step scratch pointer (no allocation)
    plan_.scratch.reset_step();

    // 1. Preprocessing: build BackendTensorView list
    std::vector<BackendTensorView> tv_inputs(plan_.input_bindings.size());
    run_preprocessing(input, tv_inputs);

    // 2. Backend execute (enqueue on MD stream)
    std::vector<BackendTensorView> tv_outputs(plan_.output_bindings.size());
    // Populate tensor names so PyTorchBackend can look up dict outputs by name.
    for (size_t i = 0; i < plan_.output_bindings.size(); ++i) {
        std::strncpy(tv_outputs[i].name,
                     plan_.output_bindings[i].tensor_name.c_str(),
                     sizeof(tv_outputs[i].name) - 1);
    }
    backend_->execute(tv_inputs, tv_outputs, input.stream);

    // 3. Postprocessing: write back to MD force/energy/virial buffers
    run_postprocessing(tv_outputs, output, input);
}

void MlipRuntime::run_preprocessing(
    const MlipInputView& input,
    std::vector<BackendTensorView>& tv_inputs)
{
    OpContext op_ctx;
    op_ctx.num_local  = input.num_local;
    op_ctx.num_ghost  = input.num_ghost;
    op_ctx.num_total  = input.num_total;
    op_ctx.num_edges  = input.neighbors.num_edges;
    op_ctx.device_id  = ctx_.device_id;
    op_ctx.stream     = input.stream;

    for (size_t i = 0; i < plan_.input_bindings.size(); ++i) {
        auto& b = plan_.input_bindings[i];
        auto& tv = tv_inputs[i];

        // Get source pointer from MD
        void* src_ptr = b.pointer_resolver(op_ctx);

        // Build a DeviceArrayView for the source
        DeviceArrayView src_view;
        src_view.data      = src_ptr;
        src_view.dtype     = b.source_dtype;
        src_view.ndim      = b.source_ndim;
        src_view.device_id = ctx_.device_id;
        src_view.is_device = true;

        // Resolve source shape from target spec using current step counts.
        // This is required so preprocessing ops (e.g. cast_float32) can read
        // in.shape[] to compute the number of elements to process.
        src_view.ndim = static_cast<int>(b.target_shape.size());
        for (int d = 0; d < src_view.ndim && d < 4; ++d) {
            auto& sd = b.target_shape[d];
            switch (sd.symbol) {
                case ShapeSymbol::Nlocal:  src_view.shape[d] = input.num_local;  break;
                case ShapeSymbol::Nghost:  src_view.shape[d] = input.num_ghost;  break;
                case ShapeSymbol::Nall:    src_view.shape[d] = input.num_total;  break;
                case ShapeSymbol::Nedges:  src_view.shape[d] = input.neighbors.num_edges; break;
                case ShapeSymbol::Literal: src_view.shape[d] = sd.value;        break;
            }
        }

        // If there are preprocess ops, run them in sequence.
        // Allocate scratch for tv.data based on target shape (required by some ops like
        // full_directed_edge_list which write their output into tv.data as scratch).
        // Set tensor name unconditionally so the backend can use it as dict key.
        std::strncpy(tv.name, b.tensor_name.c_str(), sizeof(tv.name) - 1);

        if (!b.preprocess_ops.empty()) {
            // Compute scratch size from target shape using current step counts
            size_t n_elems = 1;
            for (auto& sd : b.target_shape) {
                int64_t dim = 1;
                switch (sd.symbol) {
                    case ShapeSymbol::Nlocal:  dim = input.num_local;  break;
                    case ShapeSymbol::Nghost:  dim = input.num_ghost;  break;
                    case ShapeSymbol::Nall:    dim = input.num_total;  break;
                    case ShapeSymbol::Nedges:  dim = input.neighbors.num_edges; break;  // E_full count
                    case ShapeSymbol::Literal: dim = sd.value;         break;
                }
                n_elems *= static_cast<size_t>(dim);
            }
            size_t bytes = n_elems * 8;  // worst case: int64/float64 = 8 bytes each
            // Grow per-input scratch if needed
            if (i >= scratch_bufs_.size()) scratch_bufs_.resize(i + 1, {nullptr, 0});
            if (bytes > scratch_bufs_[i].second) {
                if (scratch_bufs_[i].first) cudaFree(scratch_bufs_[i].first);
                cudaMalloc(&scratch_bufs_[i].first, bytes);
                scratch_bufs_[i].second = bytes;
            }
            tv.data = scratch_bufs_[i].first;
        }

        DeviceArrayView current = src_view;
        for (auto& op : b.preprocess_ops) {
            // For species_map: inject LUT pointer and size into tv.stride before execute.
            if (op.name == "species_map") {
                tv.stride[0] = static_cast<int64_t>(reinterpret_cast<intptr_t>(plan_.d_species_lut));
                tv.stride[1] = static_cast<int64_t>(plan_.species_lut_size);
            }
            DeviceArrayView out_view;
            op.execute(current, out_view, tv, op_ctx);
            current = out_view;
        }

        // If no ops (identity), wrap directly
        if (b.preprocess_ops.empty()) {
            tv.data      = src_ptr;
            tv.dtype     = b.source_dtype;
            tv.device_id = ctx_.device_id;
            std::strncpy(tv.name, b.tensor_name.c_str(), sizeof(tv.name) - 1);
            // Fill shape from binding spec
            tv.ndim = static_cast<int>(b.target_shape.size());
            for (int d = 0; d < tv.ndim && d < 4; ++d) {
                auto& sd = b.target_shape[d];
                switch (sd.symbol) {
                    case ShapeSymbol::Nlocal: tv.shape[d] = input.num_local; break;
                    case ShapeSymbol::Nghost: tv.shape[d] = input.num_ghost; break;
                    case ShapeSymbol::Nall:   tv.shape[d] = input.num_total; break;
                    case ShapeSymbol::Nedges: tv.shape[d] = input.neighbors.num_edges; break;
                    case ShapeSymbol::Literal: tv.shape[d] = sd.value; break;
                }
            }
        }
        tv.requires_grad = b.requires_grad;
    }
}

void MlipRuntime::run_postprocessing(
    std::vector<BackendTensorView>& tv_outputs,
    MlipOutputView& output,
    const MlipInputView& input)
{
    OpContext op_ctx;
    op_ctx.num_local  = input.num_local;
    op_ctx.num_ghost  = input.num_ghost;
    op_ctx.num_total  = input.num_total;
    op_ctx.num_edges  = input.neighbors.num_edges;
    op_ctx.device_id  = ctx_.device_id;
    op_ctx.stream     = input.stream;

    for (size_t i = 0; i < plan_.output_bindings.size(); ++i) {
        if (i >= tv_outputs.size()) break;
        auto& b  = plan_.output_bindings[i];
        auto& tv = tv_outputs[i];

        // Get MD target pointer
        void* tgt = b.target_resolver(op_ctx);

        DeviceArrayView target_view;
        target_view.data      = tgt;
        target_view.device_id = ctx_.device_id;
        target_view.is_device = true;
        // Store scale*sign in stride[2] for copy_forces / accumulate_force ops
        {
            double sf = b.scale_factor * b.force_sign;
            std::memcpy(&target_view.stride[2], &sf, sizeof(double));
        }

        for (auto& op : b.postprocess_ops) {
            op.execute(tv, target_view, op_ctx);
        }

        // If no postprocess ops, the backend is expected to write directly
        // into the target (zero-copy path set up at initialization).
    }
}

void MlipRuntime::synchronize() {
    if (backend_) backend_->synchronize();
}

} // namespace mlip
