/**
 * execution_plan.cpp
 *
 * Compiles a MlipModelSpec into a hot-path-friendly MlipExecutionPlan.
 */

#include "mlip/runtime/execution_plan.h"
#include "mlip/operations/operation_registry.h"

#ifdef USE_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess) \
    throw std::runtime_error(std::string("[MLIP] CUDA: ")+cudaGetErrorString(_e)); } while(0)
#else
#  define CUDA_CHECK(e) (void)(e)
#endif

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace mlip {

// ─────────────────────── unit conversion ─────────────────────────────────────

static double length_factor(const std::string& s) {
    if (s == "angstrom") return 1.0;
    if (s == "nm")       return 10.0;   // 1 nm = 10 Å
    if (s == "bohr")     return 0.529177;
    if (s == "m")        return 1e10;
    throw std::runtime_error("[MLIP] Unknown length unit '" + s + "'");
}

static double energy_factor(const std::string& s) {
    if (s == "eV")         return 1.0;
    if (s == "kcal/mol")   return 0.04336410;   // kcal/mol per eV
    if (s == "kJ/mol" || s == "kJ_per_mol") return 0.01036410;
    if (s == "hartree")    return 27.2114;
    if (s == "kcal_per_mol") return 0.04336410;
    throw std::runtime_error("[MLIP] Unknown energy unit '" + s + "'");
}

UnitConversion compute_unit_conversion(const UnitSystem& md, const UnitSystem& model) {
    UnitConversion uc;
    double l_md    = length_factor(md.length);
    double l_model = length_factor(model.length);
    double e_md    = energy_factor(md.energy);
    double e_model = energy_factor(model.energy);

    // How to scale model output to MD units:
    uc.position_scale = l_md / l_model;      // rarely != 1 since we pass positions in MD units
    uc.energy_scale   = e_model / e_md;      // model_energy -> md_energy
    uc.force_scale    = (e_model / l_model) / (e_md / l_md);
    uc.virial_scale   = uc.energy_scale;
    uc.stress_scale   = uc.energy_scale / (l_md * l_md * l_md / (l_model * l_model * l_model));
    return uc;
}

// ─────────────────────── scratch allocator ───────────────────────────────────

void MlipScratchAllocator::reserve(size_t bytes) {
    if (bytes <= capacity_) return;
#ifdef USE_CUDA
    if (d_buf_) CUDA_CHECK(cudaFree(d_buf_));
    CUDA_CHECK(cudaMalloc(&d_buf_, bytes));
#else
    free(d_buf_);
    d_buf_ = malloc(bytes);
    if (!d_buf_) throw std::bad_alloc();
#endif
    capacity_ = bytes;
    offset_   = 0;
}

void* MlipScratchAllocator::allocate_view(size_t bytes, size_t alignment) {
    size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes > capacity_)
        throw std::runtime_error("[MlipScratchAllocator] Out of scratch memory");
    void* ptr = static_cast<char*>(d_buf_) + aligned;
    offset_   = aligned + bytes;
    return ptr;
}

void MlipScratchAllocator::reset_step() {
    offset_ = 0;   // no free — just reset pointer
}

MlipScratchAllocator::~MlipScratchAllocator() {
#ifdef USE_CUDA
    if (d_buf_) { cudaFree(d_buf_); d_buf_ = nullptr; }
#else
    free(d_buf_); d_buf_ = nullptr;
#endif
}

// ─────────────────────── helper: dtype element size ──────────────────────────

static size_t dtype_bytes(ScalarType d) {
    switch (d) {
        case ScalarType::Float32: return 4;
        case ScalarType::Float64: return 8;
        case ScalarType::Int32:   return 4;
        case ScalarType::Int64:   return 8;
        case ScalarType::Bool:    return 1;
        default: return 8;
    }
}

static size_t shape_elems(const ShapeSpec& shape, const OpContext& ctx) {
    size_t n = 1;
    for (auto& d : shape) {
        switch (d.symbol) {
            case ShapeSymbol::Nlocal: n *= ctx.num_local; break;
            case ShapeSymbol::Nghost: n *= ctx.num_ghost; break;
            case ShapeSymbol::Nall:   n *= ctx.num_total; break;
            case ShapeSymbol::Nedges: n *= ctx.num_edges; break;
            case ShapeSymbol::Literal: n *= d.value; break;
        }
    }
    return n;
}

// ─────────────────────── species lookup table ────────────────────────────────

static void build_species_lut(
    const MlipModelSpec& spec,
    MlipExecutionPlan&   plan,
    const MlipExecutionContext& ctx)
{
    const auto& sm = spec.species_map;
    // Determine max key to size the table
    int max_key = 0;
    if (sm.input_type == SpeciesInputType::AtomicSymbol) {
        for (auto& [sym, idx] : sm.symbol_to_index)
            max_key = std::max(max_key, idx);
    } else {
        for (auto& [k, _] : sm.int_to_index)
            max_key = std::max(max_key, k);
    }

    int lut_size = max_key + 1;
    plan.species_lut_size = lut_size;

    std::vector<int32_t> h_lut(lut_size, -1);
    if (sm.input_type == SpeciesInputType::AtomicSymbol) {
        // Symbol-based: caller must have provided a way to map symbol→int externally.
        // For now fill with symbol_to_index directly (index = model index).
        for (auto& [sym, idx] : sm.symbol_to_index) {
            (void)sym;
            if (idx < lut_size) h_lut[idx] = idx;
        }
    } else {
        for (auto& [k, v] : sm.int_to_index) {
            if (k < lut_size) h_lut[k] = v;
        }
    }

#ifdef USE_CUDA
    int32_t* d_lut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lut, sizeof(int32_t) * lut_size));
    CUDA_CHECK(cudaMemcpy(d_lut, h_lut.data(), sizeof(int32_t) * lut_size,
                          cudaMemcpyHostToDevice));
    plan.d_species_lut = d_lut;
#endif

    // Also fill the SpeciesMapping struct used by the IMlipPotential path
    plan.species_mapping.md_type_to_atomic_number.resize(lut_size);
    for (int i = 0; i < lut_size; ++i)
        plan.species_mapping.md_type_to_atomic_number[i] = h_lut[i];
}

// ─────────────────────── validate registries ─────────────────────────────────

static void validate_registries(const MlipModelSpec& spec) {
    // source registry
    for (auto& [name, inp] : spec.inputs) {
        if (!SourceRegistry::instance().has(inp.source))
            throw std::runtime_error(
                "[MLIP] YAML input '" + name + "' references unknown source '" +
                inp.source + "'. Registered sources: " +
                SourceRegistry::instance().list());
        // transforms
        for (auto& t : inp.transforms) {
            if (!PreprocessRegistry::instance().has(t))
                throw std::runtime_error(
                    "[MLIP] YAML input '" + name + "' uses unknown transform '" +
                    t + "'. Registered: " + PreprocessRegistry::instance().list());
        }
    }
    // target registry
    for (auto& [name, out] : spec.outputs) {
        if (!TargetRegistry::instance().has(out.target))
            throw std::runtime_error(
                "[MLIP] YAML output '" + name + "' references unknown target '" +
                out.target + "'");
        for (auto& op : out.operations) {
            if (!PostprocessRegistry::instance().has(op))
                throw std::runtime_error(
                    "[MLIP] YAML output '" + name + "' uses unknown operation '" + op + "'");
        }
    }
    // backend
    std::string backend_name;
    switch (spec.backend.backend) {
        case BackendType::PyTorch:      backend_name = "pytorch"; break;
        case BackendType::ONNX:         backend_name = "onnx"; break;
        case BackendType::TensorRT:     backend_name = "tensorrt"; break;
        case BackendType::NativePlugin: backend_name = "native_plugin"; break;
        default: throw std::runtime_error("[MLIP] Unknown backend type");
    }
    if (!BackendRegistry::instance().has(backend_name))
        throw std::runtime_error(
            "[MLIP] Backend '" + backend_name + "' is not registered. "
            "Available: " + BackendRegistry::instance().list());
}

// ─────────────────────── main compile function ───────────────────────────────

MlipExecutionPlan compile_execution_plan(
    const MlipModelSpec&       spec,
    const MlipExecutionContext& ctx)
{
    MlipExecutionPlan plan;

    // 1. Validate registries first (gives clear errors before any allocation)
    validate_registries(spec);

    // 2. Unit conversion
    plan.unit_conversion = compute_unit_conversion(spec.md_units, spec.model_units);

    // 3. Neighbor requirements
    plan.neighbor_requirements.cutoff               = spec.neighbor.cutoff;
    plan.neighbor_requirements.requires_full_list   = (spec.neighbor.list == NeighborListKind::Full);
    plan.neighbor_requirements.requires_directed_edges = spec.neighbor.directed;
    plan.neighbor_requirements.requires_self_edges   = spec.neighbor.self_edges;
    plan.neighbor_requirements.requires_edge_vectors = spec.neighbor.require_edge_vecs;
    plan.neighbor_requirements.requires_distances    = spec.neighbor.require_distances;
    plan.neighbor_requirements.requires_cell_shifts  = spec.neighbor.require_cell_shifts;
    plan.neighbor_requirements.supports_ghost_atoms  = spec.neighbor.include_ghost;

    // 4. Species lookup table
    build_species_lut(spec, plan, ctx);

    // 5. Autograd flag
    plan.requires_autograd = spec.autograd.enabled;
    plan.share_cuda_stream = spec.backend.share_cuda_stream;

    // 6. Estimate scratch size (conservative upper bound at plan compile time)
    // We use a fixed capacity; the scratch grows at step time if needed.
    size_t scratch_estimate = 0;

    // 7. Resolve input bindings
    plan.input_bindings.reserve(spec.inputs.size());
    for (auto& [iname, ispec] : spec.inputs) {
        ResolvedInputBinding b;
        b.tensor_name      = ispec.tensor_name;
        b.backend_index    = ispec.argument_index;
        b.target_dtype     = ispec.dtype;
        b.target_shape     = ispec.shape;
        b.requires_grad    = ispec.requires_grad;
        b.pointer_resolver = SourceRegistry::instance().get(ispec.source).pointer_resolver;
        b.source_dtype     = SourceRegistry::instance().get(ispec.source).dtype;
        b.source_ndim      = SourceRegistry::instance().get(ispec.source).rank;

        // Compile preprocess op chain
        for (auto& tname : ispec.transforms) {
            b.preprocess_ops.push_back(PreprocessRegistry::instance().get(tname));
        }

        // Estimate scratch for this input (rough: shape * dtype_bytes * 2)
        // Actual allocation happens on first step via scratch.reserve().
        OpContext est_ctx{ 1000, 200, 1200, 10000, ctx.device_id };
        if (!b.target_shape.empty()) {
            size_t elems = shape_elems(b.target_shape, est_ctx);
            scratch_estimate += elems * dtype_bytes(b.target_dtype) * 2;
        }

        plan.input_bindings.push_back(std::move(b));
    }

    // Sort by argument_index for positional style
    if (spec.backend.call_style == CallStyle::Positional) {
        std::sort(plan.input_bindings.begin(), plan.input_bindings.end(),
            [](const ResolvedInputBinding& a, const ResolvedInputBinding& b) {
                return a.backend_index < b.backend_index;
            });
    }

    // 8. Resolve output bindings
    plan.output_bindings.reserve(spec.outputs.size());
    for (auto& [oname, ospec] : spec.outputs) {
        ResolvedOutputBinding b;
        b.tensor_name      = ospec.tensor_name;
        b.backend_index    = ospec.output_index;
        b.expected_dtype   = ospec.expected_dtype;
        b.expected_shape   = ospec.expected_shape;
        b.force_sign       = ospec.sign;
        b.scale_factor     = ospec.scale;
        b.includes_ghost   = ospec.includes_ghost;
        b.force_convention  = ospec.force_convention;
        b.virial_convention = ospec.virial_convention;
        b.target_resolver  = TargetRegistry::instance().get(ospec.target);

        for (auto& op : ospec.operations)
            b.postprocess_ops.push_back(PostprocessRegistry::instance().get(op));

        plan.output_bindings.push_back(std::move(b));
    }

    // 9. Reserve scratch
    plan.scratch.reserve(scratch_estimate > 0 ? scratch_estimate : 64 * 1024 * 1024);

    std::cout << "[MlipRuntime] Execution plan compiled for model '" << spec.name
              << "' — " << plan.input_bindings.size() << " inputs, "
              << plan.output_bindings.size() << " outputs, "
              << "cutoff=" << spec.neighbor.cutoff << " Å\n";

    return plan;
}

} // namespace mlip
