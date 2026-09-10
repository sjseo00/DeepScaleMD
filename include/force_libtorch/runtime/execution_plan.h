/**
 * execution_plan.h
 *
 * Hot-path-friendly execution plan compiled once from MlipModelSpec.
 * Contains no strings in the hot path — all names are resolved to
 * indices, function pointers and cached tensors at initialization time.
 */

#pragma once
#ifndef MLIP_EXECUTION_PLAN_H
#define MLIP_EXECUTION_PLAN_H

#include "../mlip_types.h"
#include "../runtime/model_specification.h"
#include "../backend/backend_interface.h"
#include "../operations/operation_registry.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlip {

// ─────────────── unit conversion factors (computed once) ─────────────────────

struct UnitConversion {
    double position_scale = 1.0;  // md_length / model_length
    double energy_scale   = 1.0;  // md_energy / model_energy
    double force_scale    = 1.0;  // md_energy/md_length / model_energy/model_length
    double virial_scale   = 1.0;  // same as energy for virial
    double stress_scale   = 1.0;  // pressure: md_energy/md_volume / model
};

UnitConversion compute_unit_conversion(const UnitSystem& md, const UnitSystem& model);

// ─────────────── resolved input binding ──────────────────────────────────────

struct ResolvedInputBinding {
    int         backend_index = -1;   // positional or dict slot
    std::string tensor_name;          // debug / dict key

    // Source: cached pointer resolver (no string lookup at step time)
    PointerResolver  pointer_resolver;
    mlip::ScalarType source_dtype   = mlip::ScalarType::Float64;
    int        source_ndim    = 0;

    // After preprocessing, what dtype/shape is sent to the backend?
    mlip::ScalarType target_dtype   = mlip::ScalarType::Float32;
    ShapeSpec  target_shape;
    bool       requires_grad  = false;

    // Compiled preprocessing sequence (empty = identity / direct wrap)
    std::vector<PreprocessOperation> preprocess_ops;

    // Persistent scratch device buffer for this input (if conversion needed).
    // nullptr means we can wrap the source directly (zero-copy).
    void*      d_scratch      = nullptr;
    size_t     scratch_bytes  = 0;
};

// ─────────────── resolved output binding ─────────────────────────────────────

struct ResolvedOutputBinding {
    int         backend_index = -1;
    std::string tensor_name;

    PointerResolver target_resolver;
    mlip::ScalarType expected_dtype  = mlip::ScalarType::Float32;
    ShapeSpec  expected_shape;

    double     force_sign      = 1.0;
    double     scale_factor    = 1.0;
    bool       includes_ghost  = false;

    ForceConvention  force_convention  = ForceConvention::Force;
    VirialConvention virial_convention = VirialConvention::VirialTensor;

    std::vector<PostprocessOperation> postprocess_ops;

    // Scratch buffer for model output before postprocess
    void*  d_output_scratch   = nullptr;
    size_t output_scratch_bytes = 0;
};

// ─────────────── scratch allocator ───────────────────────────────────────────

class MlipScratchAllocator {
public:
    void reserve(size_t bytes);
    void* allocate_view(size_t bytes, size_t alignment = 256);
    void reset_step();   // only resets offset — no free
    ~MlipScratchAllocator();

private:
    void*  d_buf_     = nullptr;
    size_t capacity_  = 0;
    size_t offset_    = 0;
};

// ─────────────── execution plan ──────────────────────────────────────────────

struct MlipExecutionPlan {
    // Compiled from YAML at init time; never changes during simulation.
    std::vector<ResolvedInputBinding>  input_bindings;
    std::vector<ResolvedOutputBinding> output_bindings;

    NeighborRequirements neighbor_requirements;
    SpeciesMapping       species_mapping;
    UnitConversion       unit_conversion;

    bool requires_autograd  = false;
    bool share_cuda_stream  = true;

    // GPU lookup table for species mapping: [capacity] int32, index = MD type
    int32_t* d_species_lut = nullptr;
    int      species_lut_size = 0;

    MlipScratchAllocator scratch;
};

// ─────────────── compilation function ────────────────────────────────────────

/**
 * Compile a MlipModelSpec into a MlipExecutionPlan.
 * Resolves all string names to indices/function pointers.
 * Allocates persistent GPU scratch buffers.
 * Validates all registry entries and dtype/shape compatibility.
 * Throws std::runtime_error with a descriptive message on any error.
 */
MlipExecutionPlan compile_execution_plan(
    const MlipModelSpec&       spec,
    const MlipExecutionContext& ctx
);

} // namespace mlip
#endif // MLIP_EXECUTION_PLAN_H
