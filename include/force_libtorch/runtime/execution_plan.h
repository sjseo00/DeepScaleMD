
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


struct UnitConversion {
    double position_scale = 1.0;  // md_length / model_length
    double energy_scale   = 1.0;  // md_energy / model_energy
    double force_scale    = 1.0;  // md_energy/md_length / model_energy/model_length
    double virial_scale   = 1.0;  // same as energy for virial
    double stress_scale   = 1.0;  // pressure: md_energy/md_volume / model
};

UnitConversion compute_unit_conversion(const UnitSystem& md, const UnitSystem& model);


struct ResolvedInputBinding {
    int         backend_index = -1;   // positional or dict slot
    std::string tensor_name;          // debug / dict key

    PointerResolver  pointer_resolver;
    mlip::ScalarType source_dtype   = mlip::ScalarType::Float64;
    int        source_ndim    = 0;

    mlip::ScalarType target_dtype   = mlip::ScalarType::Float32;
    ShapeSpec  target_shape;
    bool       requires_grad  = false;

    std::vector<PreprocessOperation> preprocess_ops;

    void*      d_scratch      = nullptr;
    size_t     scratch_bytes  = 0;
};


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

    void*  d_output_scratch   = nullptr;
    size_t output_scratch_bytes = 0;
};


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


struct MlipExecutionPlan {
    std::vector<ResolvedInputBinding>  input_bindings;
    std::vector<ResolvedOutputBinding> output_bindings;

    NeighborRequirements neighbor_requirements;
    SpeciesMapping       species_mapping;
    UnitConversion       unit_conversion;

    bool requires_autograd  = false;
    bool share_cuda_stream  = true;

    int32_t* d_species_lut = nullptr;
    int      species_lut_size = 0;

    MlipScratchAllocator scratch;
};


MlipExecutionPlan compile_execution_plan(
    const MlipModelSpec&       spec,
    const MlipExecutionContext& ctx
);

} // namespace mlip
#endif // MLIP_EXECUTION_PLAN_H
