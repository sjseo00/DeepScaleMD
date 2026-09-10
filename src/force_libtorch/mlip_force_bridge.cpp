/**
 * mlip_force_bridge.cpp
 *
 * C++ bridge between ForceCalculatorCUDA (compiled as CUDA TU) and the
 * full MlipRuntime / md_data_sources API (compiled as a regular C++ TU).
 *
 * Isolates operation_registry / execution_plan Meyers singletons from
 * force_calculator_cuda.cu, preventing heap corruption from duplicate
 * static-local variable initialisation across CUDA TUs.
 */

#include "force_calculator_cuda.h"
#include "force_libtorch/mlip_types.h"
#include "force_libtorch/runtime/mlip_runtime.h"
#include "force_libtorch/binding/md_data_sources.h"

void mlip_bridge_register_md_sources(void* fc_cuda_ptr)
{
    ForceCalculatorCUDA* fc = static_cast<ForceCalculatorCUDA*>(fc_cuda_ptr);

    mlip::MdDevicePointers ptrs;
    ptrs.get_x         = [fc]() -> void* { return fc->d_x; };
    ptrs.get_y         = [fc]() -> void* { return fc->d_y; };
    ptrs.get_z         = [fc]() -> void* { return fc->d_z; };
    ptrs.get_fx        = [fc]() -> void* { return fc->d_fx; };
    ptrs.get_fy        = [fc]() -> void* { return fc->d_fy; };
    ptrs.get_fz        = [fc]() -> void* { return fc->d_fz; };
    ptrs.get_type      = [fc]() -> void* { return fc->d_type; };
    ptrs.get_global_id = [fc]() -> void* { return fc->d_global_id; };
    ptrs.get_edge_src  = [fc]() -> void* { return fc->d_edge_src_; };
    ptrs.get_edge_dst  = [fc]() -> void* { return fc->d_edge_dst_; };
    ptrs.get_edge_coo_packed = [fc]() -> void* { return fc->d_edge_coo_packed_; };
    ptrs.get_num_edges = [fc]() -> int64_t { return fc->num_edges_; };
    ptrs.get_num_local = [fc]() -> int { return fc->num_local_cached_; };
    ptrs.get_num_ghost = [fc]() -> int { return fc->num_ghost_cached_; };
    ptrs.get_num_total = [fc]() -> int { return fc->num_local_cached_ + fc->num_ghost_cached_; };
    ptrs.get_pe        = [fc]() -> void* { return fc->d_potential_energy; };
    ptrs.get_virial    = [fc]() -> void* { return fc->d_virial; };
    ptrs.get_pos_interleaved = [fc]() -> void* { return fc->d_pos_interleaved_; };
    ptrs.get_mlip_forces = [fc]() -> void* { return fc->d_mlip_forces_; };
    ptrs.get_cell        = [fc]() -> void* { return fc->d_cell_; };
    ptrs.get_edge_vectors = [fc]() -> void* { return fc->d_edge_vectors_; };
    ptrs.get_batch       = [fc]() -> void* { return fc->d_batch_; };
    ptrs.get_num_atoms   = [fc]() -> void* { return fc->d_num_atoms_; };

    mlip::register_md_sources(ptrs);
}

void mlip_bridge_compute(void* runtime_ptr,
                         const mlip::MlipInputView* in_view,
                         mlip::MlipOutputView*      out_view)
{
    if (!runtime_ptr || !in_view || !out_view) return;
    mlip::MlipRuntime* rt = static_cast<mlip::MlipRuntime*>(runtime_ptr);
    rt->compute(*in_view, *out_view);
}
