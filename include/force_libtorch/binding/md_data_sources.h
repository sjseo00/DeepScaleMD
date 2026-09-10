/**
 * md_data_sources.h / .cpp
 *
 * Registers the canonical MD data sources (e.g. "md.positions") with SourceRegistry,
 * and the output targets (e.g. "md.forces") with TargetRegistry.
 *
 * Called once from ForceCalculatorCUDA::initialize_mlip_sources() which passes
 * live device pointers captured by lambda closures.
 *
 * The closures are invalidated when ForceCalculatorCUDA reallocates device memory;
 * re-register is not needed because the resolvers always call the pointer-getter
 * function rather than capturing a raw pointer at registration time.
 */

#pragma once
#ifndef MLIP_MD_DATA_SOURCES_H
#define MLIP_MD_DATA_SOURCES_H

#include "force_libtorch/operations/operation_registry.h"
#include <functional>

namespace mlip {

/**
 * Struct holding all the "pointer-getter" functions for MD device arrays.
 * ForceCalculatorCUDA fills this and passes it to register_md_sources().
 * Using getters (not raw pointers) means the registry is valid even after
 * device memory reallocation — getters always return the current pointer.
 */
struct MdDevicePointers {
    std::function<void*()> get_x;
    std::function<void*()> get_y;
    std::function<void*()> get_z;
    std::function<void*()> get_fx;
    std::function<void*()> get_fy;
    std::function<void*()> get_fz;
    std::function<void*()> get_type;
    std::function<void*()> get_global_id;

    // Neighbor list (edge COO) — built by build_coo_edge_list_on_device
    std::function<void*()>    get_edge_src;  // int32 [num_edges] full list src
    std::function<void*()>    get_edge_dst;  // int32 [num_edges] full list dst
    // Packed half-list [src (E_half) | dst (E_half)] for full_directed_edge_list op
    std::function<void*()>    get_edge_coo_packed;
    std::function<int64_t()>  get_num_edges;

    // Counts
    std::function<int()> get_num_local;
    std::function<int()> get_num_ghost;
    std::function<int()> get_num_total;

    // Energy / virial device scalars
    std::function<void*()> get_pe;
    std::function<void*()> get_virial;

    // Interleaved AoS positions [N, 3] float64 — built by interleave_positions kernel
    std::function<void*()> get_pos_interleaved;
    // AoS force output buffer [N, 3] float64 — read by launch_accumulate_forces_scaled
    std::function<void*()> get_mlip_forces;
    // Orthorhombic cell matrix [9] float64: [lx,0,0, 0,ly,0, 0,0,lz]
    // Updated each step before MLIP call by compute_mlip_runtime_forces.
    std::function<void*()> get_cell;  // may be nullptr if not yet allocated

    // Minimum-image edge vectors [E_full, 3] float32, built each step.
    // Used by NequIP and other models that expect precomputed edge_vectors.
    std::function<void*()> get_edge_vectors;  // may be nullptr if not built

    // Batch indices [N] int64 (all zeros, for single-system batched models).
    std::function<void*()> get_batch;      // may be nullptr if not allocated

    // Number of atoms [1] int64 — scalar tensor holding N.
    std::function<void*()> get_num_atoms;  // may be nullptr if not allocated
};

/**
 * Register all MD data sources and targets.
 * Safe to call multiple times (skips already-registered names).
 */
void register_md_sources(const MdDevicePointers& ptrs);

} // namespace mlip
#endif // MLIP_MD_DATA_SOURCES_H
