/**
 * potential_interface.h
 *
 * General IPotential interface used by the MD core's force accumulation loop.
 * Supports composing multiple potentials (classical + MLIP + long-range + bias).
 */

#ifndef POTENTIAL_INTERFACE_H
#define POTENTIAL_INTERFACE_H

#include "../md_types.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

// ---------------------------------------------------------------------------
// Lightweight view of the simulation state passed to each potential.
// All GPU pointers are owned by ForceCalculatorCUDA.
// ---------------------------------------------------------------------------
struct SimulationStateView {
    // GPU arrays
    const double* d_x         = nullptr;  // [num_total]
    const double* d_y         = nullptr;
    const double* d_z         = nullptr;
    const int*    d_type      = nullptr;  // [num_total] MD type id
    const int*    d_global_id = nullptr;  // [num_total]

    // Neighbor list (canonical COO on GPU)
    const int32_t* d_edge_src = nullptr;  // [num_edges]
    const int32_t* d_edge_dst = nullptr;  // [num_edges]
    int64_t        num_edges  = 0;

    int    num_local  = 0;
    int    num_ghost  = 0;
    int    num_total  = 0;

    // Cell / PBC (on host — rarely changes)
    double cell[3][3] = {};
    bool   pbc[3]     = {true, true, true};

    cudaStream_t stream    = nullptr;
    int          device_id = 0;
};

// ---------------------------------------------------------------------------
// Force accumulator: potentials ADD their forces into these device buffers.
// ---------------------------------------------------------------------------
struct ForceAccumulator {
    double* d_fx = nullptr;  // [num_total] — pre-zeroed by MD core each step
    double* d_fy = nullptr;
    double* d_fz = nullptr;
    double* d_pe = nullptr;  // total potential energy (device scalar, atomically accumulated)
    double* d_virial = nullptr;  // [9] virial tensor (device, accumulated)
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Compute context passed alongside the state view.
// ---------------------------------------------------------------------------
struct ComputeContext {
    int mpi_rank  = 0;
    int mpi_size  = 1;
    int device_id = 0;
    cudaStream_t stream = nullptr;
    bool need_virial = false;
};

// ---------------------------------------------------------------------------
// Base interface for all force contributions.
// ---------------------------------------------------------------------------
class IPotential {
public:
    virtual ~IPotential() = default;

    virtual void initialize() = 0;
    virtual void finalize()   = 0;

    virtual void compute(
        const SimulationStateView& state,
        ForceAccumulator&          accumulator,
        const ComputeContext&      context
    ) = 0;

    virtual std::string name() const = 0;
};

#endif // POTENTIAL_INTERFACE_H
