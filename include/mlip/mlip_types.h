/**
 * mlip_types.h
 *
 * Common data types shared between the MD core and all MLIP adapters/backends.
 * MD core must NOT include any MLIP framework headers (torch, onnx, etc.).
 * All MLIP-framework-specific code lives only in adapters/ and backends/.
 */

#ifndef MLIP_TYPES_H
#define MLIP_TYPES_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
// Stub so headers compile without CUDA toolkit
using cudaStream_t = void*;
#endif

namespace mlip {

// ---------------------------------------------------------------------------
// Scalar dtype tag  (lives in namespace mlip — no conflict with c10::ScalarType)
// ---------------------------------------------------------------------------
enum class ScalarType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool
};

// ---------------------------------------------------------------------------
// Non-owning view of a contiguous device (or host) array.
// Ownership belongs to the MD core buffers; adapters must never free `data`.
// ---------------------------------------------------------------------------
struct DeviceArrayView {
    void*      data    = nullptr;
    ScalarType dtype   = ScalarType::Float64;
    int        ndim    = 0;
    std::array<int64_t, 4> shape  = {0, 0, 0, 0};
    std::array<int64_t, 4> stride = {0, 0, 0, 0};
    int        device_id = 0;   // CUDA device id (-1 = host)
    bool       is_device = true;
};

// ---------------------------------------------------------------------------
// Non-owning view of a GPU-resident neighbor/edge list.
// Canonical format: COO edge list (src[e], dst[e]).
// MD core keeps one canonical copy; adapters convert to model-specific format.
// ---------------------------------------------------------------------------
struct DeviceNeighborListView {
    int64_t num_edges = 0;

    const int32_t* src          = nullptr;  // [num_edges]
    const int32_t* dst          = nullptr;  // [num_edges]

    const int32_t* cell_shift   = nullptr;  // [num_edges][3], optional
    const double*  edge_vector  = nullptr;  // [num_edges][3], optional (Å)
    const double*  distance     = nullptr;  // [num_edges],    optional (Å)

    bool is_full_list       = true;   // each edge appears in both directions
    bool includes_ghost     = true;
    bool directed           = true;

    int device_id = 0;
};

// ---------------------------------------------------------------------------
// Input view passed from MD core to MLIP adapter each step.
// All pointers point into GPU-resident buffers owned by ForceCalculatorCUDA.
// ---------------------------------------------------------------------------
struct MlipInputView {
    int num_local  = 0;
    int num_ghost  = 0;
    int num_total  = 0;   // num_local + num_ghost

    DeviceArrayView positions;   // [num_total, 3] float64 on device
    DeviceArrayView species;     // [num_total]    int32   on device (MD type id)

    DeviceNeighborListView neighbors;

    DeviceArrayView cell;        // [3, 3] float64 on device
    DeviceArrayView pbc;         // [3]    int32   on device (1/0)

    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Output view: force buffer owned by MD core; MLIP writes directly into it.
// ---------------------------------------------------------------------------
struct MlipOutputView {
    DeviceArrayView forces;       // [num_total, 3] float64 on device
    DeviceArrayView per_atom_energy;  // [num_total]  float64 on device, optional
    DeviceArrayView virial;       // [9] float64 on device, optional (row-major)

    double* total_energy = nullptr;   // device scalar
    bool    has_virial   = false;

    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Execution context: one instance per MPI rank.
// ---------------------------------------------------------------------------
struct MlipExecutionContext {
    int mpi_rank = 0;
    int mpi_size = 1;

    int          device_id = 0;
    cudaStream_t stream    = nullptr;

    void* communicator      = nullptr;  // opaque MPI_Comm, cast as needed
    void* memory_resource   = nullptr;  // custom allocator, optional
};

// ---------------------------------------------------------------------------
// Model configuration loaded from yaml/config.
// ---------------------------------------------------------------------------
struct MlipModelConfig {
    std::string adapter_name;    // "mace", "nequip", "deepmd", "onnx", ...
    std::string backend_name;    // "libtorch", "onnx_cuda", "tensorrt", ...
    std::string model_path;

    std::string precision;       // "float32" or "float64"
    bool        external_neighbor_list = true;
    bool        share_cuda_stream      = true;
    bool        allow_cpu_fallback     = false;

    // MD type id -> atomic number
    std::vector<int> md_type_to_atomic_number;

    // Unit system of the model
    std::string length_unit;  // "angstrom", "bohr", ...
    std::string energy_unit;  // "eV", "kcal/mol", "hartree", ...

    double neighbor_skin         = 0.5;
    int    rebuild_check_interval = 10;
};

// ---------------------------------------------------------------------------
// Capabilities advertised by a potential adapter at initialize() time.
// ---------------------------------------------------------------------------
struct MlipCapabilities {
    bool computes_total_energy   = false;
    bool computes_atomic_energy  = false;
    bool computes_forces         = false;
    bool computes_virial         = false;
    bool computes_stress         = false;

    bool force_from_autograd     = false;
    bool supports_float32        = false;
    bool supports_float64        = false;

    bool supports_cuda           = false;
    bool supports_cpu            = false;

    bool supports_external_neighbor_list = false;
    bool supports_external_cuda_stream   = false;

    bool requires_cell           = true;
    bool requires_pbc            = true;
};

// ---------------------------------------------------------------------------
// Neighbor list requirements declared by the adapter so the MD core can
// build the correct list format before calling compute().
// ---------------------------------------------------------------------------
struct NeighborRequirements {
    double cutoff = 6.0;

    bool requires_full_list      = true;
    bool requires_directed_edges = true;
    bool requires_self_edges     = false;

    bool requires_edge_vectors   = false;
    bool requires_distances      = false;
    bool requires_cell_shifts    = false;

    bool supports_ghost_atoms    = true;
};

// ---------------------------------------------------------------------------
// Species mapping helper: MD type id -> atomic number -> model species index.
// ---------------------------------------------------------------------------
struct SpeciesMapping {
    // md_type_to_atomic_number[md_type] = Z
    std::vector<int> md_type_to_atomic_number;

    // atomic_number_to_model_index[Z] = model_index
    // Using a flat array indexed by Z (up to 118); -1 = unsupported.
    std::array<int, 119> atomic_number_to_model_index;

    SpeciesMapping() {
        atomic_number_to_model_index.fill(-1);
    }

    /// Build from md_type_to_atomic_number and an ordered list of atomic numbers
    /// that the model supports (model species index = position in that list).
    void build(const std::vector<int>& md_to_Z,
               const std::vector<int>& model_Z_order) {
        md_type_to_atomic_number = md_to_Z;
        atomic_number_to_model_index.fill(-1);
        for (int idx = 0; idx < static_cast<int>(model_Z_order.size()); ++idx) {
            int Z = model_Z_order[idx];
            if (Z >= 0 && Z <= 118)
                atomic_number_to_model_index[Z] = idx;
        }
    }

    /// Map an MD type id directly to a model species index.
    /// Returns -1 if the mapping is missing.
    int md_type_to_model_index(int md_type) const {
        if (md_type < 0 || md_type >= static_cast<int>(md_type_to_atomic_number.size()))
            return -1;
        int Z = md_type_to_atomic_number[md_type];
        if (Z < 0 || Z > 118)
            return -1;
        return atomic_number_to_model_index[Z];
    }
};


} // namespace mlip

// ---------------------------------------------------------------------------
// Backward-compatibility: pull mlip types into global namespace ONLY when
// LibTorch headers are NOT included.  When USE_LIBTORCH is defined, callers
// must qualify with mlip:: (or add "using namespace mlip;" in their .cpp).
// This avoids the ::ScalarType vs c10::ScalarType ambiguity.
// ---------------------------------------------------------------------------
#ifndef USE_LIBTORCH
using mlip::ScalarType;
using mlip::DeviceArrayView;
using mlip::DeviceNeighborListView;
using mlip::MlipInputView;
using mlip::MlipOutputView;
using mlip::MlipExecutionContext;
using mlip::MlipModelConfig;
using mlip::MlipCapabilities;
using mlip::NeighborRequirements;
using mlip::SpeciesMapping;
#endif

#endif // MLIP_TYPES_H
