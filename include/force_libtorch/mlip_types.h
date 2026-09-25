
#ifndef MLIP_TYPES_H
#define MLIP_TYPES_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

namespace mlip {

enum class ScalarType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool
};

struct DeviceArrayView {
    void*      data    = nullptr;
    ScalarType dtype   = ScalarType::Float64;
    int        ndim    = 0;
    std::array<int64_t, 4> shape  = {0, 0, 0, 0};
    std::array<int64_t, 4> stride = {0, 0, 0, 0};
    int        device_id = 0;   // CUDA device id (-1 = host)
    bool       is_device = true;
};

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

struct MlipOutputView {
    DeviceArrayView forces;       // [num_total, 3] float64 on device
    DeviceArrayView per_atom_energy;  // [num_total]  float64 on device, optional
    DeviceArrayView virial;       // [9] float64 on device, optional (row-major)

    double* total_energy = nullptr;   // device scalar
    bool    has_virial   = false;

    cudaStream_t stream = nullptr;
};

struct MlipExecutionContext {
    int mpi_rank = 0;
    int mpi_size = 1;

    int          device_id = 0;
    cudaStream_t stream    = nullptr;

    void* communicator      = nullptr;  // opaque MPI_Comm, cast as needed
    void* memory_resource   = nullptr;  // custom allocator, optional
};

struct MlipModelConfig {
    std::string adapter_name;    // "mace", "nequip", "deepmd", "onnx", ...
    std::string backend_name;    // "libtorch", "onnx_cuda", "tensorrt", ...
    std::string model_path;

    std::string precision;       // "float32" or "float64"
    bool        external_neighbor_list = true;
    bool        share_cuda_stream      = true;
    bool        allow_cpu_fallback     = false;

    std::vector<int> md_type_to_atomic_number;

    std::string length_unit;  // "angstrom", "bohr", ...
    std::string energy_unit;  // "eV", "kcal/mol", "hartree", ...

    double neighbor_skin         = 0.5;
    int    rebuild_check_interval = 10;
};

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

struct SpeciesMapping {
    std::vector<int> md_type_to_atomic_number;

    std::array<int, 119> atomic_number_to_model_index;

    SpeciesMapping() {
        atomic_number_to_model_index.fill(-1);
    }

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
