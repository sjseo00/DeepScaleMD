#ifndef MLIP_BACKEND_H
#define MLIP_BACKEND_H

// No LibTorch headers here: MD code and PythonBackend must build without LibTorch.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

// Pointers are valid only during compute(); MD may reallocate its buffers between calls.
// Slots [0, n_owned) are owned atoms, [n_owned, n_all) are ghosts with periodic shifts applied.
struct MLIPInputs {
    const double*  d_x = nullptr;             // [n_all], device
    const double*  d_y = nullptr;             // [n_all], device
    const double*  d_z = nullptr;             // [n_all], device
    const int*     d_type = nullptr;          // [n_all], device, MD type id (0-based)
    const int64_t* d_edge_index = nullptr;    // [2 * n_edges], device
    int64_t n_owned = 0;
    int64_t n_all = 0;
    int64_t n_edges = 0;
    cudaStream_t stream = nullptr;            // ForceCalculatorCUDA force_stream
    bool host_memory = false;                 // d_* pointers (and MLIPOutputs) are host memory: CPU calculator
    double cell[3] = {0.0, 0.0, 0.0};         // orthorhombic box lengths (A); used by PythonBackend
    bool pbc[3] = {false, false, false};      // periodic flags; used by PythonBackend
};

// Outputs are in MD units (kcal/mol, kcal/mol/A); the backend converts from model units.
struct MLIPOutputs {
    double* d_fx = nullptr;                   // [n_all], device, added to (not overwritten)
    double* d_fy = nullptr;                   // [n_all], device, added to
    double* d_fz = nullptr;                   // [n_all], device, added to
    double* d_atom_energy = nullptr;          // [n_all], device, nullptr if not needed. Overwritten (not added to)
    bool    need_energy = false;              // energy is copied to the host only when true
    double  energy = 0.0;                     // owned-atom energy sum, valid if need_energy
};

// The former mlip_config section keys (docs/04): everything the Python MLIP path takes.
struct PythonMLIPConfig {
    std::string force_path = "python_force";  // python_force_path
    std::string model_path;                   // mlip_config.model_path
    std::string model_type;                   // mlip_config.model_type, or force_calculation.type if empty
    std::string model_size = "small";
    std::string device = "cpu";
    std::string dtype = "float32";
    std::map<std::string, std::string> extra_kwargs;  // other mlip_config keys, passed to Python
};

struct MLIPConfig {
    std::string backend;                      // "python" | "libtorch"
    std::string model_path;                   // converted model folder (libtorch)
    std::vector<int> type_to_z;               // MD type id -> atomic number
    bool verbose = false;                     // print progress messages (MPI rank 0 only)
    PythonMLIPConfig python;                  // used when backend == "python"
};

class MLIPBackend {
public:
    virtual ~MLIPBackend() = default;
    virtual void compute(const MLIPInputs& in, MLIPOutputs& out) = 0;
    virtual double cutoff() const = 0;        // neighbor-list cutoff (A); 0 = not applicable (PythonBackend)
    virtual double ghost_cutoff() const = 0;  // ghost depth without skin (A); 0 = keep MD's own halo width
};

// Defined in MLIPBackendFactory.cpp. An empty cfg.backend means "python".
// validate_mlip_config() throws std::runtime_error for an unknown backend, for 'libtorch' in a build without
// LibTorch (CPU/SYCL builds, or CUDA builds without ENABLE_LIBTORCH), and for incomplete libtorch settings.
// create_mlip_backend() validates first, then constructs the backend.
void validate_mlip_config(const MLIPConfig& cfg);
std::unique_ptr<MLIPBackend> create_mlip_backend(const MLIPConfig& cfg);

#endif // MLIP_BACKEND_H
