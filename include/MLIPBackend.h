#ifndef MLIP_BACKEND_H
#define MLIP_BACKEND_H

// No LibTorch headers here: MD code and PythonBackend must build without LibTorch.

#include "MLIPHaloTransport.h"
#include "MLIPRankComm.h"

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
    const double*  d_edge_shift = nullptr;    // [3 * n_edges], device, [E, 3] row-major: the vector added to the position difference
                                              // of edge e (A), a lattice vector for a neighbor across a periodic boundary;
                                              // nullptr = all zero. Read only by a backend whose uses_edge_shifts() is true
    int64_t n_owned = 0;
    int64_t n_all = 0;
    int64_t n_edges = 0;
    cudaStream_t stream = nullptr;            // ForceCalculatorCUDA force_stream
    bool host_memory = false;                 // d_* pointers (and MLIPOutputs) are host memory: CPU calculator
    double cell[3] = {0.0, 0.0, 0.0};         // orthorhombic box lengths (A); used by PythonBackend
    bool pbc[3] = {false, false, false};      // periodic flags; used by PythonBackend
    MLIPHaloTransport* halo = nullptr;        // how a layer-exchange model (exchanges_layers()) moves node features between the ranks
                                              // (the MD's halo exchange); nullptr = no ghost atoms to exchange with (one rank, or the
                                              // replicated step)
};

// Outputs are in MD units (kcal/mol, kcal/mol/A); the backend converts from model units.
struct MLIPOutputs {
    double* d_fx = nullptr;                   // [n_all], device, added to (not overwritten)
    double* d_fy = nullptr;                   // [n_all], device, added to
    double* d_fz = nullptr;                   // [n_all], device, added to
    double* d_atom_energy = nullptr;          // [n_all], device, nullptr if not needed. Overwritten (not added to)
    bool    need_energy = false;              // energy is copied to the host only when true
    double  energy = 0.0;                     // owned-atom energy sum, valid if need_energy
    double  virial = 0.0;                     // sum of r.F over the atoms the model saw plus the edge-shift term (kcal/mol); only
    bool    has_virial = false;               // set (with need_energy) by a backend that returns it, i.e. one with edge shifts
};

// The former mlip_config section keys (docs/04): everything the Python MLIP path takes.
struct PythonMLIPConfig {
    std::string module_path = "MLIP_tools/mlip_python";  // python_mlip_path: the directory with ml_forces.py (relative to the working directory, or absolute)
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
    bool allow_tf32 = false;                  // libtorch: float32 matrix products on the TF32 tensor cores (advanced.tf32); faster, ~10 bits
                                              // of mantissa in the products; needs an Ampere or newer GPU
    PythonMLIPConfig python;                  // used when backend == "python"
    MLIPRankComm* comm = nullptr;             // the other MPI ranks (not owned); nullptr = a single process. PythonBackend uses it
                                              // on several ranks: every rank then computes all atoms (see PythonBackend.h)
};

class MLIPBackend {
public:
    virtual ~MLIPBackend() = default;
    virtual void compute(const MLIPInputs& in, MLIPOutputs& out) = 0;
    virtual double cutoff() const = 0;        // neighbor-list cutoff (A); 0 = not applicable (PythonBackend)
    virtual double ghost_cutoff() const = 0;  // ghost depth without skin (A); 0 = keep MD's own halo width
    // True: the model takes periodic images as edge shifts (MLIPInputs::d_edge_shift) and gives the virial, so the MD side
    // does not copy atoms across the periodic boundary for it.
    virtual bool uses_edge_shifts() const { return false; }
    // 0 or 1: the row of the edge list that the model sums its messages onto, so the MD can leave out the edges that cannot change
    // the energy of the owned atoms (EdgeBuilder::Prune); -1 = not known, every edge is kept.
    virtual int edge_receiver_row() const { return -1; }
    // True: the model exchanges node features between its layers (metadata layer_exchange). The MD then gives it ghost atoms only
    // one cutoff deep, builds edges onto owned atoms only (MLIPStepGeometry::owned_receivers_only), and sets MLIPInputs::halo.
    virtual bool exchanges_layers() const { return false; }
};

// Defined in MLIPBackendFactory.cpp. An empty cfg.backend means "python".
// validate_mlip_config() throws std::runtime_error for an unknown backend, for 'libtorch' in a build without
// LibTorch (CPU/SYCL builds, or CUDA builds without ENABLE_LIBTORCH), and for incomplete libtorch settings.
// create_mlip_backend() validates first, then constructs the backend.
void validate_mlip_config(const MLIPConfig& cfg);
std::unique_ptr<MLIPBackend> create_mlip_backend(const MLIPConfig& cfg);

#endif // MLIP_BACKEND_H
