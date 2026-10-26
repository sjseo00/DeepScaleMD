#ifndef MLIP_PYTHON_BACKEND_H
#define MLIP_PYTHON_BACKEND_H

// The embedded-CPython MLIP path (MLIP_tools/mlip_python/ml_forces.py) behind MLIPBackend. Positions go to Python
// as host arrays and the returned forces are added to the MD force arrays. In a CUDA build the device
// arrays are staged through host memory every call.
// Python.h and NumPy are used only in PythonBackend.cpp (ENABLE_PYTHON_INTERFACE). A build without it refuses this
// backend when it is created; a failure in Python (import, model, force call) stops the run with the Python traceback,
// it never turns into zero forces.
//
// Several MPI ranks (MLIPConfig::comm with size() > 1): an ASE-style MLIP cannot be given owned + ghost atoms of one subdomain
// (the ghosts would be taken for atoms of the system, next to their own periodic images, and the energy would count the ghosts
// too). So every rank collects the owned atoms of all ranks, runs the whole system with the real box and periodic flags, and
// keeps the forces of its own atoms; the energy is the whole system's on every rank. That is correct, but the work is the
// same on every rank (no speed-up): the scalable multi-GPU path is mlip_backend: libtorch.

#include "MLIPBackend.h"

#include <cstddef>
#include <string>
#include <vector>

class PythonBackend : public MLIPBackend {
public:
    explicit PythonBackend(const MLIPConfig& cfg);
    ~PythonBackend() override;

    PythonBackend(const PythonBackend&) = delete;
    PythonBackend& operator=(const PythonBackend&) = delete;

    void compute(const MLIPInputs& in, MLIPOutputs& out) override;
    double cutoff() const override { return 0.0; }        // ASE builds its own neighbor list
    double ghost_cutoff() const override { return 0.0; }  // MD keeps its own halo width

private:
    void initialize_python(const std::string& module_path);
    void initialize_ml_model(const PythonMLIPConfig& py, const std::vector<int>& type_to_z);
    void finalize_python();
    // The host arrays of this rank (n atoms, the first n_owned are owned): on one rank straight to call_python; on several the
    // atoms of all ranks go to call_python and the forces of the owned atoms are added to fx/fy/fz[0, n_owned).
    void run_on_host(const double* x, const double* y, const double* z, const int* type, size_t n, size_t n_owned,
                     double* fx, double* fy, double* fz, const double* cell, const bool* pbc);
    // Calls ml_forces.compute_force on host arrays and adds the result to fx/fy/fz[0, n).
    // The Python side gets a new atom type array whenever `type` differs from the previous call (the order of the atoms
    // changes when atoms move to another rank).
    void call_python(const double* x, const double* y, const double* z, const int* type, size_t n,
                     double* fx, double* fy, double* fz, const double* cell, const bool* pbc);

    std::string python_mlip_path_ = "MLIP_tools/mlip_python";
    void* py_ml_module_ = nullptr;             // PyObject* (ml_forces module)
    void* py_ml_compute_func_ = nullptr;       // PyObject* (ml_forces.compute_force)
    void* py_ml_get_energy_func_ = nullptr;    // PyObject* (ml_forces.get_last_energy)
    void* py_cached_atom_types_ = nullptr;     // PyObject* int32 array, rebuilt when the atom types change
    void* py_cached_pbc_ = nullptr;            // PyObject* int32 array, created once
    std::vector<int> cached_types_;            // the types py_cached_atom_types_ was built from
    MLIPRankComm* comm_ = nullptr;             // MLIPConfig::comm
    std::vector<double> pos_buf_;              // interleaved positions [x0,y0,z0, ...]
    std::vector<double> frc_buf_;              // interleaved forces
    double last_energy_ = 0.0;                 // kcal/mol, from ml_forces.get_last_energy

    // Host staging for device-resident MD arrays (CUDA builds).
    std::vector<double> h_x_, h_y_, h_z_;
    std::vector<double> h_fx_, h_fy_, h_fz_;
    std::vector<int> h_type_;
    int64_t staged_type_count_ = -1;  // atom count h_type_ was last filled for (one rank: the types never change)

    // All atoms of all ranks (several ranks): this rank's atoms first, then the others'.
    std::vector<double> g_x_, g_y_, g_z_, g_fx_, g_fy_, g_fz_, g_scratch_d_;
    std::vector<int> g_type_, g_scratch_i_;
};

#endif // MLIP_PYTHON_BACKEND_H
