#ifndef FORCE_CALCULATOR_H
#define FORCE_CALCULATOR_H

#include "md_types.h"
#include <map>
#include <string>

// Forward declaration to avoid circular dependency
class MDSystem;

class ForceCalculator {
protected:
    Real last_potential_energy = 0.0;   // Last computed PE
    Real last_kinetic_energy = 0.0;     // Last computed KE
    Real last_virial = 0.0;             // Last computed virial (for pressure calculation)
    
public:
    virtual ~ForceCalculator() = default;
    virtual void compute_forces(MDSystem& system) = 0;
    virtual void initialize() = 0;
    virtual void finalize() = 0;
    
    // Energy accessors
    virtual Real get_potential_energy() const { return last_potential_energy; }
    virtual Real get_kinetic_energy() const { return last_kinetic_energy; }
    virtual Real get_virial() const { return last_virial; }
    
    // Energy calculation methods
    virtual void calculate_kinetic_energy(MDSystem& system) = 0;
};

// Python interface for forces
class PythonForceInterface {
private:
    std::string python_force_path;  // Path to Python force modules
    bool is_initialized;
    int mpi_rank;
    int mpi_size;
    
    // Cached Python objects (to avoid repeated imports)
    void* py_module;                 // PyObject* (force_calculator module)
    void* py_ml_module;              // PyObject* (ml_forces module)
    void* py_compute_func;           // PyObject* (compute function)
    void* py_ml_compute_func;        // PyObject* (ML compute function)

    // Per-step zero-malloc optimisation buffers
    void* py_cached_atom_types;      // PyObject* – atom types never change; created once
    void* py_cached_pbc;             // PyObject* – PBC flags never change; created once
    std::vector<double> _pos_buf;    // interleaved position buffer [x0,y0,z0, ...]
    std::vector<double> _frc_buf;    // interleaved force buffer [fx0,fy0,fz0, ...]
    size_t _cached_n_particles;      // particle count when caches were built
    
public:
    PythonForceInterface();
    ~PythonForceInterface();
    
    void initialize_python();
    void initialize_python(const std::string& force_module_path);
    void finalize_python();
    void set_force_module_path(const std::string& path);

    // Initialize the user-supplied Python/Cython force provider.
    void initialize_ml_model(const std::string& model_path,
                             const std::string& model_type = "",
                             const std::string& model_size = "small",
                             const std::string& device = "cpu",
                             const std::string& dtype = "float32",
                             const std::map<std::string, std::string>& extra_kwargs = {});
    
    // Add the user force to particles.fx/fy/fz.  The Python ABI is
    // compute_force(positions, types, cell, pbc) -> [N,3].  NumPy views wrap
    // C++-owned buffers, so Cython providers receive pointers without a copy.
    void compute_force(ParticleData& particles, const ForceFieldParams& params,
                       const NeighborList& nlist, double box_x = 50.0,
                       double box_y = 50.0, double box_z = 50.0,
                       bool pbc_x = true, bool pbc_y = true, bool pbc_z = true);
    
    // Individual force computation methods (legacy, for backwards compatibility)
    void compute_bond_forces(ParticleData& particles, const ForceFieldParams& params);
    void compute_angle_forces(ParticleData& particles, const ForceFieldParams& params);
    void compute_dihedral_forces(ParticleData& particles, const ForceFieldParams& params);
    void compute_lj_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist);
    void compute_electrostatic_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist);
    void compute_ml_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist,
                           double box_x = 50.0, double box_y = 50.0, double box_z = 50.0,
                           bool pbc_x = true, bool pbc_y = true, bool pbc_z = true);
};

#endif // FORCE_CALCULATOR_H
