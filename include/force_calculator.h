#ifndef FORCE_CALCULATOR_H
#define FORCE_CALCULATOR_H

#include "md_types.h"
#include <map>
#include <string>
#include <vector>

class MDSystem;
class NeighborList;

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
    virtual void set_nonbonded_kernel(const std::string& kernel) {
        nonbonded_kernel_ = kernel;
    }
    const std::string& get_nonbonded_kernel() const { return nonbonded_kernel_; }

    virtual void set_mixed_precision(bool /*enabled*/) {}

    virtual Real get_potential_energy() const { return last_potential_energy; }
    virtual Real get_kinetic_energy() const { return last_kinetic_energy; }
    virtual Real get_virial() const { return last_virial; }

    virtual void calculate_kinetic_energy(MDSystem& system) = 0;

    struct ForceComponentTiming { std::string name; double ms; };
    virtual void set_timing_breakdown_enabled(bool /*enabled*/) {}
    virtual bool is_timing_breakdown_enabled() const { return false; }
    virtual const std::vector<ForceComponentTiming>& get_last_force_component_timings() const {
        static const std::vector<ForceComponentTiming> empty_timings;
        return empty_timings;
    }

    virtual const std::vector<int>& get_lj_pair_counts() const {
        static const std::vector<int> empty_counts;
        return empty_counts;
    }

protected:
    std::string nonbonded_kernel_ = "tiled";
};

class PythonForceInterface {
private:
    std::string python_force_path;  // Path to Python force modules
    bool is_initialized;
    int mpi_rank;
    int mpi_size;
    
    void* py_module;                 // PyObject* (force_calculator module)
    void* py_ml_module;              // PyObject* (ml_forces module)
    void* py_compute_func;           // PyObject* (compute function)
    void* py_ml_compute_func;        // PyObject* (ML compute function)
    void* py_ml_get_energy_func;     // PyObject* (ml_forces.get_last_energy)

    void* py_cached_atom_types;      // PyObject* – atom types never change; created once
    void* py_cached_pbc;             // PyObject* – PBC flags never change; created once
    std::vector<double> _pos_buf;    // interleaved position buffer [x0,y0,z0, ...]
    std::vector<double> _frc_buf;    // interleaved force buffer [fx0,fy0,fz0, ...]
    size_t _cached_n_particles;      // particle count when caches were built

    double last_computed_energy_ = 0.0;

public:
    PythonForceInterface();
    ~PythonForceInterface();
    
    void initialize_python();
    void initialize_python(const std::string& force_module_path);
    void finalize_python();
    void set_force_module_path(const std::string& path);

    void initialize_ml_model(const std::string& model_path,
                             const std::string& model_type = "",
                             const std::string& model_size = "small",
                             const std::string& device = "cpu",
                             const std::string& dtype = "float32",
                             const std::map<std::string, std::string>& extra_kwargs = {});
    
    void compute_force(ParticleData& particles, const ForceFieldParams& params,
                       const NeighborList& nlist, double box_x = 50.0,
                       double box_y = 50.0, double box_z = 50.0,
                       bool pbc_x = true, bool pbc_y = true, bool pbc_z = true);

    double get_last_energy() const { return last_computed_energy_; }


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
