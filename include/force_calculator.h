#ifndef FORCE_CALCULATOR_H
#define FORCE_CALCULATOR_H

#include "md_types.h"

// Forward declaration to avoid circular dependency
class MDSystem;

class ForceCalculator {
protected:
    Real last_potential_energy = 0.0;   // Last computed PE
    Real last_kinetic_energy = 0.0;     // Last computed KE
    
public:
    virtual ~ForceCalculator() = default;
    virtual void compute_forces(MDSystem& system) = 0;
    virtual void initialize() = 0;
    virtual void finalize() = 0;
    
    // Energy accessors
    virtual Real get_potential_energy() const { return last_potential_energy; }
    virtual Real get_kinetic_energy() const { return last_kinetic_energy; }
    
    // Energy calculation methods
    virtual void calculate_kinetic_energy(MDSystem& system) = 0;
};

// Python interface for forces
class PythonForceInterface {
private:
    std::string python_force_path;  // Path to Python force modules
    
public:
    PythonForceInterface();
    ~PythonForceInterface();
    
    void initialize_python();
    void initialize_python(const std::string& force_module_path);
    void finalize_python();
    void set_force_module_path(const std::string& path);
    
    // Unified force computation method
    void compute_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist);
    
    // Individual force computation methods (legacy, for backwards compatibility)
    void compute_bond_forces(ParticleData& particles, const ForceFieldParams& params);
    void compute_angle_forces(ParticleData& particles, const ForceFieldParams& params);
    void compute_dihedral_forces(ParticleData& particles, const ForceFieldParams& params);
    void compute_lj_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist);
    void compute_electrostatic_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist);
    void compute_ml_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist);
};

#endif // FORCE_CALCULATOR_H
