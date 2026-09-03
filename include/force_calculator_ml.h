#ifndef FORCE_CALCULATOR_ML_H
#define FORCE_CALCULATOR_ML_H

#include "force_calculator.h"
#include "md_types.h"
#include <string>

class MDSystem;

/**
 * @brief Force calculator using Machine Learning Potentials (MACE, NequIP, etc.)
 * 
 * This calculator interfaces with Python ML force modules to compute
 * forces and energies using trained neural network potentials.
 * 
 * Workflow:
 * 1. C++ MD simulation prepares particle positions, types, cell
 * 2. Data is passed to Python via PythonForceInterface
 * 3. Python ml_forces.py calls MACE/NequIP calculator
 * 4. Forces and energy are returned to C++
 * 5. MD simulation continues with ML forces
 */
class ForceCalculatorML : public ForceCalculator {
private:
    PythonForceInterface* python_interface;
    bool python_initialized;
    std::string ml_model_type;  // "mace" or "nequip"
    std::string model_path;
    std::string device;         // "cpu" or "cuda"
    std::string dtype;          // "float32" or "float64"
    
    int mpi_rank;
    int mpi_size;
    
    // Performance tracking
    double total_ml_time;
    int ml_call_count;
    
public:
    ForceCalculatorML();
    ~ForceCalculatorML();
    
    // ForceCalculator interface implementation
    void initialize() override;
    void finalize() override;
    void compute_forces(MDSystem& system) override;
    void calculate_kinetic_energy(MDSystem& system) override;
    
    // ML-specific configuration
    void set_ml_model_type(const std::string& type);
    void set_model_path(const std::string& path);
    void set_device(const std::string& dev);
    void set_dtype(const std::string& dt);
    
    // Performance info
    double get_average_ml_time() const;
    void print_performance_stats() const;
};

#endif // FORCE_CALCULATOR_ML_H
