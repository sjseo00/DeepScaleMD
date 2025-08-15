#define OMPI_SKIP_MPICXX 1
#include "force_manager.h"
#include "force_calculator_cpu.h"
#include <mpi.h>

#ifdef USE_CUDA
#include "force_calculator_cuda.h" 
#endif

#ifdef USE_SYCL
#include "force_calculator_sycl.h"
#endif

#include <iostream>
#include <map>
#include <mpi.h>

// Helper function to check if this is rank 0
static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

ForceManager::ForceManager() : backend_type("cpu") {
    config.mode = ForceMode::BUILTIN;
}

void ForceManager::initialize(const ForceConfig& force_config, const std::string& backend) {
    config = force_config;
    backend_type = backend;
    
    if (is_rank_zero()) {
        std::cout << "Initializing Force Manager:" << std::endl;
        std::cout << "  Mode: " << (config.mode == ForceMode::BUILTIN ? "Built-in" : "Python") << std::endl;
        std::cout << "  Backend: " << backend_type << std::endl;
    }
    
    if (config.mode == ForceMode::BUILTIN) {
        active_calculator = create_builtin_calculator();
    } else {
        active_calculator = create_python_calculator();
    }
    
    if (active_calculator) {
        active_calculator->initialize();
    } else {
        throw std::runtime_error("Failed to create force calculator");
    }
}

void ForceManager::finalize() {
    if (active_calculator) {
        active_calculator->finalize();
        active_calculator.reset();
    }
}

void ForceManager::compute_forces(MDSystem& system) {
    if (!active_calculator) {
        throw std::runtime_error("Force calculator not initialized");
    }
    
    active_calculator->compute_forces(system);
}

std::unique_ptr<ForceCalculator> ForceManager::create_builtin_calculator() {
    if (is_rank_zero()) {
        std::cout << "Creating built-in force calculator for " << backend_type << " backend" << std::endl;
    }
    
    if (backend_type == "cpu") {
        auto calc = std::make_unique<ForceCalculatorCPU>();
        return calc;
    }
#ifdef USE_CUDA
    else if (backend_type == "cuda") {
        auto calc = std::make_unique<ForceCalculatorCUDA>();
        return calc;
    }
#endif
#ifdef USE_SYCL  
    else if (backend_type == "sycl") {
        auto calc = std::make_unique<ForceCalculatorSYCL>();
        return calc;
    }
#endif
    else {
        if (is_rank_zero()) {
            std::cout << "Warning: Unknown backend " << backend_type << ", falling back to CPU" << std::endl;
        }
        auto calc = std::make_unique<ForceCalculatorCPU>();
        return calc;
    }
}

std::unique_ptr<ForceCalculator> ForceManager::create_python_calculator() {
    if (is_rank_zero()) {
        std::cout << "Creating Python force calculator interface" << std::endl;
    }
    
    // Select appropriate Python-interfaced calculator based on backend
    if (backend_type == "cpu") {
        return std::make_unique<ForceCalculatorCPU>();
    }
#ifdef USE_CUDA
    else if (backend_type == "cuda") {
        return std::make_unique<ForceCalculatorCUDA>();
    }
#endif
#ifdef USE_SYCL
    else if (backend_type == "sycl") {
        return std::make_unique<ForceCalculatorSYCL>();
    }
#endif
    else {
        if (is_rank_zero()) {
            std::cout << "Warning: Unknown backend for Python interface, using CPU" << std::endl;
        }
        return std::make_unique<ForceCalculatorCPU>();
    }
}

ForceManager::ForceConfig ForceManager::parse_config_from_map(const std::map<std::string, std::string>& config_map) {
    ForceConfig config;
    
    // Parse force calculation mode
    auto mode_it = config_map.find("force_calculation_mode");
    if (mode_it != config_map.end()) {
        if (mode_it->second == "builtin") {
            config.mode = ForceMode::BUILTIN;
        } else if (mode_it->second == "python") {
            config.mode = ForceMode::PYTHON;
        }
    }
    
    // Parse built-in force field configuration
    auto bonds_it = config_map.find("builtin_enable_bonds");
    if (bonds_it != config_map.end()) {
        config.enable_bonds = (bonds_it->second == "true");
    }
    
    auto angles_it = config_map.find("builtin_enable_angles");
    if (angles_it != config_map.end()) {
        config.enable_angles = (angles_it->second == "true");
    }
    
    auto dihedrals_it = config_map.find("builtin_enable_dihedrals");
    if (dihedrals_it != config_map.end()) {
        config.enable_dihedrals = (dihedrals_it->second == "true");
    }
    
    auto lj_it = config_map.find("builtin_enable_lennard_jones");
    if (lj_it != config_map.end()) {
        config.enable_lennard_jones = (lj_it->second == "true");
    }
    
    auto coulomb_it = config_map.find("builtin_enable_coulomb");
    if (coulomb_it != config_map.end()) {
        config.enable_coulomb = (coulomb_it->second == "true");
    }
    
    // Parse Python interface configuration
    auto python_enabled_it = config_map.find("python_enabled");
    if (python_enabled_it != config_map.end()) {
        config.python_enabled = (python_enabled_it->second == "true");
    }
    
    auto python_path_it = config_map.find("python_force_path");
    if (python_path_it != config_map.end()) {
        config.python_force_path = python_path_it->second;
    }
    
    return config;
}
