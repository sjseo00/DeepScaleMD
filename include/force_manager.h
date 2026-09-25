#ifndef FORCE_MANAGER_H
#define FORCE_MANAGER_H

#include "force_calculator.h"
#include "md_types.h"
#include <memory>
#include <string>
#include <map>
#include <vector>

class MDSystem;

class ForceManager {
public:
    enum class ForceMode {
        BUILTIN,  // Use built-in optimized force modules  
        PYTHON    // Use external Python force interface
    };
    
    struct ForceConfig {
        ForceMode mode = ForceMode::BUILTIN;
        
        bool enable_bonds = true;
        bool enable_angles = true; 
        bool enable_dihedrals = true;
        bool enable_lennard_jones = true;
        bool enable_coulomb = true;
        
        std::vector<std::string> force_types;  // e.g., {"bonds", "angles", "lj", "PME"}
        
        bool python_enabled = false;
        std::string python_force_path = "../python_force";
        std::string python_path = "";  // New: direct path for python mode
    };

private:
    std::unique_ptr<ForceCalculator> active_calculator;
    ForceConfig config;
    std::string backend_type; // "cpu", "cuda", "sycl"
    
    std::unique_ptr<ForceCalculator> create_builtin_calculator();
    std::unique_ptr<ForceCalculator> create_python_calculator();
    
public:
    ForceManager();
    ~ForceManager() = default;
    
    void initialize(const ForceConfig& force_config, const std::string& backend = "cpu");
    void finalize();
    
    void compute_forces(MDSystem& system);
    
    void set_backend(const std::string& backend) { backend_type = backend; }
    const std::string& get_backend() const { return backend_type; }
    
    ForceMode get_mode() const { return config.mode; }
    const ForceConfig& get_config() const { return config; }
    
    static ForceConfig parse_config_from_map(const std::map<std::string, std::string>& config_map);
};

#endif // FORCE_MANAGER_H
