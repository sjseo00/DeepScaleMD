#ifndef MD_SIMULATION_H
#define MD_SIMULATION_H

#include "md_system.h"
#include "force_calculator.h"
#include "integrator.h"
#include "thermostat.h"
#include "barostat.h"
#include "neighbor_list.h"
#include <memory>

// Global debug level variable
extern int g_debug_level;

// Helper functions for debug output
inline bool debug_level(int level) { return g_debug_level >= level; }

class MDSimulation {
private:
    std::unique_ptr<MDSystem> system;
    std::unique_ptr<ForceCalculator> force_calc;
    std::unique_ptr<Integrator> integrator;
    std::unique_ptr<Thermostat> thermostat;
    std::unique_ptr<Barostat> barostat;
    std::unique_ptr<NeighborListBuilder> neighbor_builder;
    std::unique_ptr<PythonForceInterface> python_interface;
    
    // Configuration variables
    std::string coordinate_file;
    std::string topology_file;
    Real target_temperature;
    std::string output_prefix;
    int trajectory_output_freq;
    int energy_output_freq;
    Real cutoff_distance;
    std::string neighbor_list_type_str;
    int neighbor_list_tile_size;
    int neighbor_list_update_freq;  // Added neighbor list update frequency
    Real verlet_skin;               // Verlet list skin distance
    std::string force_calculation_mode;  // "builtin" or "python"
    std::string integrator_type;
    std::string thermostat_type;
    Real tau_t;
    std::string barostat_type;
    Real target_pressure;
    Real tau_p;
    
    // Output control variables
    int output_freq;                    // General output frequency
    std::string output_info;            // What information to output (comma-separated)
    
    // Trajectory output settings
    std::string trajectory_format;      // "xyz", "dump", or "both"
    bool trajectory_include_velocities; // Include velocities in output
    bool trajectory_include_forces;     // Include forces in output
    
    int current_step;
    int total_steps;
    Real dt;
    std::string python_force_path;
    bool python_force_enabled;
    
    // Cached energy values
    Real cached_potential_energy;       // Store PE from force calculation
    
public:
    MDSimulation();
    ~MDSimulation();
    
    void initialize(const std::string& config_file);
    void run();
    void finalize();
    
    void single_step();
    void output_results(int step, double step_time_ms = 0.0);
    
    // Getters
    MDSystem* get_system() { return system.get(); }
    
private:
    void parse_simulation_config(const std::string& config_file);
    void initialize_integrator();
    void initialize_thermostat();
    void initialize_barostat();
    void calculate_potential_energy();  // Calculate and cache PE
    void check_energy_stability();      // Check for energy drift and instabilities
    
    // Trajectory output functions
    void write_trajectory(int step);
    void write_xyz_trajectory(int step);
    void write_lammps_dump(int step);
};

#endif // MD_SIMULATION_H
