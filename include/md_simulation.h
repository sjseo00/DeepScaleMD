#ifndef MD_SIMULATION_H
#define MD_SIMULATION_H

#include "md_system.h"
#include "force_calculator.h"
#include "integrator.h"
#include "thermostat.h"
#include "barostat.h"
#include "neighbor_list.h"
#include <memory>

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
    int neighbor_list_builds;       // Count of neighbor list builds
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
    
    // Critical simulation state flags (LAMMPS-style)
    bool neighbor_rebuild_forced;       // Force neighbor rebuild (e.g., after barostat)
    
    // Timing infrastructure
    bool timing_enabled;
    struct StepTiming {
        double domain_ms = 0.0;
        double nlist_build_ms = 0.0;
        double migrate_ms = 0.0;
        double force_ms = 0.0;
        double integrate_ms = 0.0;
        double thermostat_ms = 0.0;
        double ke_ms = 0.0;
        double barostat_ms = 0.0;
        double total_ms = 0.0;
    } last_timing;
    
    // Cumulative timing statistics (LAMMPS-style)
    struct CumulativeTiming {
        double pair_time = 0.0;      // Force calculation (Pair in LAMMPS)
        double neigh_time = 0.0;     // Neighbor list build
        double comm_time = 0.0;      // MPI communication (halo exchange)
        double output_time = 0.0;    // Output/IO operations
        double modify_time = 0.0;    // Thermostat/Barostat/Integrate
        double other_time = 0.0;     // Everything else
        double total_time = 0.0;     // Total simulation time
        int step_count = 0;          // Number of steps timed
    } cumulative_timing;
    
    void print_step_timing(int step);
    void print_timing_breakdown();   // Print LAMMPS-style timing breakdown
    
public:
    MDSimulation();
    ~MDSimulation();
    
    void initialize(const std::string& config_file);
    void run();
    void finalize();
    
    void single_step();
    void output_results(int step);
    void force_neighbor_rebuild() { neighbor_rebuild_forced = true; }  // Force rebuild (e.g., after barostat)
    
    // Getters
    MDSystem* get_system() { return system.get(); }
    
private:
    void parse_simulation_config(const std::string& config_file);
    void prepare_first_step();          // Initialize forces before first MD step
    void initialize_integrator();
    void initialize_thermostat();
    void initialize_barostat();
    void calculate_potential_energy();  // Calculate and cache PE
    void check_energy_stability();      // Check for energy drift and instabilities
    void zero_forces();                 // Zero all forces (local + ghost particles)
    
    // Trajectory output functions
    void write_trajectory(int step);
    void write_xyz_trajectory(int step);
    void write_lammps_dump(int step);
};

#endif // MD_SIMULATION_H
