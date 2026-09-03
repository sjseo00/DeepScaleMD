#ifndef MD_SIMULATION_H
#define MD_SIMULATION_H

#include "md_system.h"
#include "force_calculator.h"
#include "integrator.h"
#include "thermostat.h"
#include "barostat.h"
#include "neighbor_list.h"
#include <map>
#include <memory>
#include <vector>

// Forward-declare MlipRuntime so MD core does NOT need mlip headers
namespace mlip { class MlipRuntime; }
class HaloBackend;
class CommBackend;
class ForceCalculatorCUDA;

class MDSimulation {
private:
    enum class OutputField {
        STEP,
        T,
        KE,
        PE,
        ETOTAL,
        PRESS,
        MAX_SPEED,
        AVG_SPEED,
        FMAX
    };

    std::unique_ptr<MDSystem> system;
    std::unique_ptr<ForceCalculator> force_calc;
    std::unique_ptr<HaloBackend> halo_backend_;
    std::unique_ptr<CommBackend> comm_backend_;
    std::unique_ptr<Integrator> integrator;
    std::unique_ptr<Thermostat> thermostat;
    std::unique_ptr<Barostat> barostat;
    std::unique_ptr<NeighborListBuilder> neighbor_builder;
    std::unique_ptr<PythonForceInterface> python_interface;
    ForceCalculatorCUDA* cuda_calc_cached_;
    bool using_cuda_backend_cached_;

    // ── LibTorch MLIP runtime ─────────────────────────────────────────────
    std::unique_ptr<mlip::MlipRuntime> mlip_runtime_;
    std::string mlip_spec_file_;    // path to per-model YAML spec (full schema)
    
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
    // Comma-separated enabled components: builtin, python, libtorch.
    // Components are evaluated in that order and add to the same force buffer.
    std::string force_calculation_mode;
    bool enable_builtin_forces_ = true;
    bool enable_python_forces_ = false;
    bool enable_libtorch_forces_ = false;
    std::string force_calculation_type;  // "mace", "nequip", "lj", etc.

    // Model-agnostic configuration for the Python MLIP adapter.  `mace_config`
    // remains accepted by the parser solely as a legacy YAML section name.
    std::string mlip_model_path;
    std::string mlip_model_type;
    std::string mlip_model_size;
    std::string mlip_device;
    std::string mlip_dtype;
    std::map<std::string, std::string> mlip_extra_kwargs;  // extra mlip_config params (e.g. head)

    std::string integrator_type;
    std::string thermostat_type;
    Real tau_t;
    std::string barostat_type;
    Real target_pressure;
    Real tau_p;
    
    // Output control variables
    int output_freq;                    // General output frequency
    std::string output_info;            // What information to output (comma-separated)
    std::vector<OutputField> output_fields_cache;
    
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
    bool timing_every_step;
    struct StepTiming {
        double domain_ms = 0.0;
        double nlist_build_ms = 0.0;
        double comm_ms = 0.0;
        double migrate_ms = 0.0;
        double force_ms = 0.0;
        double integrate_ms = 0.0;
        double thermostat_ms = 0.0;
        double ke_ms = 0.0;
        double barostat_ms = 0.0;
        double io_ms = 0.0;
        double total_ms = 0.0;
    } last_timing;
    
    // Cumulative timing statistics (LAMMPS-style)
    struct CumulativeTiming {
        double neighbor_time = 0.0;     // Neighbor list rebuild/build
        double force_time = 0.0;        // Force calculation
        double integration_time = 0.0;  // Integrate1 + Integrate2
        double thermostat_time = 0.0;   // Thermostat apply
        double barostat_time = 0.0;     // Barostat apply (+ rebuild trigger handling)
        double ke_time = 0.0;           // Kinetic energy refresh for outputs
        double comm_time = 0.0;         // Halo communication/exchange
        double migration_time = 0.0;    // Escaped-particle migration
        double output_time = 0.0;       // Output/IO operations
        double total_time = 0.0;        // Total simulation time
        int step_count = 0;             // Number of steps timed
    } cumulative_timing;

    struct OutputMetrics {
        Real total_ke = 0.0;
        Real total_pe = 0.0;
        Real max_force = 0.0;
        size_t max_force_particle_id = 0;
        Real max_speed = 0.0;
        Real avg_speed = 0.0;
        Real temperature = 0.0;
        Real pressure = 0.0;
    };
    
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
    void rebuild_output_info_items_cache();
    void prepare_first_step();          // Initialize forces before first MD step
    bool should_output_step(int step) const;
    bool needs_ke_update_for_step(int step) const;
    bool should_write_trajectory_step(int step) const;
    bool should_print_timing_for_step(int step) const;
    void handle_step_output(int step);
    void initialize_integrator();
    void initialize_thermostat();
    void initialize_barostat();
    bool is_using_cuda_backend() const;
    void handle_integrate1_phase(ForceCalculatorCUDA* cuda_calc, double& integrate1_time);
    void handle_neighbor_phase(double& nlist_build_time, double& comm_time, double& migrate_time);
    void handle_neighbor_rebuild_if_needed(bool should_rebuild);
    void handle_force_phase(ForceCalculatorCUDA* cuda_calc, double& force_time);
    void handle_integrate2_phase(ForceCalculatorCUDA* cuda_calc, double& integrate2_time);
    void handle_thermostat_phase(ForceCalculatorCUDA* cuda_calc, double& thermostat_time);
    void handle_ke_output_phase(double& ke_time);
    void handle_barostat_phase(ForceCalculatorCUDA* cuda_calc, double& barostat_time);
    bool has_significant_box_change(Real old_lx, Real old_ly, Real old_lz,
                                    Real new_lx, Real new_ly, Real new_lz) const;
    void handle_modify_phase(ForceCalculatorCUDA* cuda_calc,
                             double& thermostat_time, double& ke_time,
                             double& barostat_time, double& modify_time);
    void accumulate_step_timing(double integrate1_time, double nlist_build_time, double comm_time,
                                double migrate_time,
                                double force_time, double integrate2_time,
                                double thermostat_time, double ke_time,
                                double barostat_time, double modify_time,
                                double total_time);
    void handle_box_change_rebuild();
    void sync_output_state_to_host_if_needed();
    void compute_output_energies(const ParticleData& particles, OutputMetrics& metrics);
    void compute_output_force_speed_stats(const ParticleData& particles, OutputMetrics& metrics);
    void compute_output_thermodynamics(const ParticleData& particles, OutputMetrics& metrics);
    void emit_rank0_output(int step, const OutputMetrics& metrics);
    std::string format_output_line(int step, const OutputMetrics& metrics) const;
    void compute_forces_with_mode(bool download_positions_for_ml);
    void reverse_halo_for_bonded_if_needed();
    void check_energy_stability();      // Check for energy drift and instabilities
    void zero_forces();                 // Zero all forces (local + ghost particles)
    
    // Trajectory output functions
    void write_trajectory(int step);
    void write_xyz_trajectory(int step);
    void write_lammps_dump(int step);
};

#endif // MD_SIMULATION_H
