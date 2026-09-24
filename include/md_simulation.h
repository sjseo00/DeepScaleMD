#ifndef MD_SIMULATION_H
#define MD_SIMULATION_H

#include "md_system.h"
#include "force_calculator.h"
#include "integrator.h"
#include "thermostat.h"
#include "barostat.h"
#include "neighbor_list.h"
#include "minimizer.h"
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_ADIOS2
#include <adios2.h>
#endif

#ifdef USE_HDF5
#include <hdf5.h>
#endif

// Forward-declare MlipRuntime so MD core does NOT need mlip headers
namespace mlip { class MlipRuntime; }
class HaloBackend;
class CommBackend;
class ForceCalculatorCUDA;
class ConstraintSolver;

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
    std::unique_ptr<NeighborList> neighbor_list_;
    std::unique_ptr<PythonForceInterface> python_interface;
    ForceCalculatorCUDA* cuda_calc_cached_;
    bool using_cuda_backend_cached_;

    // ── LibTorch MLIP runtime ─────────────────────────────────────────────
    std::unique_ptr<mlip::MlipRuntime> mlip_runtime_;
    std::string mlip_spec_file_;    // path to per-model YAML spec (full schema)

    // ── SHAKE/RATTLE constraints (CPU + velocity_verlet only; see
    // include/constraint_solver.h) ──────────────────────────────────────────
    std::unique_ptr<ConstraintSolver> constraint_solver_;
    bool constraints_enabled_ = false;
    std::vector<Real> constraint_pre_step_x_, constraint_pre_step_y_, constraint_pre_step_z_;

    // Configuration variables
    std::string coordinate_file;
    std::string topology_file;
    std::string parameter_file;
    Real target_temperature;
    std::string output_prefix;
    int trajectory_output_freq;
    int energy_output_freq;
    Real cutoff_distance;
    Real lj_switch_distance;
    int neighbor_list_update_freq;  // Added neighbor list update frequency
    Real verlet_skin;               // Verlet list skin distance
    Real neighbor_rebuild_hi_frac;  // rebuild trigger threshold as fraction of skin
    Real neighbor_rebuild_lo_frac;  // re-arm threshold as fraction of skin
    int neighbor_displacement_check_stride; // 0: auto (update_freq/4)
    int neighbor_list_builds;       // Count of neighbor list builds
    int active_graph_prunes_ = 0;   // incremental active-CSR rebuilds
    int active_graph_reuses_ = 0;   // steps that safely reused active CSR
    // Rebuild reason diagnostics (a single rebuild can increment multiple
    // buckets when conditions overlap).
    int rebuild_reason_periodic_ = 0;
    int rebuild_reason_adaptive_ = 0;
    int rebuild_reason_forced_ = 0;
    int rebuild_reason_mpi_sync_ = 0;
    int rebuild_reason_box_change_ = 0;
    int rebuild_reason_other_ = 0;
    // Comma-separated enabled components: builtin, python, libtorch.
    // Components are evaluated in that order and add to the same force buffer.
    std::string force_calculation_mode;
    bool enable_builtin_forces_ = true;
    bool enable_python_forces_ = false;
    bool enable_libtorch_forces_ = false;
    std::string force_calculation_type;
    std::string force_precision_ = "double";

    std::string nonbonded_kernel_ = "tiled";

    // Opt-in for the unverified Hopper (sm_90+) thread-block-cluster PME
    // spread kernel; see ForceCalculatorCUDA::set_pme_cluster_spread_enabled.
    // Ignored on any GPU below compute capability 9.0.
    bool pme_cluster_spread_ = false;

    // Long-range electrostatics solver: "pme" (default, FFT-based Ewald) or
    // "fmm" (Fast Multipole Method placeholder -- config surface only, not
    // implemented; rejected at parse time with a clear error). See the PME
    // reciprocal-space parallelization roadmap's stage 3.
    std::string electrostatics_solver_ = "pme";

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

    // ── Energy minimization (runs once, before the MD loop) ─────────────────
    std::string minimizer_type_ = "none";     // none | fire
    int minimizer_max_steps_ = 1000;
    Real minimizer_force_tolerance_ = 1.0;    // kcal/mol/Angstrom, global max |F_i|
    Real minimizer_dt_start_ = 0.0;           // fs; <=0 means "derive from timestep"
    Real minimizer_dt_max_ = 0.0;             // fs; <=0 means "derive from dt_start"
    int minimizer_output_freq_ = 10;          // print progress every N minimizer steps

    // Output control variables
    int output_freq;                    // General output frequency
    std::string output_info;            // What information to output (comma-separated)
    std::vector<OutputField> output_fields_cache;
    
    // Trajectory output settings
    std::string trajectory_format;      // raw config string, kept for logging
    std::set<std::string> trajectory_formats_;  // parsed: any of xyz, dump, dcd, bp, h5md
    bool trajectory_include_velocities; // Include velocities in output
    bool trajectory_include_forces;     // Include forces in output

    // ── DCD writer state (single continuously-appended file; CHARMM/NAMD
    // binary trajectory format) ─────────────────────────────────────────────
    std::ofstream dcd_file_;
    std::streampos dcd_nset_patch_pos_;  // byte offset of the NSET header field
    int dcd_frames_written_ = 0;

    // Global atom count, constant for the whole run (no particle creation/
    // destruction). Computed once in initialize_trajectory_writers() via a
    // single MPI_Allreduce and reused by the DCD NATOM record and the
    // ADIOS2 "natoms_global" attribute.
    int trajectory_natoms_global_cached_ = 0;

    // ── ADIOS2 BP writer state (genuinely parallel: every rank writes its
    // own local particles as an independent local-array block, tagged with
    // global_id so a reader can reassemble global order) ───────────────────
#ifdef USE_ADIOS2
    adios2::ADIOS adios2_instance_;
    adios2::IO adios2_io_;
    adios2::Engine adios2_engine_;
    adios2::Variable<int> adios2_var_global_id_;
    adios2::Variable<Real> adios2_var_x_, adios2_var_y_, adios2_var_z_;
    adios2::Variable<Real> adios2_var_vx_, adios2_var_vy_, adios2_var_vz_;
    adios2::Variable<Real> adios2_var_fx_, adios2_var_fy_, adios2_var_fz_;
    adios2::Variable<int> adios2_var_step_;
    adios2::Variable<Real> adios2_var_time_;
    adios2::Variable<Real> adios2_var_box_;
    bool adios2_active_ = false;
#endif

    // ── H5MD writer state (parallel HDF5: every rank writes its own local
    // particles as a hyperslab of a shared, collectively-extended dataset,
    // tagged with an "id" dataset so a reader can reassemble global order --
    // same no-gather design as the ADIOS2 writer above, on the H5MD schema) ─
#ifdef USE_HDF5
    hid_t h5md_file_ = -1;
    hid_t h5md_position_value_ = -1, h5md_position_step_ = -1, h5md_position_time_ = -1;
    hid_t h5md_velocity_value_ = -1, h5md_velocity_step_ = -1, h5md_velocity_time_ = -1;
    hid_t h5md_force_value_ = -1, h5md_force_step_ = -1, h5md_force_time_ = -1;
    hid_t h5md_id_value_ = -1;
    hid_t h5md_species_value_ = -1;
    hid_t h5md_box_edges_ = -1;
    int h5md_frame_count_ = 0;
    bool h5md_active_ = false;
#endif
    bool trajectory_writers_initialized_ = false;

    // Gathered, global_id-ordered snapshot used by the single-file serial
    // writers (xyz/dump/dcd). Populated on every rank by an MPI_Gatherv (the
    // collective must run on all ranks); only rank 0's contents are valid.
    struct SerialTrajectoryFrame {
        std::vector<Real> x, y, z;
        std::vector<Real> vx, vy, vz;
        std::vector<Real> fx, fy, fz;
        std::vector<int> type;
        std::vector<std::string> element;
    };

    int current_step;
    int total_steps;
    Real dt;
    std::string python_force_path;
    bool python_force_enabled;
    
    // Cached energy values
    Real cached_potential_energy;       // Store PE from force calculation
    
    // Critical simulation state flags (LAMMPS-style)
    bool neighbor_rebuild_forced;       // Force neighbor rebuild (e.g., after barostat)
    // Adaptive neighbor rebuild hysteresis state:
    // trigger when displacement exceeds hi threshold, re-arm only after
    // dropping below lo threshold to avoid threshold-chatter rebuilds.
    bool adaptive_rebuild_armed_ = true;
    
    // ── Timing infrastructure ────────────────────────────────────────────
    // Two independent opt-in knobs, both env-var driven (see run()):
    //   MD_TIMING_TRACE=step|every   -- print a per-step [TIMING] line
    //   MD_TIMING_TRACE_FORCE=1      -- also break "force" into its GPU
    //                                   components (bonded/lj/electrostatic/
    //                                   ...); folds into BOTH the per-step
    //                                   trace line (when tracing is on) and
    //                                   the always-on end-of-run report.
    // Replaces the older MD_TIMING / MD_TIMING_EVERY_STEP /
    // MD_TIMING_FORCE_BREAKDOWN env vars from earlier in this codebase's
    // history -- those names are no longer read.
    enum class TimingTraceMode { Off, Step, Every };
    TimingTraceMode timing_trace_mode_ = TimingTraceMode::Off;
    bool timing_trace_force_ = false;
    static TimingTraceMode parse_timing_trace_mode();

    struct StepTiming {
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
        // Only populated when timing_trace_force_ is set; empty otherwise.
        std::vector<std::pair<std::string, double>> force_components_ms;
    } last_timing;

    // Cumulative timing statistics, printed as a tree by print_timing_report()
    // (src/timing_breakdown.cpp) at the end of the run. total_time is the sum
    // of per-step wall times (i.e. the MD loop's own time, "Loop time" in
    // LAMMPS terms); the whole-program wall clock (which also covers
    // init_time, minimize_time, and the negligible finalize() cost itself)
    // is passed into finalize() by main.cpp rather than re-measured here.
    struct CumulativeTiming {
        double init_time = 0.0;         // MDSimulation::initialize()
        double minimize_time = 0.0;     // run_energy_minimization() (FIRE)
        double neighbor_time = 0.0;     // Neighbor list rebuild/build
        double force_time = 0.0;        // Force calculation (all components)
        double integration_time = 0.0;  // Integrate1 + Integrate2
        double thermostat_time = 0.0;   // Thermostat apply
        double barostat_time = 0.0;     // Barostat apply (+ rebuild trigger handling)
        double ke_time = 0.0;           // Kinetic energy refresh for outputs
        double comm_time = 0.0;         // Halo communication/exchange
        double migration_time = 0.0;    // Escaped-particle migration
        double output_time = 0.0;       // Output/IO operations
        double total_time = 0.0;        // MD loop wall time (sum of step totals)
        int step_count = 0;             // Number of steps timed
        // Accumulated across steps; only non-empty when timing_trace_force_.
        std::vector<std::pair<std::string, double>> force_component_time;
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
    // Prints the end-of-run timing tree (src/timing_breakdown.cpp): init,
    // minimize, md_loop and its per-phase (and, with MD_TIMING_TRACE_FORCE=1,
    // per-force-component) buckets. program_wallclock_ms is main.cpp's own
    // measured span covering the whole program (the only place that can see
    // time spent before this object existed); passing < 0 just omits the
    // "total (wall clock)"/"unaccounted" rows that compare against it.
    void print_timing_report(double program_wallclock_ms) const;

    // The actual body of initialize(); wrapped by the public entry point
    // below purely so its wall-clock time can be recorded into
    // cumulative_timing.init_time for the report above.
    void initialize_impl(const std::string& config_file);

public:
    MDSimulation();
    ~MDSimulation();

    void initialize(const std::string& config_file);
    void run();
    void finalize(double program_wallclock_ms = -1.0);
    
    void single_step();
    void output_results(int step, bool emit_text = true);
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
    void zero_velocities();             // Zero all velocities (local + ghost particles)

    // ── Energy minimization (FIRE) ──────────────────────────────────────────
    void run_energy_minimization();
    void compute_fire_reduction_cpu(Real& power, Real& f_norm_sq,
                                    Real& v_norm_sq, Real& f_max_sq) const;
    void apply_fire_update_cpu(const FireMinimizer::StepParams& params);

    // Trajectory output functions. write_trajectory() is collective (called
    // by every rank, via output_results()) because gathering for the serial
    // formats and the ADIOS2 BP writer both require every rank to
    // participate; only the per-format file writers below are rank-0-only.
    void write_trajectory(int step);
    void gather_particles_for_serial_output(SerialTrajectoryFrame& frame) const;
    void write_xyz_trajectory(int step, const SerialTrajectoryFrame& frame);
    void write_lammps_dump(int step, const SerialTrajectoryFrame& frame);
    void write_dcd_frame(int step, const SerialTrajectoryFrame& frame);
    void initialize_trajectory_writers();
    void finalize_trajectory_writers();
    void write_dcd_header(int32_t natom_global);

#ifdef USE_ADIOS2
    void initialize_adios2_writer();
    void write_adios2_frame(int step);
    void finalize_adios2_writer();
#endif

#ifdef USE_HDF5
    void initialize_h5md_writer();
    void write_h5md_frame(int step);
    void finalize_h5md_writer();
#endif
};

#endif // MD_SIMULATION_H
