#include "md_simulation.h"
#include "neighbor_list.h"
#include "force_calculator_cuda.h"
#include "constraint_solver.h"
#include "thermostat.h"
#include "barostat.h"
#include "halo_backend.h"
#include "comm_backend.h"
#include "simulation_backend_factory.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <array>
#include <set>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

// ── C++ MLIP Runtime (mlip_inst.md architecture) ────────────────────────────
#include "force_libtorch/runtime/mlip_runtime.h"
#include "force_libtorch/mlip_types.h"
// MlipExecutionContext is in the global namespace (defined in mlip_types.h)
// MlipRuntime is in namespace mlip
using namespace mlip;

namespace {
template <typename F>
double measure_seconds(F&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr) && (std::string(value) == "1");
}

// Writes one Fortran-unformatted record (4-byte length, payload, matching
// 4-byte length trailer) -- the wrapping every field/array uses in the
// classic CHARMM/NAMD DCD binary trajectory format.
void dcd_write_record(std::ofstream& f, const char* data, int32_t nbytes) {
    f.write(reinterpret_cast<const char*>(&nbytes), sizeof(int32_t));
    f.write(data, nbytes);
    f.write(reinterpret_cast<const char*>(&nbytes), sizeof(int32_t));
}

std::string trim_copy(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

std::set<std::string> parse_force_components(const std::string& value) {
    std::set<std::string> result;
    std::stringstream stream(value);
    std::string component;
    while (std::getline(stream, component, ',')) {
        component = trim_copy(component);
        if (!component.empty()) result.insert(component);
    }
    return result;
}

std::string uppercase_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

std::map<std::string, std::string> read_dsmd_input_manifest(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open dsMD_input manifest: " + filename);
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim_copy(line.substr(0, colon));
        std::string value = trim_copy(line.substr(colon + 1));
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        if (!key.empty() && !value.empty()) values[key] = value;
    }
    if (values["schema_version"] != "1")
        throw std::runtime_error("dsMD_input manifest schema_version must be 1");
    if (values["format"] != "dsMD_input")
        throw std::runtime_error("dsMD_input manifest format must be dsMD_input");
    return values;
}
}  // namespace


MDSimulation::MDSimulation() : 
    cuda_calc_cached_(nullptr),
    using_cuda_backend_cached_(false),
    mlip_spec_file_(""),
    coordinate_file(""),
    topology_file(""),
    parameter_file(""),
    target_temperature(0.0),
    output_prefix("md_output"),
    trajectory_output_freq(100),
    energy_output_freq(10),
    cutoff_distance(12.0),
    lj_switch_distance(-1.0),
    neighbor_list_update_freq(10),  // Default update frequency: every 10 steps
    verlet_skin(2.0),               // Default Verlet skin distance
    neighbor_rebuild_hi_frac(0.5),
    neighbor_rebuild_lo_frac(0.4),
    neighbor_displacement_check_stride(0),
    neighbor_list_builds(0),        // Initialize counter
    force_calculation_mode("builtin"),  // Default: use builtin force calculations
    force_calculation_type(""),         // Default: no specific ML type
    mlip_model_path(""),
    mlip_model_type(""),
    mlip_model_size("small"),
    mlip_device("cpu"),
    mlip_dtype("float32"),
    integrator_type("verlet"),
    thermostat_type("none"),
    tau_t(0.1),
    barostat_type("none"),
    target_pressure(1.0),
    tau_p(1.0),
    output_freq(10),                     // Default: every 10 steps
    output_info("STEP, T, KE"),          // Default output info
    trajectory_format("xyz"),            // Default: XYZ format
    trajectory_include_velocities(false), // Default: no velocities
    trajectory_include_forces(false),    // Default: no forces
    current_step(0), 
    total_steps(0), 
    dt(0.001), 
    python_force_path("python_force"), 
    python_force_enabled(false),
    cached_potential_energy(0.0),
    neighbor_rebuild_forced(false),
    adaptive_rebuild_armed_(true) {
    system = std::make_unique<MDSystem>();
}

MDSimulation::~MDSimulation() = default;

void MDSimulation::rebuild_output_info_items_cache() {
    output_fields_cache.clear();
    std::vector<std::string> unknown_tokens;

    std::stringstream ss(output_info);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::string token = uppercase_copy(trim_copy(item));
        if (token.empty()) {
            continue;
        }

        if (token == "STEP") output_fields_cache.push_back(OutputField::STEP);
        else if (token == "T") output_fields_cache.push_back(OutputField::T);
        else if (token == "KE") output_fields_cache.push_back(OutputField::KE);
        else if (token == "PE") output_fields_cache.push_back(OutputField::PE);
        else if (token == "ETOTAL") output_fields_cache.push_back(OutputField::ETOTAL);
        else if (token == "P" || token == "PRESS") output_fields_cache.push_back(OutputField::PRESS);
        else if (token == "MAX_SPEED") output_fields_cache.push_back(OutputField::MAX_SPEED);
        else if (token == "AVG_SPEED") output_fields_cache.push_back(OutputField::AVG_SPEED);
        else if (token == "FMAX") output_fields_cache.push_back(OutputField::FMAX);
        else unknown_tokens.push_back(token);
    }

    if (system && system->get_rank() == 0) {
        for (const std::string& token : unknown_tokens) {
            std::cerr << "Warning: unknown output_info token '" << token
                      << "' will be ignored." << std::endl;
        }

        if (output_fields_cache.empty()) {
            std::cerr << "Warning: output_info has no valid tokens; falling back to STEP,T,KE." << std::endl;
        }
    }
}

void MDSimulation::initialize(const std::string& config_file) {
    const double init_time = measure_seconds([&]() {
        initialize_impl(config_file);
    });
    cumulative_timing.init_time += init_time;
}

void MDSimulation::initialize_impl(const std::string& config_file) {
    // Load configuration
    system->load_config(config_file);
    
    // Parse simulation parameters from config file
    parse_simulation_config(config_file);

    // Load coordinate file from configuration first
    system->load_coordinates(coordinate_file);
    
    // Load topology file from configuration and assign properties to particles
    // MUST be done BEFORE velocity initialization to ensure masses are set
    if (!topology_file.empty()) {
        system->load_topology(topology_file, parameter_file);
        // Set up force field parameters after topology is loaded
        system->setup_force_field_params();
        
        if (system->get_rank() == 0) {
            std::cout << "Topology and force field parameters loaded" << std::endl;
        }
    }
    
    // Initialize velocities with Maxwell-Boltzmann distribution using target temperature
    // CRITICAL: Must be done BEFORE domain decomposition so all particles get velocities
    system->initialize_velocities(target_temperature);
    
    // Set up domain decomposition after particles are loaded and velocities initialized
    system->setup_domain_decomposition();

    // Build-dependent backend choice is centralized in simulation_backend_factory.
    force_calc = create_force_calculator_for_build(system->get_rank());
    force_calc->set_nonbonded_kernel(nonbonded_kernel_);
    if (system->get_rank() == 0) {
        std::cout << "[Nonbonded kernel] loaded force_calculation.nonbonded_kernel = "
                  << nonbonded_kernel_ << std::endl;
        if (nonbonded_kernel_ == "graph") {
            std::cout << "[Nonbonded kernel] graph dispatch enabled: outer CSR(rc+skin)"
                      << " -> displacement-triggered active directed CSR(rc+skin/2)"
                      << " -> exact rc force check"
                      << " (legacy tile graph disabled)"
                      << std::endl;
        } else {
            std::cout << "[Nonbonded kernel] tiled dispatch enabled (legacy path)"
                      << std::endl;
        }
    }
    cuda_calc_cached_ = dynamic_cast<ForceCalculatorCUDA*>(force_calc.get());
    using_cuda_backend_cached_ = (cuda_calc_cached_ != nullptr);

    halo_backend_ = create_halo_backend(force_calculation_mode, force_calc.get());
    comm_backend_ = create_comm_backend();
#ifdef USE_CUDA
    if (cuda_calc_cached_) {
        cuda_calc_cached_->set_builtin_forces_enabled(enable_builtin_forces_);
        cuda_calc_cached_->set_mixed_precision(force_precision_ == "mixed");
        if (force_precision_ == "mixed" && system->get_rank() == 0) {
            std::cout << "[Force precision] mixed precision enabled: LJ/Ewald "
                      << "real-space pair math runs in float, forces/energy/"
                      << "virial still accumulate in double" << std::endl;
        }
        cuda_calc_cached_->set_pme_cluster_spread_enabled(pme_cluster_spread_);
        if (pme_cluster_spread_ && system->get_rank() == 0) {
            std::cout << "[PME] thread-block-cluster spread kernel requested -- "
                      << "unverified on real Hopper hardware; only takes "
                      << "effect if this rank's GPU is compute capability "
                      << ">= 9.0, otherwise the usual kernel runs."
                      << std::endl;
        }
    }
#endif
    // CPU backend: force_precision_ previously only reached the CUDA branch
    // above -- "precision: mixed" in the YAML had no effect at all on a CPU
    // run. set_mixed_precision() prints its own confirmation (including the
    // AVX2-build fallback warning), so nothing else is needed here. Only
    // call for "mixed" (never explicitly pass false) so a config left at the
    // "double" default doesn't clobber the MD_NONBONDED_PRECISION env-var
    // escape hatch ForceCalculatorCPU also supports.
    if (!cuda_calc_cached_ && force_precision_ == "mixed") {
        force_calc->set_mixed_precision(true);
    }

    // Python is a separate additive component; it does not replace built-in
    // or LibTorch forces.
    if (enable_python_forces_) {
        python_interface = std::make_unique<PythonForceInterface>();
        python_interface->initialize_python(python_force_path);
        if (system->get_rank() == 0) {
            std::cout << "Python force calculation interface initialized" << std::endl;
        }

        // Initialize a model through the model-agnostic Python MLIP adapter.
        // Auto-upgrade device to CUDA when compiled with GPU support and not explicitly set to CPU.
        {
            if (is_using_cuda_backend() && mlip_device != "cpu") {
                mlip_device = "cuda";
            }
            if (mlip_dtype.empty()) mlip_dtype = "float32";
            const std::string effective_model_type =
                !mlip_model_type.empty() ? mlip_model_type : force_calculation_type;
            if (system->get_rank() == 0) {
                if (effective_model_type.empty()) {
                    std::cout << "Using Python force provider without MLIP model initialization" << std::endl;
                } else {
                    std::cout << "Initializing Python MLIP (type=" << effective_model_type << ")..." << std::endl;
                    std::cout << "  model_path: " << mlip_model_path << std::endl;
                    std::cout << "  model_type: " << effective_model_type << std::endl;
                    std::cout << "  model_size: " << mlip_model_size << std::endl;
                    std::cout << "  device:     " << mlip_device << std::endl;
                    std::cout << "  dtype:      " << mlip_dtype << std::endl;
                }
            }
            if (!effective_model_type.empty()) {
                python_interface->initialize_ml_model(
                    mlip_model_path, effective_model_type, mlip_model_size,
                    mlip_device, mlip_dtype, mlip_extra_kwargs);
            }
        }
    }

    // ── C++ LibTorch MLIP path ─────────────────────────────────────────────
    // Activated when force_calculation.mode contains "libtorch".
    // Reads mlip_spec_file_ (a full mlip_inst.md schema YAML) and wires
    // MlipRuntime into ForceCalculatorCUDA for GPU-resident execution.
    if (enable_libtorch_forces_) {
#ifndef USE_CUDA
        throw std::runtime_error(
            "[LibTorch] libtorch force mode requires a CUDA build (USE_CUDA). "
            "Current build does not enable CUDA.");
#else
        if (!is_using_cuda_backend()) {
            throw std::runtime_error(
                "[LibTorch] libtorch force mode requires an active CUDA backend. "
                "Current run is not using ForceCalculatorCUDA.");
        }

        if (mlip_spec_file_.empty()) {
            throw std::runtime_error(
                "[LibTorch] force_calculation.mode = libtorch requires "
                "'mlip_spec_file' key in the MD YAML pointing to a model spec file.");
        }

        if (system->get_rank() == 0) {
            std::cout << "[MlipRuntime] Initializing C++ MLIP runtime...\n"
                      << "  spec: " << mlip_spec_file_ << "\n";
        }

        MlipExecutionContext ctx;
        ctx.mpi_rank  = system->get_rank();
        ctx.mpi_size  = system->get_size();
        ctx.device_id = 0;   // TODO: multi-GPU rank mapping

        mlip_runtime_ = std::make_unique<mlip::MlipRuntime>();

        // Register MD device pointer sources into global SourceRegistry BEFORE
        // MlipRuntime::initialize() validates that source names exist.
        {
            auto* fc_cuda = cuda_calc_cached_;
            if (fc_cuda) fc_cuda->pre_register_mlip_sources();
        }

        mlip_runtime_->initialize(mlip_spec_file_, ctx);

        if (system->get_rank() == 0) {
            std::cout << "[MlipRuntime] ✅ Runtime ready. cutoff="
                      << mlip_runtime_->cutoff() << " Å\n";
        }

        // Wire MlipRuntime into ForceCalculatorCUDA
        auto* fc_cuda = cuda_calc_cached_;
        if (fc_cuda) {
            fc_cuda->set_mlip_runtime(static_cast<void*>(mlip_runtime_.get()));
        }
#endif
    }
    
    // Initialize the neighbor list. cpu_spatial_verlet (CpuSpatialVerletList
    // in neighbor_list.h) is the only backend; there is no config knob to
    // select anything else any more.
    Real cutoff = cutoff_distance;
    if (system->get_rank() == 0) {
        std::cout << "Using cpu_spatial_verlet neighbor list with skin distance: "
                  << verlet_skin << std::endl;
    }

    neighbor_list_ = create_neighbor_list(NeighborListBackend::CPU_SPATIAL_VERLET, cutoff, verlet_skin);
    neighbor_list_->set_nonbonded_kernel(nonbonded_kernel_);
    // Only the LibTorch MLIP GNN edge-list builder reads the atom-pair CSR
    // outside of graph-kernel mode (see set_mlip_edge_list_needed()'s doc
    // comment) -- tell the neighbor list so it can skip building that CSR
    // entirely in the common tiled+builtin-forces configuration.
    neighbor_list_->set_mlip_edge_list_needed(enable_libtorch_forces_);
    system->set_neighbor_list(neighbor_list_.get());

    // Initialize components based on configuration
    initialize_integrator();
    initialize_thermostat();
    initialize_barostat();

    // ── Constraints (SHAKE/RATTLE) ──────────────────────────────────────────
    // velocity_verlet only. CPU uses ConstraintSolver; CUDA uses the
    // GPU-resident kernels in ForceCalculatorCUDA (apply_shake_on_device /
    // apply_rattle_on_device). See include/constraint_solver.h for the
    // derivation both paths implement.
    if (!system->get_topology().get_constraints().empty()) {
        bool is_velocity_verlet = dynamic_cast<VelocityVerletIntegrator*>(integrator.get()) != nullptr;
#ifdef USE_CUDA
        is_velocity_verlet = is_velocity_verlet ||
            dynamic_cast<VelocityVerletIntegratorCUDA*>(integrator.get()) != nullptr;
#endif
        if (!is_velocity_verlet) {
            throw std::runtime_error(
                "[Constraints] SHAKE/RATTLE are only implemented for the velocity_verlet integrator.");
        }
        constraints_enabled_ = true;
        if (!cuda_calc_cached_) {
            constraint_solver_ = std::make_unique<ConstraintSolver>();
        }
        if (system->get_rank() == 0) {
            std::cout << "[Constraints] SHAKE/RATTLE enabled ("
                      << (cuda_calc_cached_ ? "CUDA" : "CPU") << ") for "
                      << system->get_topology().get_constraints().size()
                      << " constraint(s)" << std::endl;
            if (system->get_size() > 1) {
                std::cout << "[Constraints] MPI mode: cross-rank constraints "
                          << "(one endpoint on a ghost slot) are skipped by "
                          << "the CPU solver to avoid inconsistent owner/ghost "
                          << "state updates." << std::endl;
            }
        }
    }

    // Collective: opens the DCD/ADIOS2 files (if requested) now that domain
    // decomposition has established local particle counts.
    initialize_trajectory_writers();
}

bool MDSimulation::is_using_cuda_backend() const {
    return using_cuda_backend_cached_;
}

void MDSimulation::single_step() {
    // Per-phase timing always accumulates (cheap chrono calls); whether it's
    // ever printed depends on MD_TIMING_TRACE (see run()).
    auto step_start = std::chrono::high_resolution_clock::now();
    
    // Check if using CUDA (GPU-resident mode)
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    // ==================== Step 1: Position Integration ====================
    double integrate1_time = 0.0;
    handle_integrate1_phase(cuda_calc, integrate1_time);
    
    double comm_time = 0.0;   // Track halo communication time
    double migrate_time = 0.0; // Track migration check/exchange time
    
    // ==================== Neighbor List Management ====================
    double nlist_build_time = 0.0;
    handle_neighbor_phase(nlist_build_time, comm_time, migrate_time);
    
    // ==================== Force Calculation at t+dt ====================
    double force_time = 0.0;
    handle_force_phase(cuda_calc, force_time);

    // ==================== Step 3: Complete velocity update ====================
    double integrate2_time = 0.0;
    handle_integrate2_phase(cuda_calc, integrate2_time);
    
    // ==================== Apply thermostat and barostat ====================
    double thermostat_time = 0.0;
    double ke_time = 0.0;
    double barostat_time = 0.0;
    double modify_time = 0.0;
    handle_modify_phase(
        cuda_calc,
        thermostat_time, ke_time,
        barostat_time, modify_time);
    
    auto step_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(step_end - step_start).count();

    accumulate_step_timing(
        integrate1_time, nlist_build_time, comm_time,
        migrate_time,
        force_time, integrate2_time,
        thermostat_time, ke_time,
        barostat_time, modify_time,
        total_time);
        
    current_step++;
}

void MDSimulation::prepare_first_step() {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    // Special initialization for Leapfrog integrator
    if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        integrator->initialize_velocities(system->get_particles(), dt, system->get_local_particles());
    }

    // Ensure GPU buffers are initialized before any device-side KE/output path is used.
    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->initial_upload_to_device(*system);
#endif
    }

    // Velocity Verlet requires F(r(t)) before its first half-kick.  Building
    // only the device buffers/kinetic energy here leaves the initial force at
    // zero, reports a false PE=0 at step 0, and advances the first step with
    // the wrong acceleration.
    double neighbor_time = 0.0;
    double comm_time = 0.0;
    double migrate_time = 0.0;
    handle_neighbor_phase(neighbor_time, comm_time, migrate_time);

    double force_time = 0.0;
    handle_force_phase(cuda_calc, force_time);
    
    // Calculate initial kinetic energy after velocity initialization
    // GPU-resident mode: KE calculated on GPU
    if (force_calc) {
        force_calc->calculate_kinetic_energy(*system);
    }

}

void MDSimulation::handle_integrate1_phase(ForceCalculatorCUDA* cuda_calc, double& integrate1_time) {
    integrate1_time = measure_seconds([&]() {
        if (cuda_calc) {
#ifdef USE_CUDA
            if (constraints_enabled_) {
                cuda_calc->snapshot_constraint_positions(*system);
            }
            cuda_calc->integrate_on_device(*system, dt, integrator_type);
            if (constraints_enabled_) {
                if (!cuda_calc->apply_shake_on_device(*system, dt)) {
                    throw std::runtime_error(
                        "[Constraints] CUDA SHAKE did not converge; reduce the timestep or check the constraint geometry.");
                }
            }
            return;
#else
#endif
        }
        if (constraints_enabled_) {
            ParticleData& particles = system->get_particles();
            // A constraint's non-owning atom can be a ghost slot (index >=
            // local_particles), exactly like bonds -- size/copy the FULL
            // SoA (locals+ghosts), not just the local prefix, or apply_shake
            // reads out of bounds for any constraint spanning into a ghost.
            const size_t n = particles.x.size();
            constraint_pre_step_x_.resize(n);
            constraint_pre_step_y_.resize(n);
            constraint_pre_step_z_.resize(n);
            std::copy(particles.x.begin(), particles.x.begin() + n, constraint_pre_step_x_.begin());
            std::copy(particles.y.begin(), particles.y.begin() + n, constraint_pre_step_y_.begin());
            std::copy(particles.z.begin(), particles.z.begin() + n, constraint_pre_step_z_.begin());
        }
        integrator->integrate(system->get_particles(), system->get_box(), dt, system->get_local_particles());
        if (constraints_enabled_) {
            if (!constraint_solver_->apply_shake(*system, dt,
                    constraint_pre_step_x_, constraint_pre_step_y_, constraint_pre_step_z_)) {
                throw std::runtime_error(
                    "[Constraints] CPU SHAKE did not converge; reduce the timestep or check the constraint geometry.");
            }
        }
    });
}

void MDSimulation::handle_neighbor_phase(double& nlist_build_time, double& comm_time, double& migrate_time) {
    // prepare_first_step() performs build zero. Avoid rebuilding the same list
    // immediately in the first call to single_step() while current_step is
    // still zero.
    const bool periodic_rebuild_due = (neighbor_list_builds == 0) ||
        (current_step > 0 && current_step % neighbor_list_update_freq == 0);
    bool should_rebuild = periodic_rebuild_due;
    bool reason_periodic = periodic_rebuild_due;
    bool reason_adaptive = false;
    bool reason_forced = false;
    bool reason_mpi_sync = false;
    migrate_time = 0.0;

    if (periodic_rebuild_due) {
        adaptive_rebuild_armed_ = true;
    }

    // Adaptive rebuild gate: when periodic rebuild is not due, rebuild only
    // if particles moved more than skin/2 since the last neighbor build.
    // To avoid per-step D2H synchronization overhead on CUDA, only sample
    // displacement every few steps between periodic full rebuild points.
    if (!should_rebuild && neighbor_list_builds > 0) {
        const int displacement_check_stride =
            (neighbor_displacement_check_stride > 0)
                ? neighbor_displacement_check_stride
                : std::max(1, neighbor_list_update_freq / 4);
        const bool run_displacement_check =
            (current_step > 0) && (current_step % displacement_check_stride == 0);

        if (!run_displacement_check) {
            should_rebuild = false;
        } else {
        const Real skin = (verlet_skin > 0.0) ? verlet_skin : 2.0;
        const Real hi_frac = neighbor_rebuild_hi_frac;
        const Real lo_frac = neighbor_rebuild_lo_frac;
        const Real disp_hi_sq = (skin * hi_frac) * (skin * hi_frac);
        const Real disp_lo_sq = (skin * lo_frac) * (skin * lo_frac);

#ifdef USE_CUDA
        const bool using_cuda_backend = (cuda_calc_cached_ != nullptr);
        if (using_cuda_backend) {
            const Real max_disp_sq = cuda_calc_cached_->compute_max_displacement_on_device(
                system->get_local_particles(), system->get_box());
            if (max_disp_sq <= disp_lo_sq) {
                adaptive_rebuild_armed_ = true;
            }
            should_rebuild = adaptive_rebuild_armed_ && (max_disp_sq > disp_hi_sq);
            if (should_rebuild) {
                adaptive_rebuild_armed_ = false;
                reason_adaptive = true;
            }
        } else
#endif
        if (!should_rebuild) {
            // Mirror the CUDA branch above exactly (same disp_hi_sq/disp_lo_sq
            // band, same adaptive_rebuild_armed_ state): rearm only once
            // displacement drops back below the LOW threshold, not merely
            // below the high one. The previous version called
            // needs_rebuild(), which only knows a single hardcoded 0.5*skin
            // threshold with no low band of its own -- the outer wrapper's
            // "rearm whenever not displaced" then re-armed as soon as
            // displacement dipped just under that one threshold, giving CPU
            // a narrower/noisier rebuild band than CUDA's and triggering
            // extra rebuilds when a few atoms hover near the threshold
            // (e.g. from residual steric clashes after a loose minimize).
            Real local_max_disp_sq = neighbor_list_->compute_local_max_displacement_sq(
                system->get_particles(), system->get_box(), system->get_local_particles());
            // A negative return is this class's sentinel for "owned-atom set
            // changed, treat as unconditional rebuild" (see
            // CpuSpatialVerletList::compute_local_max_displacement_sq) --
            // promote it to +inf so MPI_MAX can't let another rank's smaller,
            // ordinary displacement mask it.
            if (local_max_disp_sq < 0.0) {
                local_max_disp_sq = std::numeric_limits<Real>::max();
            }
            Real max_disp_sq = local_max_disp_sq;
            int mpi_init = 0;
            MPI_Initialized(&mpi_init);
            if (mpi_init) {
                MPI_Allreduce(&local_max_disp_sq, &max_disp_sq, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            }
            if (max_disp_sq <= disp_lo_sq) {
                adaptive_rebuild_armed_ = true;
            }
            should_rebuild = adaptive_rebuild_armed_ && (max_disp_sq > disp_hi_sq);
            if (should_rebuild) {
                adaptive_rebuild_armed_ = false;
                reason_adaptive = true;
            }
        }
        }
    }

    if (neighbor_rebuild_forced) {
        should_rebuild = true;
        reason_forced = true;
        adaptive_rebuild_armed_ = true;
        neighbor_rebuild_forced = false;
    }

    // GPU-resident single-rank optimisation: skip host-side halo handling.
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    bool gpu_single_rank = comm_backend_->is_gpu_single_rank(cuda_calc != nullptr);
    if (gpu_single_rank) {
        comm_time = 0.0;
    } else {
        auto comm_start = std::chrono::high_resolution_clock::now();

        bool gpu_direct_halo_done = false;
        if (halo_backend_) {
            gpu_direct_halo_done = halo_backend_->try_gpu_direct_halo(
                *system, force_calc.get(), should_rebuild);
        }

        if (cuda_calc && !gpu_direct_halo_done) {
            throw std::runtime_error(
                "CUDA execution requires the dsDecompX GPU-resident halo backend; "
                "host ParticleData fallback is disabled");
        }

        if (!gpu_direct_halo_done) {
            const bool rebuild_before_sync = should_rebuild;
            should_rebuild = system->sync_distributed_particles(should_rebuild);
            if (!rebuild_before_sync && should_rebuild) {
                reason_mpi_sync = true;
            }
        }

        if (halo_backend_) {
            halo_backend_->on_post_halo(
                *system, force_calc.get(), should_rebuild, gpu_direct_halo_done);
        }

#ifdef USE_CUDA
        if (cuda_calc) {
            const auto& cfg = system->get_force_manager().get_config();
            if (cfg.enable_bonds || cfg.enable_angles || cfg.enable_dihedrals) {
                cuda_calc->refresh_bonded_topology_on_device(*system);
            }
        }
#endif

        auto comm_end = std::chrono::high_resolution_clock::now();
        comm_time = std::chrono::duration<double>(comm_end - comm_start).count();
    }

    nlist_build_time = measure_seconds([&]() {
        if (should_rebuild) {
            if (reason_periodic) ++rebuild_reason_periodic_;
            if (reason_adaptive) ++rebuild_reason_adaptive_;
            if (reason_forced) ++rebuild_reason_forced_;
            if (reason_mpi_sync) ++rebuild_reason_mpi_sync_;
            if (!reason_periodic && !reason_adaptive && !reason_forced && !reason_mpi_sync) {
                ++rebuild_reason_other_;
            }
        }
        handle_neighbor_rebuild_if_needed(should_rebuild);
        // The graph backend keeps a long-lived rc+skin outer list but
        // compacts it against current coordinates every step.  This is
        // deliberately independent of the outer-list rebuild cadence.
        if (nonbonded_kernel_ == "graph" && !cuda_calc_cached_ && neighbor_list_builds > 0) {
            const bool prune_due = neighbor_list_->needs_active_prune(
                system->get_particles(), system->get_box(),
                system->get_local_particles());
            if (prune_due) {
                neighbor_list_->prune_active(system->get_particles(), system->get_box(),
                                             system->get_local_particles());
                ++active_graph_prunes_;
            } else {
                ++active_graph_reuses_;
            }
        }
    });
}

void MDSimulation::compute_forces_with_mode(bool download_positions_for_ml) {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    // The calculator clears the force buffer once, runs enabled built-in terms,
    // then (if attached) accumulates the LibTorch MLIP contribution.
    // CUDA calculator must also run for Python-only mode to clear its device
    // force buffer; its built-in kernels have been disabled at initialization.
    if (force_calc && (enable_builtin_forces_ || enable_libtorch_forces_ || cuda_calc)) {
        force_calc->compute_forces(*system);
    } else {
        zero_forces();
    }

    // Python's additive-force ABI returns forces only (no energy), so its PE
    // contribution is pulled separately via get_last_energy() and folded into
    // cached_potential_energy below -- otherwise a pure-python run would
    // report whatever force_calc's (disabled) builtin/LibTorch PE happens to
    // hold, typically zero, instead of the MLIP model's actual energy.
    Real python_potential_energy = 0.0;

    if (enable_python_forces_ && python_interface) {
        SimulationBox& box = system->get_box();

        if (download_positions_for_ml && cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->download_positions_for_ml(
                system->get_particles(), system->get_num_particles());
            cuda_calc->download_forces_for_ml(
                system->get_particles(), system->get_num_particles());
#endif
        }

        python_interface->compute_force(
            system->get_particles(), system->get_ff_params(),
            system->get_neighbor_list(),
            box.lx, box.ly, box.lz,
            box.periodic_x, box.periodic_y, box.periodic_z);
        python_potential_energy = static_cast<Real>(python_interface->get_last_energy());

        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->upload_forces_from_ml(
                system->get_particles(), system->get_num_particles());
#endif
        }
    }

    cached_potential_energy =
        (force_calc ? force_calc->get_potential_energy() : 0.0) + python_potential_energy;
}

void MDSimulation::reverse_halo_for_bonded_if_needed() {
    if (!comm_backend_->is_distributed()) {
        return;
    }

    const auto& cfg = system->get_force_manager().get_config();
    const bool has_bonded = cfg.enable_bonds || cfg.enable_angles || cfg.enable_dihedrals;
    const bool has_special_pairs =
        (cfg.enable_lennard_jones || cfg.enable_coulomb) &&
        !system->get_local_special_pairs().empty();

    // Reverse halo is required for any force term that writes onto ghost slots
    // and expects those contributions to be accumulated back to owners.
    // Bonded terms do this, and special_pairs currently does as well.
    if (!has_bonded && !has_special_pairs) {
        return;
    }

#ifdef USE_CUDA
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    if (cuda_calc) {
        system->reverse_halo_on_device(*cuda_calc);
        return;
    }
#endif
    system->perform_reverse_halo_exchange();
}

void MDSimulation::handle_neighbor_rebuild_if_needed(bool should_rebuild) {
    if (!should_rebuild) {
        return;
    }

    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    if (cuda_calc) {
#ifdef USE_CUDA
        // CUDA builds are GPU-resident by contract. Do not silently download
        // particles and rebuild on the host if the device builder fails.
        cuda_calc->build_neighbor_list_on_device(*system);
#else
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
#endif
    } else {
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
    }

    neighbor_list_builds++;

}

void MDSimulation::handle_force_phase(ForceCalculatorCUDA* cuda_calc, double& force_time) {
    force_time = measure_seconds([&]() {
        if (!cuda_calc) {
            zero_forces();
        }

        compute_forces_with_mode(true);
        reverse_halo_for_bonded_if_needed();
    });
}

void MDSimulation::handle_integrate2_phase(ForceCalculatorCUDA* cuda_calc, double& integrate2_time) {
    integrate2_time = measure_seconds([&]() {
        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->complete_velocity_update_on_device(*system, dt, integrator_type);
            if (constraints_enabled_) {
                if (!cuda_calc->apply_rattle_on_device(*system)) {
                    throw std::runtime_error(
                        "[Constraints] CUDA RATTLE did not converge; reduce the timestep or check the constraint geometry.");
                }
            }
            return;
#else
#endif
        }
        integrator->complete_velocity_update(system->get_particles(), dt, system->get_local_particles());
        if (constraints_enabled_) {
            if (!constraint_solver_->apply_rattle(*system)) {
                throw std::runtime_error(
                    "[Constraints] CPU RATTLE did not converge; reduce the timestep or check the constraint geometry.");
            }
        }
    });
}

void MDSimulation::accumulate_step_timing(double integrate1_time, double nlist_build_time, double comm_time,
                                          double migrate_time,
                                          double force_time, double integrate2_time,
                                          double thermostat_time, double ke_time,
                                          double barostat_time, double modify_time,
                                          double total_time) {
    // Keep per-step phase timing in milliseconds for optional diagnostics.
    last_timing.nlist_build_ms = nlist_build_time * 1000.0;
    last_timing.comm_ms = comm_time * 1000.0;
    last_timing.migrate_ms = migrate_time * 1000.0;
    last_timing.force_ms = force_time * 1000.0;
    last_timing.integrate_ms = (integrate1_time + integrate2_time) * 1000.0;
    last_timing.thermostat_ms = thermostat_time * 1000.0;
    last_timing.ke_ms = ke_time * 1000.0;
    last_timing.barostat_ms = barostat_time * 1000.0;
    last_timing.total_ms = total_time * 1000.0;
    last_timing.force_components_ms.clear();

    cumulative_timing.neighbor_time += nlist_build_time;
    cumulative_timing.force_time += force_time;
    cumulative_timing.integration_time += (integrate1_time + integrate2_time);
    cumulative_timing.thermostat_time += thermostat_time;
    cumulative_timing.ke_time += ke_time;
    cumulative_timing.barostat_time += barostat_time;
    cumulative_timing.comm_time += comm_time;
    cumulative_timing.migration_time += migrate_time;
    cumulative_timing.total_time += total_time;
    cumulative_timing.step_count++;

    // Force-component breakdown (MD_TIMING_TRACE_FORCE=1): cudaEvent-based on
    // CUDA (ForceCalculatorCUDA::compute_forces(), zero added synchronization
    // when enabled), std::chrono wall-clock on CPU (ForceCalculatorCPU::
    // compute_forces()) -- see set_timing_breakdown_enabled().
    if (timing_trace_force_ && force_calc) {
        for (const auto& component : force_calc->get_last_force_component_timings()) {
            last_timing.force_components_ms.emplace_back(component.name, component.ms);

            auto it = std::find_if(cumulative_timing.force_component_time.begin(),
                                    cumulative_timing.force_component_time.end(),
                                    [&](const auto& entry) { return entry.first == component.name; });
            if (it != cumulative_timing.force_component_time.end()) {
                it->second += component.ms / 1000.0;
            } else {
                cumulative_timing.force_component_time.emplace_back(component.name, component.ms / 1000.0);
            }
        }
    }

    (void)modify_time;
}

void MDSimulation::handle_thermostat_phase(ForceCalculatorCUDA* cuda_calc,
                                           double& thermostat_time) {
    thermostat_time = measure_seconds([&]() {
        if (thermostat) {
            if (cuda_calc) {
#ifdef USE_CUDA
                thermostat->apply_device(cuda_calc->get_device_vx(), cuda_calc->get_device_vy(),
                                         cuda_calc->get_device_vz(), cuda_calc->get_device_mass(),
                                         system->get_local_particles(), dt, 0);
#else
                thermostat->apply(system->get_particles(), dt, system->get_local_particles());
#endif
            } else {
                thermostat->apply(system->get_particles(), dt, system->get_local_particles());
            }
        }
    });
}

void MDSimulation::handle_ke_output_phase(double& ke_time) {
    ke_time = measure_seconds([&]() {
        int next_step = current_step + 1;
        bool ke_needed = needs_ke_update_for_step(next_step);
        if (force_calc && ke_needed) {
            force_calc->calculate_kinetic_energy(*system);
        }
    });
}

void MDSimulation::handle_barostat_phase(ForceCalculatorCUDA* cuda_calc,
                                         double& barostat_time) {
    barostat_time = measure_seconds([&]() {
        if (barostat) {
            if (force_calc) {
                Real local_virial = force_calc->get_virial();
                Real global_virial = comm_backend_->sum_reduce(local_virial);
                barostat->set_virial(global_virial);
            }

            Real old_lx = system->get_box().lx;
            Real old_ly = system->get_box().ly;
            Real old_lz = system->get_box().lz;

            if (cuda_calc) {
#ifdef USE_CUDA
                barostat->apply_device(cuda_calc->get_device_x(), cuda_calc->get_device_y(),
                                       cuda_calc->get_device_z(),
                                       cuda_calc->get_device_vx(), cuda_calc->get_device_vy(),
                                       cuda_calc->get_device_vz(), cuda_calc->get_device_mass(),
                                       system->get_box(), dt, system->get_local_particles(), 0);
#else
                barostat->apply(system->get_particles(), system->get_box(), dt, system->get_local_particles());
#endif
            } else {
                barostat->apply(system->get_particles(), system->get_box(), dt, system->get_local_particles());
            }

            Real new_lx = system->get_box().lx;
            Real new_ly = system->get_box().ly;
            Real new_lz = system->get_box().lz;
            bool box_changed = has_significant_box_change(
                old_lx, old_ly, old_lz,
                new_lx, new_ly, new_lz);

            if (box_changed) {
                handle_box_change_rebuild();
            }
        }
    });
}

bool MDSimulation::has_significant_box_change(Real old_lx, Real old_ly, Real old_lz,
                                              Real new_lx, Real new_ly, Real new_lz) const {
    const Real eps = 16.0 * std::numeric_limits<Real>::epsilon();
    return std::abs(new_lx - old_lx) > eps * std::max<Real>(1.0, std::abs(old_lx))
        || std::abs(new_ly - old_ly) > eps * std::max<Real>(1.0, std::abs(old_ly))
        || std::abs(new_lz - old_lz) > eps * std::max<Real>(1.0, std::abs(old_lz));
}

void MDSimulation::handle_modify_phase(ForceCalculatorCUDA* cuda_calc,
                                       double& thermostat_time, double& ke_time,
                                       double& barostat_time, double& modify_time) {
    modify_time = measure_seconds([&]() {
        handle_thermostat_phase(cuda_calc, thermostat_time);
        handle_ke_output_phase(ke_time);
        handle_barostat_phase(cuda_calc, barostat_time);
    });
}

void MDSimulation::handle_box_change_rebuild() {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    neighbor_rebuild_forced = true;
    if (comm_backend_->is_distributed()) {
        system->sync_decomposition_domain();
        system->invalidate_halo_cache();
    }

    // Coordinate scaling invalidates both the neighbor list and F(x).
    // Rebuild and recompute now so the next Velocity-Verlet half-kick
    // never uses forces evaluated at the pre-scaled coordinates.
    if (comm_backend_->is_distributed()) {
#ifdef USE_CUDA
        if (cuda_calc) {
            system->exchange_halo_on_device(*cuda_calc, true);
        } else
#endif
        {
            system->perform_halo_exchange(true);
        }
    }

    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->build_neighbor_list_on_device(*system);
#else
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
#endif
    } else {
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
    }

    ++neighbor_list_builds;
    ++rebuild_reason_box_change_;
    neighbor_rebuild_forced = false;

    if (!cuda_calc) {
        zero_forces();
    }
    compute_forces_with_mode(true);
}

void MDSimulation::zero_forces() {
    // Zero forces for LOCAL particles only (LAMMPS-style optimization)
    // Ghost particle forces will be overwritten during force calculation
    ParticleData& particles = system->get_particles();
    size_t local_count = system->get_local_particles();

    for (size_t i = 0; i < local_count; i++) {
        particles.fx[i] = 0.0;
        particles.fy[i] = 0.0;
        particles.fz[i] = 0.0;
    }
}

void MDSimulation::zero_velocities() {
    ParticleData& particles = system->get_particles();
    size_t n = particles.vx.size();

    for (size_t i = 0; i < n; i++) {
        particles.vx[i] = 0.0;
        particles.vy[i] = 0.0;
        particles.vz[i] = 0.0;
    }
}

void MDSimulation::compute_fire_reduction_cpu(Real& power, Real& f_norm_sq,
                                              Real& v_norm_sq, Real& f_max_sq) const {
    const ParticleData& particles = system->get_particles();
    const size_t n = system->get_local_particles();

    power = 0.0; f_norm_sq = 0.0; v_norm_sq = 0.0; f_max_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        const Real fx = particles.fx[i], fy = particles.fy[i], fz = particles.fz[i];
        const Real vx = particles.vx[i], vy = particles.vy[i], vz = particles.vz[i];
        power += fx * vx + fy * vy + fz * vz;
        const Real f2 = fx * fx + fy * fy + fz * fz;
        f_norm_sq += f2;
        v_norm_sq += vx * vx + vy * vy + vz * vz;
        f_max_sq = std::max(f_max_sq, f2);
    }
}

void MDSimulation::apply_fire_update_cpu(const FireMinimizer::StepParams& params) {
    ParticleData& particles = system->get_particles();
    const SimulationBox& box = system->get_box();
    const size_t n = system->get_local_particles();
    const Real f_norm_inv = (params.mix_velocity && params.f_norm > 0.0)
        ? (1.0 / params.f_norm) : 0.0;

    for (size_t i = 0; i < n; i++) {
        Real vx = particles.vx[i], vy = particles.vy[i], vz = particles.vz[i];

        if (params.reset_velocity) {
            vx = vy = vz = 0.0;
        } else if (params.mix_velocity) {
            const Real fhx = particles.fx[i] * f_norm_inv;
            const Real fhy = particles.fy[i] * f_norm_inv;
            const Real fhz = particles.fz[i] * f_norm_inv;
            vx = (1.0 - params.alpha) * vx + params.alpha * params.v_norm * fhx;
            vy = (1.0 - params.alpha) * vy + params.alpha * params.v_norm * fhy;
            vz = (1.0 - params.alpha) * vz + params.alpha * params.v_norm * fhz;
        }

        const Real inv_m = 1.0 / particles.mass[i];
        vx += params.dt * particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        vy += params.dt * particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        vz += params.dt * particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        particles.vx[i] = vx;
        particles.vy[i] = vy;
        particles.vz[i] = vz;
        particles.x[i] += params.dt * vx;
        particles.y[i] += params.dt * vy;
        particles.z[i] += params.dt * vz;
        box.apply_pbc(particles.x[i], particles.y[i], particles.z[i]);
    }
}

void MDSimulation::run_energy_minimization() {
    if (minimizer_type_.empty() || minimizer_type_ == "none") {
        return;
    }
    if (minimizer_type_ != "fire") {
        throw std::runtime_error(
            "Unknown minimizer type: '" + minimizer_type_ + "' (supported: none, fire)");
    }

    FireConfig cfg;
    cfg.max_steps = minimizer_max_steps_;
    cfg.force_tolerance = minimizer_force_tolerance_;
    cfg.dt_start = (minimizer_dt_start_ > 0.0) ? minimizer_dt_start_ : (dt > 0.0 ? dt : 0.5);
    cfg.dt_max = (minimizer_dt_max_ > 0.0) ? minimizer_dt_max_ : (cfg.dt_start * 4.0);
    FireMinimizer fire(cfg);

    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    if (system->get_rank() == 0) {
        std::cout << "Starting FIRE energy minimization (max_steps=" << cfg.max_steps
                  << ", force_tolerance=" << cfg.force_tolerance << " kcal/mol/A)" << std::endl;
    }

    // FIRE starts from rest: the Maxwell-Boltzmann velocities initialize()
    // already seeded for the MD run would otherwise bias the first few
    // iterations, so this pseudo-dynamics optimizer gets its own v(0)=0.
    zero_velocities();

    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->initial_upload_to_device(*system);
#endif
    }

    double neighbor_time = 0.0, comm_time = 0.0, migrate_time = 0.0, force_time = 0.0;
    neighbor_rebuild_forced = true;
    handle_neighbor_phase(neighbor_time, comm_time, migrate_time);
    handle_force_phase(cuda_calc, force_time);

    int iter = 0;
    bool converged = false;
    for (; iter < cfg.max_steps; ++iter) {
        Real local_power = 0.0, local_f2 = 0.0, local_v2 = 0.0, local_fmax2 = 0.0;

        if (cuda_calc) {
#ifdef USE_CUDA
            const auto local = cuda_calc->compute_fire_reduction_on_device(
                system->get_local_particles());
            local_power = local.power;
            local_f2 = local.f_norm_sq;
            local_v2 = local.v_norm_sq;
            local_fmax2 = local.f_max_sq;
#endif
        } else {
            compute_fire_reduction_cpu(local_power, local_f2, local_v2, local_fmax2);
        }

        const Real global_power = comm_backend_->sum_reduce(local_power);
        const Real global_f_norm = std::sqrt(std::max<Real>(comm_backend_->sum_reduce(local_f2), 0.0));
        const Real global_v_norm = std::sqrt(std::max<Real>(comm_backend_->sum_reduce(local_v2), 0.0));
        const Real fmax = std::sqrt(std::max<Real>(comm_backend_->max_reduce(local_fmax2), 0.0));

        // FIRE has no built-in step-size safety check: an aggressive dt
        // combined with a badly-clashing starting geometry (large LJ/Coulomb
        // forces) can make the pseudo-dynamics diverge. Once a coordinate
        // goes non-finite, the GPU cell-list builder (build_cell_list_kernel
        // in neighbor_list_kernels.cu) is fed a NaN/Inf position on the next
        // handle_neighbor_phase() call below; without this guard that fault
        // propagates silently and is only ever surfaced several iterations
        // later (stdout for the intervening "[minimize] step N" lines is
        // buffered, so the divergence looks like an "immediate" crash) as a
        // misleading cudaErrorInvalidDevice from an unrelated thrust call.
        // Fail fast here instead, with an accurate diagnostic.
        if (!std::isfinite(global_power) || !std::isfinite(global_f_norm) ||
            !std::isfinite(global_v_norm) || !std::isfinite(fmax)) {
            throw std::runtime_error(
                "[minimize] FIRE diverged at step " + std::to_string(iter) +
                " (non-finite force/velocity norm). The starting geometry likely "
                "has severe atomic overlaps/clashes for the configured "
                "minimize.dt_start/minimize.dt_max; reduce the initial timestep "
                "or pre-relax the structure.");
        }

        if (system->get_rank() == 0 && (iter % minimizer_output_freq_ == 0)) {
            std::cout << "[minimize] step " << iter << "  PE=" << cached_potential_energy
                      << "  Fmax=" << fmax << std::endl;
        }

        if (fmax < cfg.force_tolerance) {
            converged = true;
            break;
        }

        const FireMinimizer::StepParams params =
            fire.advance(global_power, global_f_norm, global_v_norm);

        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->apply_fire_update_on_device(
                system->get_local_particles(), system->get_box(),
                params.reset_velocity, params.mix_velocity,
                params.alpha, params.v_norm, params.f_norm, params.dt);
#endif
        } else {
            apply_fire_update_cpu(params);
        }

        // Positions moved: force a full neighbor/halo refresh before the next
        // force evaluation. FIRE's step size is adaptive and can be large
        // early on, so there is no cheap displacement-based skip here like
        // in the main MD loop.
        neighbor_rebuild_forced = true;
        handle_neighbor_phase(neighbor_time, comm_time, migrate_time);
        handle_force_phase(cuda_calc, force_time);
    }

    if (system->get_rank() == 0) {
        std::cout << "[minimize] " << (converged ? "converged" : "stopped at max_steps")
                  << " after " << iter << " step(s), PE=" << cached_potential_energy
                  << std::endl;
    }

    if (cuda_calc) {
#ifdef USE_CUDA
        // Sync the host mirror with the minimized (device-resident) geometry
        // BEFORE clearing velocities -- download_from_device() would otherwise
        // bring FIRE's leftover pseudo-velocities back onto the host.
        cuda_calc->download_from_device(*system);
#endif
    }

    // FIRE's velocities are optimizer state, not physical thermal velocities.
    // Generate a fresh Maxwell distribution over the current post-migration
    // MPI ownership and scale it globally to target_temperature.
    system->initialize_distributed_velocities(target_temperature);

    // Force prepare_first_step() to rebuild the neighbor list and recompute
    // F(x) at the minimized (or max_steps-truncated) geometry.
    neighbor_rebuild_forced = true;
}


MDSimulation::TimingTraceMode MDSimulation::parse_timing_trace_mode() {
    const char* value = std::getenv("MD_TIMING_TRACE");
    if (!value) return TimingTraceMode::Off;
    const std::string v(value);
    if (v == "every") return TimingTraceMode::Every;
    if (v == "step") return TimingTraceMode::Step;
    return TimingTraceMode::Off;
}

void MDSimulation::run() {
    timing_trace_mode_ = parse_timing_trace_mode();
    timing_trace_force_ = env_flag_enabled("MD_TIMING_TRACE_FORCE");
    if (force_calc) {
        force_calc->set_timing_breakdown_enabled(timing_trace_force_);
    }
    if (system->get_rank() == 0) {
        std::cout << "Starting MD simulation for " << total_steps << " steps" << std::endl;
        if (timing_trace_mode_ != TimingTraceMode::Off) {
            std::cout << "Per-step timing trace enabled (MD_TIMING_TRACE="
                      << (timing_trace_mode_ == TimingTraceMode::Every ? "every" : "step") << ")"
                      << (timing_trace_force_ ? ", with force-component breakdown" : "") << std::endl;
        } else if (timing_trace_force_) {
            std::cout << "Force-component timing enabled (MD_TIMING_TRACE_FORCE=1); "
                      << "will appear in the end-of-run report" << std::endl;
        }
    }

    // Reset state flags
    current_step = 0;
    neighbor_rebuild_forced = false;
    adaptive_rebuild_armed_ = true;
    rebuild_reason_periodic_ = 0;
    rebuild_reason_adaptive_ = 0;
    rebuild_reason_forced_ = 0;
    rebuild_reason_mpi_sync_ = 0;
    rebuild_reason_box_change_ = 0;
    rebuild_reason_other_ = 0;

    // Optional energy minimization runs once before the MD loop, then restores
    // a physical velocity distribution at target_temperature. It leaves
    // neighbor_rebuild_forced set so prepare_first_step() rebuilds against the
    // relaxed geometry.
    const double minimize_time = measure_seconds([&]() {
        run_energy_minimization();
    });
    cumulative_timing.minimize_time += minimize_time;

    // Prepare initial forces at t=0 before starting MD loop
    prepare_first_step();

    // Output initial state (step 0)
    output_results(0);

    // Main simulation loop - current_step is already 1 from initialization
    for (int step = 1; step <= total_steps; step++) {
        single_step();

        if (timing_trace_mode_ == TimingTraceMode::Every && should_print_timing_for_step(step)) {
            print_step_timing(step);
        }

        handle_step_output(step);
    }

    // Output neighbor list statistics at end
    if (system->get_rank() == 0) {
        std::cout << "Neighbor list builds = " << neighbor_list_builds << std::endl;
        if (nonbonded_kernel_ == "graph") {
            std::cout << "Active graph incremental prunes = " << active_graph_prunes_
                      << ", safe reuses = " << active_graph_reuses_ << std::endl;
        }
        std::cout << "Neighbor rebuild reasons (overlapping counts):"
                  << " periodic=" << rebuild_reason_periodic_
                  << " adaptive=" << rebuild_reason_adaptive_
                  << " forced=" << rebuild_reason_forced_
                  << " mpi_sync=" << rebuild_reason_mpi_sync_
                  << " box_change=" << rebuild_reason_box_change_
                  << " other=" << rebuild_reason_other_
                  << std::endl;
    }
}

void MDSimulation::print_step_timing(int step) {
    if (timing_trace_mode_ == TimingTraceMode::Off || system->get_rank() != 0) {
        return;
    }

    std::cout << "[TIMING] step=" << step
              << " total_ms=" << last_timing.total_ms
              << " integrate_ms=" << last_timing.integrate_ms
              << " nlist_build_ms=" << last_timing.nlist_build_ms
              << " comm_ms=" << last_timing.comm_ms
              << " migrate_ms=" << last_timing.migrate_ms
              << " force_ms=" << last_timing.force_ms
              << " thermo_ms=" << last_timing.thermostat_ms
              << " ke_ms=" << last_timing.ke_ms
              << " baro_ms=" << last_timing.barostat_ms
              << " io_ms=" << last_timing.io_ms;
    for (const auto& [name, ms] : last_timing.force_components_ms) {
        std::cout << " force." << name << "_ms=" << ms;
    }
    std::cout << std::endl;
}

bool MDSimulation::should_output_step(int step) const {
    return (output_freq > 0) && (step % output_freq == 0);
}

bool MDSimulation::needs_ke_update_for_step(int step) const {
    return should_output_step(step) || should_write_trajectory_step(step) ||
           (step == total_steps);
}

bool MDSimulation::should_write_trajectory_step(int step) const {
    return (trajectory_output_freq > 0) && (step % trajectory_output_freq == 0);
}

bool MDSimulation::should_print_timing_for_step(int step) const {
    if (timing_trace_mode_ == TimingTraceMode::Off) {
        return false;
    }
    if (timing_trace_mode_ == TimingTraceMode::Every) {
        return true;
    }
    return should_output_step(step);
}

void MDSimulation::handle_step_output(int step) {
    const bool emit_text = should_output_step(step);
    const bool write_frame = should_write_trajectory_step(step);
    if (!emit_text && !write_frame) {
        return;
    }

    if (emit_text && should_print_timing_for_step(step) && timing_trace_mode_ != TimingTraceMode::Every) {
        print_step_timing(step);
    }

    // OPTIMIZATION: Removed MPI_Barrier before output.
    // The MPI_Allreduce calls in calculate_*_energy already synchronize enough.
    output_results(step, emit_text);
}

void MDSimulation::compute_output_energies(const ParticleData& particles, OutputMetrics& metrics) {
    if (force_calc) {
        metrics.total_ke = force_calc->get_kinetic_energy() * mvv2e;
        // cached_potential_energy is kept current every step in
        // compute_forces_with_mode(): force_calc's own accumulated PE (which
        // already sums bonds/angles/dihedrals/LJ/electrostatics/LibTorch
        // consistently for CPU, SYCL and CUDA) plus the Python MLIP
        // contribution, if enabled.
        metrics.total_pe = cached_potential_energy;
        return;
    }

    for (size_t i = 0; i < particles.num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i]
                + particles.vy[i] * particles.vy[i]
                + particles.vz[i] * particles.vz[i];
        metrics.total_ke += 0.5 * particles.mass[i] * v2;
    }
    metrics.total_ke *= mvv2e;
    metrics.total_pe = cached_potential_energy;
}

void MDSimulation::compute_output_force_speed_stats(const ParticleData& particles, OutputMetrics& metrics) {
    size_t num_particles_for_stats = system->get_local_particles();
    Real total_speed = 0.0;

    metrics.max_force_particle_id = 0;
    for (size_t i = 0; i < num_particles_for_stats; i++) {
        Real f_mag = std::sqrt(particles.fx[i] * particles.fx[i]
                             + particles.fy[i] * particles.fy[i]
                             + particles.fz[i] * particles.fz[i]);
        if (f_mag > metrics.max_force) {
            metrics.max_force = f_mag;
            metrics.max_force_particle_id = i;
        }

        Real speed = std::sqrt(particles.vx[i] * particles.vx[i]
                             + particles.vy[i] * particles.vy[i]
                             + particles.vz[i] * particles.vz[i]);
        if (speed > metrics.max_speed) {
            metrics.max_speed = speed;
        }
        total_speed += speed;
    }

    metrics.avg_speed = (particles.num_particles > 0) ? (total_speed / particles.num_particles) : 0.0;
}

void MDSimulation::compute_output_thermodynamics(const ParticleData& particles, OutputMetrics& metrics) {
    int local_particle_count = static_cast<int>(system->get_local_particles());
    int global_particles = comm_backend_->sum_int(local_particle_count);
    comm_backend_->reduce_max_force_particle(
        metrics.max_force, metrics.max_force_particle_id, system->get_rank(),
        metrics.max_force, metrics.max_force_particle_id);

    Real global_virial_for_press = 0.0;
    if (force_calc) {
        Real local_virial = force_calc->get_virial();
        global_virial_for_press = comm_backend_->sum_reduce(local_virial);
    }

    if (particles.num_particles > 0 && system->get_rank() == 0) {
        metrics.temperature = (2.0 * metrics.total_ke) / (3.0 * global_particles * kB);

        Real volume = system->get_box().lx * system->get_box().ly * system->get_box().lz;
        if (volume > 0.0) {
            metrics.pressure = (2.0 * metrics.total_ke + global_virial_for_press) / (3.0 * volume) * nktv2p;
        }
    }
}

void MDSimulation::emit_rank0_output(int step, const OutputMetrics& metrics) {
    if (system->get_rank() != 0) {
        return;
    }

    std::cout << format_output_line(step, metrics) << std::endl;
}

void MDSimulation::sync_output_state_to_host_if_needed() {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->download_from_device(*system);
#endif
    }
}

std::string MDSimulation::format_output_line(int step, const OutputMetrics& metrics) const {
    std::stringstream output_line;
    output_line << std::fixed << std::setprecision(6);

    static const std::vector<OutputField> kDefaultOutputFields = {
        OutputField::STEP, OutputField::T, OutputField::KE
    };
    const std::vector<OutputField>& fields =
        output_fields_cache.empty() ? kDefaultOutputFields : output_fields_cache;

    for (OutputField field : fields) {
        switch (field) {
        case OutputField::STEP:
            output_line << "Step: " << step;
            break;
        case OutputField::T:
            output_line << ", T: " << metrics.temperature;
            break;
        case OutputField::KE:
            output_line << ", KE: " << metrics.total_ke;
            break;
        case OutputField::PE:
            output_line << ", PE: " << metrics.total_pe;
            break;
        case OutputField::ETOTAL:
            output_line << ", Etot: " << metrics.total_ke + metrics.total_pe;
            break;
        case OutputField::PRESS:
            output_line << ", P: " << metrics.pressure;
            break;
        case OutputField::MAX_SPEED:
            output_line << ", MaxSpd: " << metrics.max_speed;
            break;
        case OutputField::AVG_SPEED:
            output_line << ", AvgSpd: " << metrics.avg_speed;
            break;
        case OutputField::FMAX:
            output_line << ", Fmax: " << metrics.max_force
                        << " (particle " << metrics.max_force_particle_id << ")";
            break;
        }
    }

    return output_line.str();
}

void MDSimulation::output_results(int step, bool emit_text) {
    double output_time = measure_seconds([&]() {
        sync_output_state_to_host_if_needed();

        OutputMetrics metrics;
        ParticleData& particles = system->get_particles();

        compute_output_energies(particles, metrics);
        compute_output_force_speed_stats(particles, metrics);
        compute_output_thermodynamics(particles, metrics);

        if (emit_text) {
            emit_rank0_output(step, metrics);
        }

        // Collective: every rank must call this (MPI_Gatherv for the serial
        // formats, ADIOS2 Put/BeginStep/EndStep for bp) -- it cannot live
        // inside emit_rank0_output(), which returns early on rank != 0.
        if (should_write_trajectory_step(step)) {
            write_trajectory(step);
        }
    });

    last_timing.io_ms = output_time * 1000.0;
    cumulative_timing.output_time += output_time;
}

void MDSimulation::finalize(double program_wallclock_ms) {
    // CRITICAL: All ranks must call this function -- the multi-rank path in
    // print_timing_report() uses MPI_Reduce.
    if (cumulative_timing.step_count > 0) {
        print_timing_report(program_wallclock_ms);
    }

    if (force_calc) {
        // Unregister pinned buffers before destroying the CUDA context
        ForceCalculatorCUDA* cuda_fin = cuda_calc_cached_;
        if (cuda_fin && force_calculation_type == "mace") {
#ifdef USE_CUDA
            cuda_fin->unregister_ml_pinned_buffers(system->get_particles());
#endif
        }
        force_calc->finalize();
    }
    
    if (python_interface) {
        python_interface->finalize_python();
    }
    
    // Collective: patches the DCD NSET header and closes the ADIOS2 engine
    // (Engine::Close() is collective across all writing ranks).
    finalize_trajectory_writers();

    if (system->get_rank() == 0) {
        std::cout << "MD simulation finalized" << std::endl;
    }

    // Note: MPI_Finalize() should be called in main() after simulation object is destroyed
}

void MDSimulation::parse_simulation_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_file);
    }
    
    // Track which required parameters are found
    bool found_coordinate_file = false;
    bool found_target_temperature = false;
    bool found_timestep = false;
    bool found_total_steps = false;
    std::string input_system;
    
    std::string line;
    std::string current_section = "";
    

    while (std::getline(file, line)) {
        // Remove inline comments (everything after # character)
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        // Skip empty lines after comment removal
        if (line.empty()) continue;
        
        // Preserve indentation before trimming. A non-indented key/value
        // after a mapping section closes that section.
        const bool is_indented = !line.empty() && (line[0] == ' ' || line[0] == '\t');

        // Check for YAML section headers BEFORE trimming
        // A section header is a non-indented line ending with ':'  (no value after ':')
        bool is_section_header = false;
        if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
            size_t colon_at = line.find(':');
            if (colon_at != std::string::npos) {
                // Check that the rest of the line (after colon, trimmed) is empty or a comment
                std::string after_colon = line.substr(colon_at + 1);
                after_colon.erase(0, after_colon.find_first_not_of(" \t\r\n"));
                if (after_colon.empty()) {
                    is_section_header = true;
                    current_section = line.substr(0, colon_at);
                    current_section.erase(0, current_section.find_first_not_of(" \t"));
                    current_section.erase(current_section.find_last_not_of(" \t") + 1);
                }
            }
        }

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Skip empty lines after trimming
        if (line.empty()) continue;
        
        // Skip section header lines (they have no value to parse)
        if (is_section_header) continue;

        if (!is_indented) {
            current_section.clear();
        }
        
        // Parse key-value pairs
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        
        // Trim whitespace and indentation (including \r and \n)
        const char* ws = " \t\n\r";
        
        // Trim key
        size_t key_start = key.find_first_not_of(ws);
        if (key_start != std::string::npos) {
            size_t key_end = key.find_last_not_of(ws);
            key = key.substr(key_start, key_end - key_start + 1);
        } else {
            key.clear();
        }
        
        // Trim value
        size_t value_start = value.find_first_not_of(ws);
        if (value_start != std::string::npos) {
            size_t value_end = value.find_last_not_of(ws);
            value = value.substr(value_start, value_end - value_start + 1);
        } else {
            value.clear();
        }
        
        // Remove quotes from string values
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
            // Trim again after removing quotes
            size_t start = value.find_first_not_of(ws);
            if (start != std::string::npos) {
                size_t end = value.find_last_not_of(ws);
                value = value.substr(start, end - start + 1);
            } else {
                value.clear();
            }
        }
        
        // Parse simulation parameters based on section and key
        if (current_section == "force_calculation") {
            if (key == "mode" || key == "force_calculation_mode") {
                force_calculation_mode = trim_copy(value);
            } else if (key == "type" || key == "force_calculation_type") {
                force_calculation_type = value;
            } else if (key == "spec_file" || key == "mlip_spec_file") {
                mlip_spec_file_ = value;
            } else if (key == "precision") {
                force_precision_ = trim_copy(value);
            } else if (key == "nonbonded_kernel") {
                nonbonded_kernel_ = trim_copy(value);
            } else if (key == "pme_cluster_spread") {
                std::string v = trim_copy(value);
                pme_cluster_spread_ = (v == "true" || v == "1" || v == "on");
            } else if (key == "electrostatics_solver") {
                electrostatics_solver_ = trim_copy(value);
            }
        } else if (current_section == "minimize") {
            if (key == "method") {
                minimizer_type_ = value;
            } else if (key == "max_steps") {
                minimizer_max_steps_ = std::stoi(value);
            } else if (key == "force_tolerance") {
                minimizer_force_tolerance_ = std::stod(value);
            } else if (key == "dt_start") {
                minimizer_dt_start_ = std::stod(value);
            } else if (key == "dt_max") {
                minimizer_dt_max_ = std::stod(value);
            } else if (key == "output_freq") {
                minimizer_output_freq_ = std::stoi(value);
            }
        } else if (current_section == "mlip_config" || current_section == "mace_config") {
            // mlip_config  — unified section name (mace_config kept for backward compat)
            if (key == "model_path") {
                mlip_model_path = value;
            } else if (key == "model_type") {
                mlip_model_type = value;
            } else if (key == "model_size") {
                mlip_model_size = value;
            } else if (key == "device") {
                mlip_device = value;
            } else if (key == "dtype") {
                mlip_dtype = value;
            } else if (!key.empty()) {
                // Any other mlip_config keys (e.g. head, task_name, species_to_type_name)
                // are forwarded as extra kwargs to initialize_ml_forcefield_generic().
                mlip_extra_kwargs[key] = value;
            }
        } else if (key == "input_system") {
            input_system = value;
        } else if (key == "coordinate_file") {
            coordinate_file = value;
            found_coordinate_file = true;
        } else if (key == "topology_file") {
            topology_file = value;
        } else if (key == "parameter_file") {
            parameter_file = value;
        } else if (key == "target_temperature") {
            target_temperature = std::stod(value);
            found_target_temperature = true;
        } else if (key == "timestep") {
            dt = std::stod(value);
            found_timestep = true;
        } else if (key == "total_steps") {
            total_steps = std::stoi(value);
            found_total_steps = true;
        } else if (key == "output_prefix") {
            output_prefix = value;
        } else if (key == "trajectory_output_freq") {
            trajectory_output_freq = std::stoi(value);
        } else if (key == "trajectory_format") {
            trajectory_format = value;
        } else if (key == "trajectory_include_velocities") {
            trajectory_include_velocities = (value == "true");
        } else if (key == "trajectory_include_forces") {
            trajectory_include_forces = (value == "true");
        } else if (key == "energy_output_freq") {
            energy_output_freq = std::stoi(value);
        } else if (key == "output_freq") {
            output_freq = std::stoi(value);
        } else if (key == "output_info") {
            output_info = value;
        } else if (key == "cutoff_distance") {
            cutoff_distance = std::stod(value);
        } else if (key == "cutoff") {
            cutoff_distance = std::stod(value);
        } else if (key == "lj_switch_distance" || key == "switchdist") {
            lj_switch_distance = std::stod(value);
        } else if (key == "neighbor_list_update_freq") {
            neighbor_list_update_freq = std::stoi(value);
        } else if (key == "verlet_skin") {
            verlet_skin = std::stod(value);
        } else if (key == "neighbor_rebuild_hi_frac") {
            neighbor_rebuild_hi_frac = std::stod(value);
        } else if (key == "neighbor_rebuild_lo_frac") {
            neighbor_rebuild_lo_frac = std::stod(value);
        } else if (key == "neighbor_displacement_check_stride") {
            neighbor_displacement_check_stride = std::stoi(value);
        } else if (key == "integrator") {
            integrator_type = value;
        } else if (key == "thermostat") {
            thermostat_type = value;
        } else if (key == "tau_t") {
            tau_t = std::stod(value);
        } else if (key == "thermostat_damping") {
            tau_t = std::stod(value);  // thermostat_damping is alias for tau_t
        } else if (key == "barostat") {
            barostat_type = value;
        } else if (key == "target_pressure") {
            target_pressure = std::stod(value);
        } else if (key == "tau_p") {
            tau_p = std::stod(value);
        } else if (key == "barostat_damping") {
            tau_p = std::stod(value);  // barostat_damping is alias for tau_p
        } else if (key == "python_force_path") {
            python_force_path = value;
        } else if (key == "python_force_enabled") {
            python_force_enabled = (value == "true");
        } else if (key == "mlip_spec_file") {
            mlip_spec_file_ = value;
        } else if (key == "box_x") {
            system->get_box().lx = std::stod(value);
        } else if (key == "box_y") {
            system->get_box().ly = std::stod(value);
        } else if (key == "box_z") {
            system->get_box().lz = std::stod(value);
        } else if (key == "periodic_x") {
            system->get_box().periodic_x = (value == "true");
        } else if (key == "periodic_y") {
            system->get_box().periodic_y = (value == "true");
        } else if (key == "periodic_z") {
            system->get_box().periodic_z = (value == "true");
        }
    }
    
    if (!input_system.empty()) {
        const auto manifest = read_dsmd_input_manifest(input_system);
        const std::filesystem::path base =
            std::filesystem::absolute(std::filesystem::path(input_system)).parent_path();
        auto resolve = [&](const char* key) -> std::string {
            auto it = manifest.find(key);
            if (it == manifest.end()) return "";
            const std::filesystem::path path(it->second);
            return (path.is_absolute() ? path : base / path).lexically_normal().string();
        };
        if (coordinate_file.empty()) coordinate_file = resolve("coordinate_file");
        if (topology_file.empty()) topology_file = resolve("topology_file");
        if (parameter_file.empty()) parameter_file = resolve("parameter_file");
        found_coordinate_file = !coordinate_file.empty();
    }

    // Validate required parameters
    std::vector<std::string> missing_params;
    if (!found_coordinate_file) missing_params.push_back("coordinate_file");
    if (!input_system.empty() && topology_file.empty()) missing_params.push_back("topology_file");
    if (!input_system.empty() && parameter_file.empty()) missing_params.push_back("parameter_file");
    if (!found_target_temperature) missing_params.push_back("target_temperature");
    if (!found_timestep) missing_params.push_back("timestep");
    if (!found_total_steps) missing_params.push_back("total_steps");
    
    if (!missing_params.empty()) {
        std::string error_msg = "Missing required configuration parameters: ";
        for (size_t i = 0; i < missing_params.size(); ++i) {
            if (i > 0) error_msg += ", ";
            error_msg += missing_params[i];
        }
        throw std::runtime_error(error_msg);
    }
    
    // Parse force components.  `mode` accepts a comma-separated set, e.g.
    // "builtin, python, libtorch".
    force_calculation_mode = trim_copy(force_calculation_mode);
    const auto force_components = parse_force_components(force_calculation_mode);
    for (const auto& component : force_components) {
        if (component != "builtin" && component != "python" && component != "libtorch") {
            throw std::runtime_error(
                "Invalid force_calculation.mode component: '" + component
                + "'. Supported components are: builtin, python, libtorch");
        }
    }
    if (force_components.empty()) {
        throw std::runtime_error(
            "Invalid force_calculation.mode: '" + force_calculation_mode
            + "'. Specify one or more of: builtin, python, libtorch");
    }
    enable_builtin_forces_  = force_components.count("builtin") != 0;
    enable_python_forces_   = force_components.count("python") != 0;
    enable_libtorch_forces_ = force_components.count("libtorch") != 0;

    force_precision_ = trim_copy(force_precision_);
    if (force_precision_.empty()) {
        force_precision_ = "double";
    }
    if (force_precision_ != "double" && force_precision_ != "mixed") {
        throw std::runtime_error(
            "Invalid force_calculation.precision: '" + force_precision_
            + "'. Supported values are: double, mixed");
    }

    nonbonded_kernel_ = trim_copy(nonbonded_kernel_);
    if (nonbonded_kernel_.empty()) {
        nonbonded_kernel_ = "tiled";
    }
    std::transform(nonbonded_kernel_.begin(), nonbonded_kernel_.end(),
                   nonbonded_kernel_.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (nonbonded_kernel_ != "tiled" && nonbonded_kernel_ != "graph") {
        throw std::runtime_error(
            "Invalid force_calculation.nonbonded_kernel: '" + nonbonded_kernel_
            + "'. Supported values are: tiled, graph");
    }

    // STAGE 3 PLACEHOLDER (see PME reciprocal-space parallelization roadmap):
    // "fmm" is accepted here and threaded through to
    // ForceCalculatorCPU/ForceCalculatorCUDA so the config surface and
    // dispatch plumbing already exist, but there is no FMM implementation
    // yet -- selecting it fails fast with a clear message rather than
    // silently falling back to PME. Wire up the real solve when FFT-based
    // PME's communication cost (even with slab-decomposed FFTW MPI /
    // cuFFTMp, see stage 2) becomes the hard limit at very large atom/rank
    // counts.
    electrostatics_solver_ = trim_copy(electrostatics_solver_);
    if (electrostatics_solver_.empty()) {
        electrostatics_solver_ = "pme";
    }
    if (electrostatics_solver_ != "pme" && electrostatics_solver_ != "fmm") {
        throw std::runtime_error(
            "Invalid force_calculation.electrostatics_solver: '" + electrostatics_solver_
            + "'. Supported values are: pme, fmm");
    }
    if (electrostatics_solver_ == "fmm") {
        throw std::runtime_error(
            "force_calculation.electrostatics_solver = fmm is a placeholder for a "
            "future Fast Multipole Method solver and is not implemented yet. Use "
            "'pme' (the default).");
    }


    // Parse trajectory_format.  Accepts a comma-separated set, e.g.
    // "xyz, dcd, bp". "both" is a legacy alias for "xyz, dump".
    {
        std::string normalized = trajectory_format;
        if (trim_copy(normalized) == "both") normalized = "xyz,dump";
        trajectory_formats_ = parse_force_components(normalized);
        for (const auto& fmt : trajectory_formats_) {
            if (fmt != "xyz" && fmt != "dump" && fmt != "dcd" && fmt != "bp" && fmt != "h5md") {
                throw std::runtime_error(
                    "Invalid trajectory_format component: '" + fmt
                    + "'. Supported components are: xyz, dump, dcd, bp, h5md (or legacy 'both')");
            }
        }
#ifndef USE_ADIOS2
        if (trajectory_formats_.count("bp")) {
            throw std::runtime_error(
                "trajectory_format 'bp' requires a build with ADIOS2 enabled "
                "(-DENABLE_ADIOS2=ON at CMake configure time).");
        }
#endif
#ifndef USE_HDF5
        if (trajectory_formats_.count("h5md")) {
            throw std::runtime_error(
                "trajectory_format 'h5md' requires a build with parallel HDF5 enabled "
                "(-DENABLE_HDF5=ON at CMake configure time).");
        }
#endif
    }

    // Validate parameter values
    if (dt <= 0) {
        throw std::runtime_error("Invalid timestep: must be positive");
    }
    if (total_steps <= 0) {
        throw std::runtime_error("Invalid total_steps: must be positive");
    }
    if (target_temperature < 0) {
        throw std::runtime_error("Invalid target_temperature: must be non-negative");
    }
    if (thermostat_type != "none" && !thermostat_type.empty() && tau_t <= 0.0) {
        throw std::runtime_error("Invalid thermostat damping time: must be positive");
    }
    if (minimizer_type_ != "none" && !minimizer_type_.empty()) {
        if (minimizer_type_ != "fire") {
            throw std::runtime_error(
                "Invalid minimize.method: '" + minimizer_type_ + "' (supported: none, fire)");
        }
        if (minimizer_max_steps_ <= 0) {
            throw std::runtime_error("Invalid minimize.max_steps: must be positive");
        }
        if (minimizer_force_tolerance_ <= 0.0) {
            throw std::runtime_error("Invalid minimize.force_tolerance: must be positive");
        }
    }
    if (neighbor_list_update_freq <= 0) {
        throw std::runtime_error("Invalid neighbor_list_update_freq: must be positive");
    }
    if (!(neighbor_rebuild_lo_frac > 0.0 && neighbor_rebuild_lo_frac < 1.0)) {
        throw std::runtime_error("Invalid neighbor_rebuild_lo_frac: must be in (0, 1)");
    }
    if (!(neighbor_rebuild_hi_frac > 0.0 && neighbor_rebuild_hi_frac < 1.0)) {
        throw std::runtime_error("Invalid neighbor_rebuild_hi_frac: must be in (0, 1)");
    }
    if (neighbor_rebuild_lo_frac >= neighbor_rebuild_hi_frac) {
        throw std::runtime_error("Invalid adaptive rebuild thresholds: neighbor_rebuild_lo_frac must be smaller than neighbor_rebuild_hi_frac");
    }
    // Verlet-list safety invariant: two particles closing head-on can together
    // cover at most `skin` before a rebuild is forced, which requires each
    // particle's own displacement trigger to fire at or before skin/2. This
    // must be enforced here, not just defaulted correctly: an hi_frac above
    // 0.5 lets a close-contact pair go completely unlisted between rebuilds
    // (zero force applied while it keeps closing), producing a large,
    // unphysical force spike and injecting energy the moment it's finally
    // detected. The CPU path (CpuSpatialVerletList::needs_rebuild)
    // hard-codes 0.5 for exactly this reason; the GPU/adaptive path exposed
    // hi_frac as a free config knob without this bound, which is what let an
    // unsafe default (0.55) ship silently.
    if (neighbor_rebuild_hi_frac > 0.5) {
        throw std::runtime_error(
            "Invalid neighbor_rebuild_hi_frac: must be <= 0.5 to preserve the Verlet-skin "
            "safety invariant (two particles closing head-on must not together cover more "
            "than the skin distance before a rebuild is forced); got " +
            std::to_string(neighbor_rebuild_hi_frac));
    }
    if (neighbor_displacement_check_stride < 0) {
        throw std::runtime_error("Invalid neighbor_displacement_check_stride: must be non-negative (0 means auto)");
    }
    if (lj_switch_distance >= 0.0 &&
        (lj_switch_distance <= 0.0 || lj_switch_distance >= cutoff_distance)) {
        throw std::runtime_error("Invalid lj_switch_distance: must be positive and smaller than cutoff_distance");
    }
    
    // Set the physical cutoff in the system for force calculations
    system->set_physical_cutoff(cutoff_distance);
    system->set_verlet_skin(verlet_skin);
    system->get_ff_params().lj_switch_distance = lj_switch_distance;
    rebuild_output_info_items_cache();
    if (system->get_rank() == 0) {
        std::cout << "Configuration loaded successfully:" << std::endl;
        std::cout << "  Coordinate file: " << coordinate_file << std::endl;
        if (!parameter_file.empty()) {
            std::cout << "  Parameter file: " << parameter_file << std::endl;
        }
        std::cout << "  Target temperature: " << target_temperature << " K" << std::endl;
        std::cout << "  Timestep: " << dt << " fs" << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;
        std::cout << "  Integrator: " << integrator_type << std::endl;
        std::cout << "  Thermostat: " << thermostat_type << std::endl;
        std::cout << "  Barostat: " << barostat_type << std::endl;
        if (minimizer_type_ != "none" && !minimizer_type_.empty()) {
            std::cout << "  Minimizer: " << minimizer_type_
                      << " (max_steps=" << minimizer_max_steps_
                      << ", force_tolerance=" << minimizer_force_tolerance_
                      << " kcal/mol/A)" << std::endl;
        }
        {
            std::string fmt_list;
            for (const auto& fmt : trajectory_formats_) {
                if (!fmt_list.empty()) fmt_list += ", ";
                fmt_list += fmt;
            }
            std::cout << "  Trajectory formats: " << fmt_list << std::endl;
        }
        std::cout << "  Box size: " << system->get_box().lx << " x " << system->get_box().ly << " x " << system->get_box().lz << std::endl;
        std::cout << "  Periodic BC: " << (system->get_box().periodic_x ? "T" : "F") 
                  << " " << (system->get_box().periodic_y ? "T" : "F") 
                  << " " << (system->get_box().periodic_z ? "T" : "F") << std::endl;
        std::cout << "  Cutoff distance: " << cutoff_distance << std::endl;
        if (lj_switch_distance > 0.0) {
            std::cout << "  LJ switch distance: " << lj_switch_distance << std::endl;
        }
        std::cout << "  Neighbor list update frequency: " << neighbor_list_update_freq << " steps" << std::endl;
        std::cout << "  Neighbor adaptive rebuild: hi=" << neighbor_rebuild_hi_frac
              << "*skin, lo=" << neighbor_rebuild_lo_frac
              << "*skin, check_stride="
              << (neighbor_displacement_check_stride > 0
                  ? std::to_string(neighbor_displacement_check_stride)
                  : std::string("auto(update_freq/4)"))
              << std::endl;
        std::cout << "  Force calculation mode: " << force_calculation_mode << std::endl;
        if (enable_python_forces_) {
            std::cout << "  Python force path: " << python_force_path << std::endl;
        }
        std::cout << "  Python force interface: " << (python_force_enabled ? "enabled" : "disabled") << std::endl;
        if (python_force_enabled) {
            std::cout << "  Python force path: " << python_force_path << std::endl;
        }
    }
}

void MDSimulation::initialize_integrator() {
    const char* label = backend_label();

    if (integrator_type == "velocity_verlet") {
        integrator = create_velocity_verlet_integrator_for_build(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Velocity-Verlet integrator" << std::endl;
        }

    } else if (integrator_type == "r_respa") {
        throw std::runtime_error(
            "r_respa requires explicit fast/slow force components and is not supported by the current force API");

    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        integrator = create_leap_frog_integrator_for_build(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Leap-frog integrator" << std::endl;
        }
    } else {
        // Default to velocity-verlet
        integrator = create_velocity_verlet_integrator_for_build(dt);
        if (system->get_rank() == 0) {
            std::cout << "Unknown integrator type '" << integrator_type 
                      << "', using Velocity-Verlet as default" << std::endl;
        }
    }
}

void MDSimulation::initialize_thermostat() {
    const char* label = backend_label();

    if (thermostat_type == "none" || thermostat_type.empty()) {
        thermostat = nullptr;
        if (system->get_rank() == 0) {
            std::cout << "No thermostat initialized" << std::endl;
        }
        return;
    }
    
    if (thermostat_type == "nose_hoover") {
        // The thermostat computes Q = dof*kB*T*tau_t^2 once the global atom
        // count is known.  Pass the requested relaxation time here.
        Real Q = tau_t;
        thermostat = create_nose_hoover_thermostat_for_build(target_temperature, Q);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Nose-Hoover thermostat (T=" << target_temperature << " K)" << std::endl;
        }
    } else if (thermostat_type == "langevin") {
        Real gamma = 1.0 / tau_t; // friction coefficient from time constant
        thermostat = create_langevin_thermostat_for_build(target_temperature, gamma);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Langevin thermostat (T=" << target_temperature 
                      << " K, tau=" << tau_t << " ps)" << std::endl;
        }
    } else if (thermostat_type == "andersen") {
        Real collision_freq = 1.0 / tau_t; // collision frequency from time constant
        thermostat = create_andersen_thermostat_for_build(target_temperature, collision_freq);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Andersen thermostat (T=" << target_temperature 
                      << " K, freq=" << collision_freq << " ps^-1)" << std::endl;
        }
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown thermostat type '" << thermostat_type 
                      << "', no thermostat initialized" << std::endl;
        }
    }
}

void MDSimulation::initialize_barostat() {
    const char* label = backend_label();

    if (barostat_type == "none" || barostat_type.empty()) {
        barostat = nullptr;
        if (system->get_rank() == 0 ) {
            std::cout << "No barostat initialized" << std::endl;
        }
        return;
    }
    
    // Validate identically on every rank.  Rank-local early returns would leave
    // different ranks with different barostat state and deadlock collectives.
    bool ok = true;
    if (system->get_rank() == 0) {
        std::cout << "[DEBUG PARAM] === NPT 파라미터 검증 시작 ===" << std::endl;
        
        if (dt <= 0 || dt > 0.005) {
            std::cerr << "[DEBUG PARAM] WARN: dt=" << dt 
                      << " — NPT에서는 0.001~0.002 권장" << std::endl;
        }
        
        if (tau_p <= 0) {
            std::cerr << "[DEBUG PARAM] FATAL: tau_p=" << tau_p << " — 양수여야 함" << std::endl;
            ok = false;
        } else if (tau_p < 10.0 * dt) {
            std::cerr << "[DEBUG PARAM] WARN: tau_p=" << tau_p 
                      << " 가 너무 작음 (dt의 " << (tau_p / dt) << "배)" << std::endl;
            std::cerr << "  → 최소 tau_p >= 10*dt = " << (10.0 * dt) << " 권장" << std::endl;
        }
        
        if (!std::isfinite(target_pressure)) {
            std::cerr << "[DEBUG PARAM] FATAL: target_pressure must be finite" << std::endl;
        }
        
        // Check cutoff vs sigma (assuming sigma ~1.0 for LJ)
        Real cutoff = system->get_physical_cutoff();
        Real sigma_approx = 1.0;  // Typical for LJ systems
        if (cutoff < 2.0 * sigma_approx) {
            std::cerr << "[DEBUG PARAM] WARN: cutoff=" << cutoff << " < 2*sigma=" << (2.0 * sigma_approx) << std::endl;
            std::cerr << "  → virial 계산 부정확으로 압력 오차 증가" << std::endl;
        }
        
    }

    ok = (dt > 0.0) && (tau_p > 0.0) && std::isfinite(target_pressure);
    ok = comm_backend_->all_true(ok);
    if (!ok) {
        if (system->get_rank() == 0) {
            std::cerr << "Invalid NPT parameters; disabling barostat on every rank." << std::endl;
        }
        barostat_type = "none";
        barostat.reset();
        return;
    }
    
    if (barostat_type == "parrinello_rahman") {
        int local_particles = static_cast<int>(system->get_local_particles());
        int global_particles = comm_backend_->sum_int(local_particles);
        // Extended-system mass in energy*time^2 units.
        Real thermal_energy = std::max<Real>(target_temperature, 1.0) * kB;
        Real W = (3.0 * global_particles + 3.0) * thermal_energy * tau_p * tau_p;
        
        if (system->get_rank() == 0) {
            std::cout << "[DEBUG] Isotropic Parrinello-Rahman: tau_p=" << tau_p
                      << " W=" << W << std::endl;
        }
        barostat = create_parrinello_rahman_barostat_for_build(target_pressure, W);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                  << " Parrinello-Rahman barostat (P=" << target_pressure 
                      << " bar)" << std::endl;
        }
    } else if (barostat_type == "andersen") {
        // Store beta/tau_p so apply(dt) implements beta*dt/tau_p coupling.
        Real compressibility = 1e-5 / tau_p;
        barostat = create_andersen_barostat_for_build(target_pressure, compressibility);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                  << " Andersen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " ps)" << std::endl;
        }
    } else if (barostat_type == "berendsen") {
        // Berendsen barostat - simpler and more stable than Parrinello-Rahman
        // Does not produce correct NPT ensemble but good for equilibration
        barostat = std::make_unique<BerendsenBarostat>(target_pressure, tau_p);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Berendsen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " fs)" << std::endl;
        }
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown barostat type '" << barostat_type 
                      << "', no barostat initialized" << std::endl;
        }
    }
}

void MDSimulation::check_energy_stability() {
    static Real last_total_energy = 0.0;
    static bool first_check = true;
    
    if (!force_calc) return;
    
    Real kinetic_energy = force_calc->get_kinetic_energy();
    Real potential_energy = force_calc->get_potential_energy();
    Real total_energy = kinetic_energy + potential_energy;
    
    if (first_check) {
        last_total_energy = total_energy;
        first_check = false;
        return;
    }
    
    // Check for energy drift or sudden jumps
    Real energy_change = std::abs(total_energy - last_total_energy);
    Real relative_change = energy_change / std::abs(last_total_energy + 1e-10);
    
    // If energy change is more than 10% per step, something is wrong
    if (relative_change > 0.1) {
        if (system->get_rank() == 0) {
            std::cout << "Warning: Large energy change detected at step " << current_step 
                      << " (ΔE/E = " << relative_change << ")" << std::endl;
            std::cout << "  Previous total energy: " << last_total_energy << " kcal/mol" << std::endl;
            std::cout << "  Current total energy:  " << total_energy << " kcal/mol" << std::endl;
            std::cout << "  Kinetic energy:       " << kinetic_energy << " kcal/mol" << std::endl;
            std::cout << "  Potential energy:     " << potential_energy << " kcal/mol" << std::endl;
        }
        
        // Consider reducing timestep if energy is increasing rapidly
        if (total_energy > last_total_energy * 1.05) { // More sensitive threshold
            dt *= 0.5; // Reduce timestep by 50%
            if (system->get_rank() == 0) {
                std::cout << "  Reducing timestep to " << dt << " ps" << std::endl;
            }
            
            // If timestep becomes too small, halt simulation
            if (dt < 1e-6) {
                if (system->get_rank() == 0) {
                    std::cout << "ERROR: Timestep became too small (dt < 1e-6 ps). Simulation unstable!" << std::endl;
                }
                throw std::runtime_error("Simulation became numerically unstable");
            }
        }
    }
    
    last_total_energy = total_energy;
}

// Trajectory output functions
// Collective: gathers every rank's LOCAL (owned, non-ghost) particles onto
// rank 0 via MPI_Gatherv, reordered by global_id. Every rank owns a
// spatially-determined, non-contiguous subset of global ids that also
// changes over time as dsDecompX migrates particles, so the gather order
// (by MPI rank) must be un-scrambled back into atom-index order here --
// otherwise the same atom would land in a different row of the output file
// on every frame. Must be called by every rank (it is a collective MPI
// operation); only rank 0's `frame` is populated.
void MDSimulation::gather_particles_for_serial_output(SerialTrajectoryFrame& frame) const {
    const int rank = system->get_rank();
    const int size = system->get_size();
    const size_t local_n = system->get_local_particles();
    ParticleData& particles = system->get_particles();
    const bool need_v = trajectory_include_velocities;
    const bool need_f = trajectory_include_forces;

    if (size == 1) {
        frame.x.resize(local_n); frame.y.resize(local_n); frame.z.resize(local_n);
        frame.type.resize(local_n); frame.element.resize(local_n);
        if (need_v) { frame.vx.resize(local_n); frame.vy.resize(local_n); frame.vz.resize(local_n); }
        if (need_f) { frame.fx.resize(local_n); frame.fy.resize(local_n); frame.fz.resize(local_n); }
        for (size_t k = 0; k < local_n; ++k) {
            const size_t gid = static_cast<size_t>(particles.global_id[k]);
            if (gid >= local_n) {
                throw std::runtime_error("Invalid global_id while gathering serial trajectory output");
            }
            frame.x[gid] = particles.x[k]; frame.y[gid] = particles.y[k]; frame.z[gid] = particles.z[k];
            frame.type[gid] = particles.type[k];
            frame.element[gid] = system->has_topology()
                ? system->get_topology().get_atom_element(static_cast<int>(gid)) : "C";
            if (need_v) { frame.vx[gid] = particles.vx[k]; frame.vy[gid] = particles.vy[k]; frame.vz[gid] = particles.vz[k]; }
            if (need_f) { frame.fx[gid] = particles.fx[k]; frame.fy[gid] = particles.fy[k]; frame.fz[gid] = particles.fz[k]; }
        }
        return;
    }

    const int local_count = static_cast<int>(local_n);
    std::vector<int> counts, displs;
    int total = 0;
    if (rank == 0) counts.resize(size);
    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs.resize(size);
        for (int r = 0; r < size; ++r) { displs[r] = total; total += counts[r]; }
    }

    std::vector<int> local_gid(particles.global_id.begin(), particles.global_id.begin() + local_n);
    std::vector<int> local_type(particles.type.begin(), particles.type.begin() + local_n);
    std::vector<int> gathered_gid(rank == 0 ? total : 0);
    std::vector<int> gathered_type(rank == 0 ? total : 0);
    MPI_Gatherv(local_gid.data(), local_count, MPI_INT,
                rank == 0 ? gathered_gid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_type.data(), local_count, MPI_INT,
                rank == 0 ? gathered_type.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    // MPI_DOUBLE assumes Real == double, true everywhere in this codebase.
    auto gather_field = [&](const Real* local_ptr, std::vector<Real>& scratch) {
        scratch.resize(rank == 0 ? total : 0);
        MPI_Gatherv(local_ptr, local_count, MPI_DOUBLE,
                    rank == 0 ? scratch.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    };

    std::vector<Real> tmp_x, tmp_y, tmp_z;
    gather_field(particles.x.data(), tmp_x);
    gather_field(particles.y.data(), tmp_y);
    gather_field(particles.z.data(), tmp_z);

    std::vector<Real> tmp_vx, tmp_vy, tmp_vz, tmp_fx, tmp_fy, tmp_fz;
    if (need_v) {
        gather_field(particles.vx.data(), tmp_vx);
        gather_field(particles.vy.data(), tmp_vy);
        gather_field(particles.vz.data(), tmp_vz);
    }
    if (need_f) {
        gather_field(particles.fx.data(), tmp_fx);
        gather_field(particles.fy.data(), tmp_fy);
        gather_field(particles.fz.data(), tmp_fz);
    }

    if (rank != 0) return;

    frame.x.resize(total); frame.y.resize(total); frame.z.resize(total);
    frame.type.resize(total); frame.element.resize(total);
    if (need_v) { frame.vx.resize(total); frame.vy.resize(total); frame.vz.resize(total); }
    if (need_f) { frame.fx.resize(total); frame.fy.resize(total); frame.fz.resize(total); }

    for (int k = 0; k < total; ++k) {
        const int gid = gathered_gid[k];
        if (gid < 0 || gid >= total) {
            throw std::runtime_error("Invalid global_id while gathering serial trajectory output");
        }
        frame.x[gid] = tmp_x[k]; frame.y[gid] = tmp_y[k]; frame.z[gid] = tmp_z[k];
        frame.type[gid] = gathered_type[k];
        if (need_v) { frame.vx[gid] = tmp_vx[k]; frame.vy[gid] = tmp_vy[k]; frame.vz[gid] = tmp_vz[k]; }
        if (need_f) { frame.fx[gid] = tmp_fx[k]; frame.fy[gid] = tmp_fy[k]; frame.fz[gid] = tmp_fz[k]; }
    }
    for (int gid = 0; gid < total; ++gid) {
        frame.element[gid] = system->has_topology()
            ? system->get_topology().get_atom_element(gid) : "C";
    }
}

// Collective: called by every rank (via output_results()), because the
// serial-format gather above, the ADIOS2 BP writer, and the H5MD writer
// below all need every rank's participation.
void MDSimulation::write_trajectory(int step) {
    if (trajectory_formats_.empty()) return;

    const bool need_serial = trajectory_formats_.count("xyz") || trajectory_formats_.count("dump")
        || trajectory_formats_.count("dcd");

    if (need_serial) {
        SerialTrajectoryFrame frame;
        gather_particles_for_serial_output(frame);

        if (system->get_rank() == 0) {
            if (trajectory_formats_.count("xyz"))  write_xyz_trajectory(step, frame);
            if (trajectory_formats_.count("dump")) write_lammps_dump(step, frame);
            if (trajectory_formats_.count("dcd"))  write_dcd_frame(step, frame);
        }
    }

#ifdef USE_ADIOS2
    if (trajectory_formats_.count("bp")) {
        write_adios2_frame(step);
    }
#endif

#ifdef USE_HDF5
    if (trajectory_formats_.count("h5md")) {
        write_h5md_frame(step);
    }
#endif
}

void MDSimulation::write_xyz_trajectory(int step, const SerialTrajectoryFrame& frame) {
    std::string filename = output_prefix + "_" + std::to_string(step) + ".xyz";
    std::ofstream traj_file(filename);

    if (!traj_file.is_open()) {
        std::cerr << "Warning: Cannot open trajectory file " << filename << std::endl;
        return;
    }

    const size_t n = frame.x.size();

    // XYZ format header
    traj_file << n << "\n";
    traj_file << "Step " << step << " Time " << std::fixed << std::setprecision(3)
              << step * dt << " fs";

    // Add energy information to comment line
    if (force_calc) {
        Real ke = force_calc->get_kinetic_energy() * mvv2e;
        Real pe = cached_potential_energy;
        traj_file << " KE=" << std::setprecision(6) << ke
                  << " PE=" << pe << " Total=" << (ke + pe);
    }
    traj_file << "\n";

    // Particle coordinates
    for (size_t i = 0; i < n; i++) {
        traj_file << frame.element[i] << " "
                  << std::fixed << std::setprecision(6)
                  << frame.x[i] << " "
                  << frame.y[i] << " "
                  << frame.z[i];

        // Optionally include velocities and forces
        if (trajectory_include_velocities) {
            traj_file << " " << frame.vx[i] << " "
                      << frame.vy[i] << " " << frame.vz[i];
        }

        if (trajectory_include_forces) {
            traj_file << " " << frame.fx[i] << " "
                      << frame.fy[i] << " " << frame.fz[i];
        }

        traj_file << "\n";
    }

    traj_file.close();
}

void MDSimulation::write_lammps_dump(int step, const SerialTrajectoryFrame& frame) {
    std::string filename = output_prefix + "_dump." + std::to_string(step);
    std::ofstream dump_file(filename);

    if (!dump_file.is_open()) {
        std::cerr << "Warning: Cannot open dump file " << filename << std::endl;
        return;
    }

    const size_t n = frame.x.size();
    SimulationBox& box = system->get_box();

    // LAMMPS dump format header
    dump_file << "ITEM: TIMESTEP\n" << step << "\n";
    dump_file << "ITEM: NUMBER OF ATOMS\n" << n << "\n";

    // Box bounds (assuming orthorhombic box)
    dump_file << "ITEM: BOX BOUNDS pp pp pp\n";
    dump_file << "0.0 " << std::fixed << std::setprecision(6) << box.lx << "\n";
    dump_file << "0.0 " << box.ly << "\n";
    dump_file << "0.0 " << box.lz << "\n";

    // Atom data header
    dump_file << "ITEM: ATOMS id type x y z";
    if (trajectory_include_velocities) {
        dump_file << " vx vy vz";
    }
    if (trajectory_include_forces) {
        dump_file << " fx fy fz";
    }
    dump_file << "\n";

    // Atom data
    for (size_t i = 0; i < n; i++) {
        dump_file << (i + 1) << " " << frame.type[i] << " "
                  << std::fixed << std::setprecision(6)
                  << frame.x[i] << " "
                  << frame.y[i] << " "
                  << frame.z[i];

        if (trajectory_include_velocities) {
            dump_file << " " << frame.vx[i] << " "
                      << frame.vy[i] << " " << frame.vz[i];
        }

        if (trajectory_include_forces) {
            dump_file << " " << frame.fx[i] << " "
                      << frame.fy[i] << " " << frame.fz[i];
        }

        dump_file << "\n";
    }

    dump_file.close();
}

// CHARMM-format DCD header (content length 84 bytes, matching the layout
// used by VMD's dcdplugin.c / CHARMM / NAMD -- the reference implementation
// most DCD readers/writers mirror). After the 4-byte "CORD" tag, ICNTRL(1..20)
// as 32-bit ints:
//   1 NSET (frame count; written as 0 here and patched by
//     finalize_trajectory_writers() once the true count is known),
//   2 ISTART, 3 NSAVC, 4 NSTEP (cosmetic), 5-8 unused,
//   9 NAMNF (fixed atoms, unsupported here so 0),
//   10 DELTA (timestep, as a float32 bit pattern, in AKMA time units:
//      1 AKMA = 48.888211 fs),
//   11 with-unit-cell flag (1: a 6-double cell record precedes coordinates
//      in every frame), 12-19 unused, 20 VERSION (24 = "charmm format",
//      which is what tells readers DELTA is float32 rather than float64).
void MDSimulation::write_dcd_header(int32_t natom_global) {
    const int32_t reclen1 = 84;
    dcd_file_.write(reinterpret_cast<const char*>(&reclen1), 4);
    dcd_file_.write("CORD", 4);

    int32_t icntrl[20] = {0};
    icntrl[2] = std::max(1, trajectory_output_freq);  // NSAVC
    icntrl[3] = total_steps;                          // NSTEP (cosmetic)
    const float delta_akma = static_cast<float>(dt / 48.888211);
    std::memcpy(&icntrl[9], &delta_akma, sizeof(float));
    icntrl[10] = 1;                                   // unit cell record present
    icntrl[19] = 24;                                  // VERSION (charmm format)

    dcd_nset_patch_pos_ = dcd_file_.tellp();  // icntrl[0] == NSET, patched at close
    dcd_file_.write(reinterpret_cast<const char*>(icntrl), sizeof(icntrl));
    dcd_file_.write(reinterpret_cast<const char*>(&reclen1), 4);

    // Title record (1 line, 80 chars).
    std::string title = "Created by DeepScaleMD";
    title.resize(80, ' ');
    const int32_t ntitle = 1;
    const int32_t title_reclen = static_cast<int32_t>(sizeof(int32_t) + 80);
    dcd_file_.write(reinterpret_cast<const char*>(&title_reclen), 4);
    dcd_file_.write(reinterpret_cast<const char*>(&ntitle), 4);
    dcd_file_.write(title.data(), 80);
    dcd_file_.write(reinterpret_cast<const char*>(&title_reclen), 4);

    // NATOM record.
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(&natom_global),
                      static_cast<int32_t>(sizeof(int32_t)));
}

// DCD is a positions-only format by convention (readers assume the per-frame
// records are X/Y/Z); trajectory_include_velocities/forces have no effect
// here -- use xyz or bp for velocity/force output.
void MDSimulation::write_dcd_frame(int step, const SerialTrajectoryFrame& frame) {
    (void)step;
    if (!dcd_file_.is_open()) return;

    const SimulationBox& box = system->get_box();
    // Unit cell record: {A, gamma, B, beta, alpha, C}. Angles are stored as
    // their cosine (0.0 == 90 degrees) -- the convention both legacy and
    // modern DCD readers interpret correctly for an orthorhombic box
    // (DeepScaleMD has no triclinic tilt).
    double cell[6] = {box.lx, 0.0, box.ly, 0.0, 0.0, box.lz};
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(cell), static_cast<int32_t>(sizeof(cell)));

    const int32_t n = static_cast<int32_t>(frame.x.size());
    std::vector<float> buf(n);

    for (int32_t i = 0; i < n; ++i) buf[i] = static_cast<float>(frame.x[i]);
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(buf.data()), n * static_cast<int32_t>(sizeof(float)));
    for (int32_t i = 0; i < n; ++i) buf[i] = static_cast<float>(frame.y[i]);
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(buf.data()), n * static_cast<int32_t>(sizeof(float)));
    for (int32_t i = 0; i < n; ++i) buf[i] = static_cast<float>(frame.z[i]);
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(buf.data()), n * static_cast<int32_t>(sizeof(float)));

    ++dcd_frames_written_;
}

void MDSimulation::initialize_trajectory_writers() {
    if (trajectory_formats_.empty()) return;

    if (trajectory_formats_.count("dcd") || trajectory_formats_.count("bp")
        || trajectory_formats_.count("h5md")) {
        // Global atom count is constant for the run; every rank must take
        // part in this reduce even though only rank 0 (DCD) or the ADIOS2/
        // H5MD attribute (all ranks, but from identical data) consume it.
        int local_n = static_cast<int>(system->get_local_particles());
        int global_n = 0;
        MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        trajectory_natoms_global_cached_ = global_n;
    }

    if (trajectory_formats_.count("dcd") && system->get_rank() == 0) {
        const std::string filename = output_prefix + ".dcd";
        dcd_file_.open(filename, std::ios::binary | std::ios::trunc);
        if (!dcd_file_.is_open()) {
            throw std::runtime_error("Cannot open DCD trajectory file: " + filename);
        }
        write_dcd_header(trajectory_natoms_global_cached_);
        dcd_frames_written_ = 0;
    }

#ifdef USE_ADIOS2
    if (trajectory_formats_.count("bp")) {
        initialize_adios2_writer();
    }
#endif

#ifdef USE_HDF5
    if (trajectory_formats_.count("h5md")) {
        initialize_h5md_writer();
    }
#endif

    trajectory_writers_initialized_ = true;
}

void MDSimulation::finalize_trajectory_writers() {
    if (!trajectory_writers_initialized_) return;

    if (dcd_file_.is_open()) {
        // Patch NSET now that the true frame count is known -- the standard
        // way MD codes finalize an incrementally written DCD file without
        // knowing the frame count up front (NAMD/CHARMM do the same).
        dcd_file_.seekp(dcd_nset_patch_pos_);
        const int32_t nset = dcd_frames_written_;
        dcd_file_.write(reinterpret_cast<const char*>(&nset), sizeof(nset));
        dcd_file_.close();
    }

#ifdef USE_ADIOS2
    if (trajectory_formats_.count("bp")) {
        finalize_adios2_writer();
    }
#endif

#ifdef USE_HDF5
    if (trajectory_formats_.count("h5md")) {
        finalize_h5md_writer();
    }
#endif
}

#ifdef USE_ADIOS2
// ==================== ADIOS2 BP parallel trajectory writer ====================
// Genuinely parallel: every rank writes only the particles it currently owns
// as an independent "local array" block (Shape left empty -- ADIOS2's local
// array/local value idiom), tagged with global_id. There is no gather, no
// rank-0 funnel, and no cross-rank collective metadata step per array (the
// BP5 engine's aggregator-based writers merge per-rank blocks into the
// dataset without a global lock/hyperslab-style coordination step). Reading
// back a consistent global-order trajectory means sorting each step's blocks
// by their global_id array -- ordinary domain-decomposed-code practice for
// ADIOS2, not something DeepScaleMD needs to resolve at write time.
//
// Local block sizes (this rank's particle count) can change every step under
// migration, so the Count of every local array is refreshed via
// SetSelection() immediately before each Put() rather than fixed once at
// DefineVariable() time.
void MDSimulation::initialize_adios2_writer() {
    adios2_instance_ = adios2::ADIOS(MPI_COMM_WORLD);
    adios2_io_ = adios2_instance_.DeclareIO("DeepScaleMD_Trajectory");
    adios2_io_.SetEngine("BP5");

    const size_t local_n = system->get_local_particles();

    adios2_var_global_id_ = adios2_io_.DefineVariable<int>("global_id", {}, {}, {local_n});
    adios2_var_x_ = adios2_io_.DefineVariable<Real>("x", {}, {}, {local_n});
    adios2_var_y_ = adios2_io_.DefineVariable<Real>("y", {}, {}, {local_n});
    adios2_var_z_ = adios2_io_.DefineVariable<Real>("z", {}, {}, {local_n});
    if (trajectory_include_velocities) {
        adios2_var_vx_ = adios2_io_.DefineVariable<Real>("vx", {}, {}, {local_n});
        adios2_var_vy_ = adios2_io_.DefineVariable<Real>("vy", {}, {}, {local_n});
        adios2_var_vz_ = adios2_io_.DefineVariable<Real>("vz", {}, {}, {local_n});
    }
    if (trajectory_include_forces) {
        adios2_var_fx_ = adios2_io_.DefineVariable<Real>("fx", {}, {}, {local_n});
        adios2_var_fy_ = adios2_io_.DefineVariable<Real>("fy", {}, {}, {local_n});
        adios2_var_fz_ = adios2_io_.DefineVariable<Real>("fz", {}, {}, {local_n});
    }

    // "Local value" scalars (no shape): every rank provides its own copy
    // each step. box/step/time are tiny and identical across ranks -- the
    // redundancy costs nothing at BP's typical particle-array data volumes,
    // and avoids any ambiguity around single-writer global-scalar semantics.
    adios2_var_step_ = adios2_io_.DefineVariable<int>("step");
    adios2_var_time_ = adios2_io_.DefineVariable<Real>("time_fs");
    adios2_var_box_ = adios2_io_.DefineVariable<Real>("box", {}, {}, {3});

    adios2_io_.DefineAttribute<int>("natoms_global", trajectory_natoms_global_cached_);

    adios2_engine_ = adios2_io_.Open(output_prefix + ".bp", adios2::Mode::Write);
    adios2_active_ = true;
}

void MDSimulation::write_adios2_frame(int step) {
    const size_t local_n = system->get_local_particles();
    ParticleData& particles = system->get_particles();

    adios2_engine_.BeginStep();

    adios2_var_global_id_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_global_id_, particles.global_id.data());

    adios2_var_x_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_x_, particles.x.data());
    adios2_var_y_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_y_, particles.y.data());
    adios2_var_z_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_z_, particles.z.data());

    if (trajectory_include_velocities) {
        adios2_var_vx_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_vx_, particles.vx.data());
        adios2_var_vy_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_vy_, particles.vy.data());
        adios2_var_vz_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_vz_, particles.vz.data());
    }
    if (trajectory_include_forces) {
        adios2_var_fx_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_fx_, particles.fx.data());
        adios2_var_fy_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_fy_, particles.fy.data());
        adios2_var_fz_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_fz_, particles.fz.data());
    }

    adios2_engine_.Put(adios2_var_step_, step);
    adios2_engine_.Put(adios2_var_time_, static_cast<Real>(step) * dt);

    std::array<Real, 3> box_vals = {system->get_box().lx, system->get_box().ly, system->get_box().lz};
    adios2_var_box_.SetSelection({{}, {3}});
    adios2_engine_.Put(adios2_var_box_, box_vals.data());

    adios2_engine_.EndStep();
}

void MDSimulation::finalize_adios2_writer() {
    if (adios2_active_) {
        adios2_engine_.Close();
        adios2_active_ = false;
    }
}
#endif // USE_ADIOS2

#ifdef USE_HDF5
// ==================== H5MD parallel trajectory writer ====================
// Same no-gather design as the ADIOS2 writer above, expressed against the
// H5MD (https://www.h5md.org) schema so the trajectory is directly
// readable by h5py/MDAnalysis/VMD's h5md plugin without a custom reader.
//
// Every rank writes only the particles it currently owns, as a hyperslab
// of a dataset shared by the whole run (shape (n_frames, N_global, ...)).
// Per-rank write offsets along the particle axis are NOT stable across
// steps -- dsDecompX migrates particles between ranks -- so every step
// recomputes this rank's offset with an MPI_Exscan (exclusive prefix sum)
// over local particle counts, and every frame also writes an "id" dataset
// recording each row's global_id so a reader can restore atom-index order
// (the same problem gather_particles_for_serial_output() solves for the
// serial writers above, solved here without ever collecting onto rank 0).
//
// Dataset extension (H5Dset_extent) is a collective HDF5 metadata call and
// must be issued identically by every rank. The big per-particle arrays
// (position/velocity/force/id/species) are then written with disjoint,
// per-rank hyperslabs under a COLLECTIVE transfer property list, which lets
// the MPI-IO layer merge them into large, stripe-friendly requests -- this
// is what makes the writer scale on Lustre/GPFS instead of serializing
// through one rank the way the xyz/dump/dcd writers do. The handful of
// tiny, run-wide scalars (step/time/box edges) are instead written by rank
// 0 alone under the default INDEPENDENT transfer mode: every rank holds an
// identical value, so nothing is lost by having only one rank issue the
// write, and it avoids the undefined behavior of multiple ranks
// collectively writing overlapping hyperslabs.

namespace {

hid_t h5md_create_extendible_dataset(hid_t loc, const char* name, hid_t type,
                                      int rank, const hsize_t* dims, const hsize_t* chunk_dims) {
    std::vector<hsize_t> maxdims(dims, dims + rank);
    maxdims[0] = H5S_UNLIMITED;
    hid_t space = H5Screate_simple(rank, dims, maxdims.data());
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, rank, chunk_dims);
    hid_t dset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(space);
    return dset;
}

void h5md_write_string_attr(hid_t loc, const char* name, const std::string& value) {
    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, value.size() + 1);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, strtype, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, strtype, value.c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strtype);
}

void h5md_extend(hid_t dset, std::initializer_list<hsize_t> dims) {
    std::vector<hsize_t> d(dims);
    H5Dset_extent(dset, d.data());
}

} // namespace

void MDSimulation::initialize_h5md_writer() {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    const std::string filename = output_prefix + ".h5";
    h5md_file_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (h5md_file_ < 0) {
        throw std::runtime_error("Cannot open H5MD trajectory file: " + filename);
    }

    hid_t h5md_group = H5Gcreate2(h5md_file_, "/h5md", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        int version[2] = {1, 1};
        hsize_t dims2[1] = {2};
        hid_t space = H5Screate_simple(1, dims2, nullptr);
        hid_t attr = H5Acreate2(h5md_group, "version", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, version);
        H5Aclose(attr);
        H5Sclose(space);
    }
    hid_t creator_group = H5Gcreate2(h5md_group, "creator", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5md_write_string_attr(creator_group, "name", "DeepScaleMD");
    H5Gclose(creator_group);
    hid_t author_group = H5Gcreate2(h5md_group, "author", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const char* user_env = std::getenv("USER");
    if (!user_env) user_env = std::getenv("USERNAME");
    h5md_write_string_attr(author_group, "name", user_env ? user_env : "unknown");
    H5Gclose(author_group);
    H5Gclose(h5md_group);

    hid_t particles_group = H5Gcreate2(h5md_file_, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t all_group = H5Gcreate2(particles_group, "all", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t box_group = H5Gcreate2(all_group, "box", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        int dim = 3;
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(box_group, "dimension", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &dim);
        H5Aclose(attr);
        H5Sclose(space);
    }
    {
        hid_t strtype = H5Tcopy(H5T_C_S1);
        H5Tset_size(strtype, 9);  // "periodic" + NUL
        hsize_t dims3[1] = {3};
        hid_t space = H5Screate_simple(1, dims3, nullptr);
        hid_t attr = H5Acreate2(box_group, "boundary", strtype, space, H5P_DEFAULT, H5P_DEFAULT);
        char boundary[3][9] = {"periodic", "periodic", "periodic"};
        H5Awrite(attr, strtype, boundary);
        H5Aclose(attr);
        H5Sclose(space);
        H5Tclose(strtype);
    }
    {
        hid_t edges_group = H5Gcreate2(box_group, "edges", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {0, 3};
        hsize_t chunk[2] = {1, 3};
        h5md_box_edges_ = h5md_create_extendible_dataset(edges_group, "value", H5T_NATIVE_DOUBLE, 2, dims, chunk);
        H5Gclose(edges_group);
    }
    H5Gclose(box_group);

    const hsize_t n_global = static_cast<hsize_t>(trajectory_natoms_global_cached_);

    {
        hid_t id_group = H5Gcreate2(all_group, "id", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {0, n_global};
        hsize_t chunk[2] = {1, n_global};
        h5md_id_value_ = h5md_create_extendible_dataset(id_group, "value", H5T_NATIVE_INT, 2, dims, chunk);
        H5Gclose(id_group);
    }
    {
        hid_t species_group = H5Gcreate2(all_group, "species", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {0, n_global};
        hsize_t chunk[2] = {1, n_global};
        h5md_species_value_ = h5md_create_extendible_dataset(species_group, "value", H5T_NATIVE_INT, 2, dims, chunk);
        H5Gclose(species_group);
    }
    {
        hid_t position_group = H5Gcreate2(all_group, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims1[1] = {0};
        hsize_t chunk1[1] = {1};
        h5md_position_step_ = h5md_create_extendible_dataset(position_group, "step", H5T_NATIVE_INT, 1, dims1, chunk1);
        h5md_position_time_ = h5md_create_extendible_dataset(position_group, "time", H5T_NATIVE_DOUBLE, 1, dims1, chunk1);
        hsize_t dims3[3] = {0, n_global, 3};
        hsize_t chunk3[3] = {1, n_global, 3};
        h5md_position_value_ = h5md_create_extendible_dataset(position_group, "value", H5T_NATIVE_DOUBLE, 3, dims3, chunk3);
        H5Gclose(position_group);
    }
    if (trajectory_include_velocities) {
        hid_t velocity_group = H5Gcreate2(all_group, "velocity", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims1[1] = {0};
        hsize_t chunk1[1] = {1};
        h5md_velocity_step_ = h5md_create_extendible_dataset(velocity_group, "step", H5T_NATIVE_INT, 1, dims1, chunk1);
        h5md_velocity_time_ = h5md_create_extendible_dataset(velocity_group, "time", H5T_NATIVE_DOUBLE, 1, dims1, chunk1);
        hsize_t dims3[3] = {0, n_global, 3};
        hsize_t chunk3[3] = {1, n_global, 3};
        h5md_velocity_value_ = h5md_create_extendible_dataset(velocity_group, "value", H5T_NATIVE_DOUBLE, 3, dims3, chunk3);
        H5Gclose(velocity_group);
    }
    if (trajectory_include_forces) {
        hid_t force_group = H5Gcreate2(all_group, "force", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims1[1] = {0};
        hsize_t chunk1[1] = {1};
        h5md_force_step_ = h5md_create_extendible_dataset(force_group, "step", H5T_NATIVE_INT, 1, dims1, chunk1);
        h5md_force_time_ = h5md_create_extendible_dataset(force_group, "time", H5T_NATIVE_DOUBLE, 1, dims1, chunk1);
        hsize_t dims3[3] = {0, n_global, 3};
        hsize_t chunk3[3] = {1, n_global, 3};
        h5md_force_value_ = h5md_create_extendible_dataset(force_group, "value", H5T_NATIVE_DOUBLE, 3, dims3, chunk3);
        H5Gclose(force_group);
    }

    H5Gclose(all_group);
    H5Gclose(particles_group);

    h5md_frame_count_ = 0;
    h5md_active_ = true;
}

void MDSimulation::write_h5md_frame(int step) {
    if (!h5md_active_) return;

    int local_n = static_cast<int>(system->get_local_particles());
    ParticleData& particles = system->get_particles();
    const hsize_t n_global = static_cast<hsize_t>(trajectory_natoms_global_cached_);
    const hsize_t frame = static_cast<hsize_t>(h5md_frame_count_);
    const int rank = system->get_rank();

    // Exclusive prefix sum over this frame's local particle counts -- must
    // be recomputed every step because dsDecompX migration changes how many
    // particles (and which ones) each rank owns. Rank 0 receives no partial
    // sum from MPI_Exscan since nothing precedes it.
    int local_offset_int = 0;
    MPI_Exscan(&local_n, &local_offset_int, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) local_offset_int = 0;
    const hsize_t local_offset = static_cast<hsize_t>(local_offset_int);
    const hsize_t local_count = static_cast<hsize_t>(local_n);

    // Collective extend: every rank must call this identically before any
    // rank writes into the newly extended frame.
    h5md_extend(h5md_position_step_, {frame + 1});
    h5md_extend(h5md_position_time_, {frame + 1});
    h5md_extend(h5md_position_value_, {frame + 1, n_global, 3});
    h5md_extend(h5md_id_value_, {frame + 1, n_global});
    h5md_extend(h5md_species_value_, {frame + 1, n_global});
    h5md_extend(h5md_box_edges_, {frame + 1, 3});
    if (trajectory_include_velocities) {
        h5md_extend(h5md_velocity_step_, {frame + 1});
        h5md_extend(h5md_velocity_time_, {frame + 1});
        h5md_extend(h5md_velocity_value_, {frame + 1, n_global, 3});
    }
    if (trajectory_include_forces) {
        h5md_extend(h5md_force_step_, {frame + 1});
        h5md_extend(h5md_force_time_, {frame + 1});
        h5md_extend(h5md_force_value_, {frame + 1, n_global, 3});
    }

    hid_t dxpl_collective = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_collective, H5FD_MPIO_COLLECTIVE);

    // Writes this rank's disjoint [local_offset, local_offset+local_count)
    // slice of a (1, N_global, 3) frame slab. An empty selection (local_n
    // == 0, e.g. more ranks than particles in a region) is expressed as
    // H5S_SELECT_NONE rather than a zero-length hyperslab.
    auto write_vec3 = [&](hid_t dset, const std::vector<Real>& buf) {
        hid_t filespace = H5Dget_space(dset);
        if (local_count > 0) {
            hsize_t offset[3] = {frame, local_offset, 0};
            hsize_t count[3] = {1, local_count, 3};
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        } else {
            H5Sselect_none(filespace);
        }
        hsize_t mem_dims[3] = {1, local_count, 3};
        hid_t memspace = H5Screate_simple(3, mem_dims, nullptr);
        if (local_count == 0) H5Sselect_none(memspace);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_collective, buf.data());
        H5Sclose(memspace);
        H5Sclose(filespace);
    };
    auto write_ids = [&](hid_t dset, const int* data) {
        hid_t filespace = H5Dget_space(dset);
        if (local_count > 0) {
            hsize_t offset[2] = {frame, local_offset};
            hsize_t count[2] = {1, local_count};
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        } else {
            H5Sselect_none(filespace);
        }
        hsize_t mem_dims[2] = {1, local_count};
        hid_t memspace = H5Screate_simple(2, mem_dims, nullptr);
        if (local_count == 0) H5Sselect_none(memspace);
        H5Dwrite(dset, H5T_NATIVE_INT, memspace, filespace, dxpl_collective, data);
        H5Sclose(memspace);
        H5Sclose(filespace);
    };
    auto pack_xyz = [&](const Real* a, const Real* b, const Real* c) {
        std::vector<Real> buf(static_cast<size_t>(local_n) * 3);
        for (int i = 0; i < local_n; ++i) {
            buf[3 * i + 0] = a[i]; buf[3 * i + 1] = b[i]; buf[3 * i + 2] = c[i];
        }
        return buf;
    };

    write_vec3(h5md_position_value_, pack_xyz(particles.x.data(), particles.y.data(), particles.z.data()));
    write_ids(h5md_id_value_, particles.global_id.data());
    write_ids(h5md_species_value_, particles.type.data());
    if (trajectory_include_velocities) {
        write_vec3(h5md_velocity_value_, pack_xyz(particles.vx.data(), particles.vy.data(), particles.vz.data()));
    }
    if (trajectory_include_forces) {
        write_vec3(h5md_force_value_, pack_xyz(particles.fx.data(), particles.fy.data(), particles.fz.data()));
    }

    H5Pclose(dxpl_collective);

    // Tiny, run-wide scalars: identical on every rank, so only rank 0 needs
    // to actually issue the write (default H5P_DEFAULT dxpl == independent
    // access -- see the file-header comment for why this beats a redundant
    // collective write to the same overlapping hyperslab).
    if (rank == 0) {
        auto write_scalar = [&](hid_t dset, hid_t type, const void* value) {
            hid_t filespace = H5Dget_space(dset);
            hsize_t offset[1] = {frame};
            hsize_t count[1] = {1};
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            hid_t memspace = H5Screate_simple(1, count, nullptr);
            H5Dwrite(dset, type, memspace, filespace, H5P_DEFAULT, value);
            H5Sclose(memspace);
            H5Sclose(filespace);
        };
        write_scalar(h5md_position_step_, H5T_NATIVE_INT, &step);
        const Real time_fs = static_cast<Real>(step) * dt;
        write_scalar(h5md_position_time_, H5T_NATIVE_DOUBLE, &time_fs);
        if (trajectory_include_velocities) {
            write_scalar(h5md_velocity_step_, H5T_NATIVE_INT, &step);
            write_scalar(h5md_velocity_time_, H5T_NATIVE_DOUBLE, &time_fs);
        }
        if (trajectory_include_forces) {
            write_scalar(h5md_force_step_, H5T_NATIVE_INT, &step);
            write_scalar(h5md_force_time_, H5T_NATIVE_DOUBLE, &time_fs);
        }

        Real box_vals[3] = {system->get_box().lx, system->get_box().ly, system->get_box().lz};
        hid_t filespace = H5Dget_space(h5md_box_edges_);
        hsize_t offset2[2] = {frame, 0};
        hsize_t count2[2] = {1, 3};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset2, nullptr, count2, nullptr);
        hid_t memspace = H5Screate_simple(2, count2, nullptr);
        H5Dwrite(h5md_box_edges_, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, box_vals);
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    ++h5md_frame_count_;
}

void MDSimulation::finalize_h5md_writer() {
    if (!h5md_active_) return;

    H5Dclose(h5md_position_step_);
    H5Dclose(h5md_position_time_);
    H5Dclose(h5md_position_value_);
    H5Dclose(h5md_id_value_);
    H5Dclose(h5md_species_value_);
    H5Dclose(h5md_box_edges_);
    if (trajectory_include_velocities) {
        H5Dclose(h5md_velocity_step_);
        H5Dclose(h5md_velocity_time_);
        H5Dclose(h5md_velocity_value_);
    }
    if (trajectory_include_forces) {
        H5Dclose(h5md_force_step_);
        H5Dclose(h5md_force_time_);
        H5Dclose(h5md_force_value_);
    }
    H5Fclose(h5md_file_);
    h5md_active_ = false;
}
#endif // USE_HDF5

// print_timing_report() is defined in timing_breakdown.cpp
