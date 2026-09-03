#include "md_simulation.h"
#include "neighbor_list.h"
#include "force_calculator_cuda.h"
#include "thermostat.h"
#include "barostat.h"
#include "halo_backend.h"
#include "comm_backend.h"
#include "simulation_backend_factory.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <set>

// ── C++ MLIP Runtime (mlip_inst.md architecture) ────────────────────────────
#include "mlip/runtime/mlip_runtime.h"
#include "mlip/mlip_types.h"
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
        // Transitional spelling for existing input files.
        if (component == "mlip_runtime") component = "libtorch";
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
}  // namespace


MDSimulation::MDSimulation() : 
    cuda_calc_cached_(nullptr),
    using_cuda_backend_cached_(false),
    mlip_spec_file_(""),
    coordinate_file(""),
    topology_file(""),
    target_temperature(0.0),
    output_prefix("md_output"),
    trajectory_output_freq(100),
    energy_output_freq(10),
    cutoff_distance(12.0),
    neighbor_list_type_str("verlet_cell_list"),
    neighbor_list_tile_size(256),
    neighbor_list_update_freq(10),  // Default update frequency: every 10 steps
    verlet_skin(2.0),               // Default Verlet skin distance
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
    timing_enabled(false),
    timing_every_step(false) {
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
    // Load configuration
    system->load_config(config_file);
    
    // Parse simulation parameters from config file
    parse_simulation_config(config_file);

    // Load coordinate file from configuration first
    system->load_coordinates(coordinate_file);
    
    // Load topology file from configuration and assign properties to particles
    // MUST be done BEFORE velocity initialization to ensure masses are set
    if (!topology_file.empty()) {
        system->load_topology(topology_file);
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
    cuda_calc_cached_ = dynamic_cast<ForceCalculatorCUDA*>(force_calc.get());
    using_cuda_backend_cached_ = (cuda_calc_cached_ != nullptr);

    const bool using_cuda_backend = is_using_cuda_backend();

    halo_backend_ = create_halo_backend(force_calculation_mode, force_calc.get());
    comm_backend_ = create_comm_backend();
#ifdef USE_CUDA
    if (cuda_calc_cached_) {
        cuda_calc_cached_->set_builtin_forces_enabled(enable_builtin_forces_);
    }
#endif

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
        if (!using_cuda_backend) {
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
    
    // Initialize neighbor list builder
    Real cutoff = cutoff_distance; // Use configured cutoff distance
    
    // Parse neighbor list type
    NeighborListType nlist_type = NeighborListType::VERLET_CELL_LIST; // Use verlet cell list by default
    if (neighbor_list_type_str == "morton_code") {
        nlist_type = NeighborListType::MORTON_Z_ORDER_CSR;
        if (system->get_rank() == 0) {
            std::cout << "Using Morton Z-order neighbor list (optimized for MPI/CUDA)" << std::endl;
        }
    } else if (neighbor_list_type_str == "verlet_cell_list") {
        nlist_type = NeighborListType::VERLET_CELL_LIST;
        if (system->get_rank() == 0) {
            std::cout << "Using Verlet-Cell hybrid neighbor list with skin distance: " << verlet_skin << std::endl;
        }
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown neighbor list type '" << neighbor_list_type_str 
                      << "', using verlet cell list (default)" << std::endl;
        }
        nlist_type = NeighborListType::VERLET_CELL_LIST;
    }
    
    // Set up neighbor list configuration based on type
    std::string nlist_config;
    if (nlist_type == NeighborListType::VERLET_CELL_LIST) {
        // For Verlet-Cell hybrid, use verlet_skin distance
        nlist_config = std::to_string(verlet_skin);
    } else {
        // For Morton Z-order, use verlet_skin for stability
        nlist_config = std::to_string(verlet_skin);
    }
    
    neighbor_builder = create_neighbor_list_builder(nlist_type, cutoff, nlist_config, system->get_rank());
    neighbor_builder->initialize();

    // Seed neighbor-list metadata for GPU-first builds.
    // CUDA neighbor build uses nlist.cutoff before any host-side build occurs.
    {
        NeighborList& nlist = system->get_neighbor_list();
        nlist.skin_distance = verlet_skin;
        nlist.cutoff = cutoff_distance + verlet_skin;
        nlist.needs_rebuild = true;
    }
    
    // Set update frequency information for proper logging
    VerletCellListNeighborBuilder* verlet_builder = dynamic_cast<VerletCellListNeighborBuilder*>(neighbor_builder.get());
    if (verlet_builder) {
        verlet_builder->set_update_info(neighbor_list_update_freq, 0);
        // Enable automatic rebuild check for better performance (LAMMPS-style)
        verlet_builder->enable_check(true);
        // CUDA path: use full-list (Newton OFF) so GPU kernel (which does not apply
        // Newton's 3rd law atomics) sees every i-j pair and produces correct forces.
        // CPU/SYCL paths keep half-list behavior.
        if (using_cuda_backend) {
            verlet_builder->set_use_half_list(false);
        }
    }
    MortonZOrderNeighborBuilder* morton_builder = dynamic_cast<MortonZOrderNeighborBuilder*>(neighbor_builder.get());
    if (morton_builder) {
        morton_builder->set_update_info(neighbor_list_update_freq, 0);
    }
    
    // Initialize components based on configuration
    initialize_integrator();
    initialize_thermostat();
    initialize_barostat();
}

bool MDSimulation::is_using_cuda_backend() const {
    return using_cuda_backend_cached_;
}

void MDSimulation::single_step() {
    // Performance timing (enabled via MD_TIMING=1 environment variable)
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
            cuda_calc->integrate_on_device(*system, dt, integrator_type);
            return;
#else
#endif
        }
        integrator->integrate(system->get_particles(), system->get_box(), dt, system->get_local_particles());
    });
}

void MDSimulation::handle_neighbor_phase(double& nlist_build_time, double& comm_time, double& migrate_time) {
    bool should_rebuild = (current_step % neighbor_list_update_freq == 0);
    migrate_time = 0.0;

    if (neighbor_rebuild_forced) {
        should_rebuild = true;
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
                "CUDA execution requires the dsDecomp GPU-resident halo backend; "
                "host ParticleData fallback is disabled");
        }

        if (!gpu_direct_halo_done) {
            should_rebuild = system->sync_distributed_particles(
                should_rebuild, &system->get_neighbor_list());
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
        handle_neighbor_rebuild_if_needed(should_rebuild);
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

        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->upload_forces_from_ml(
                system->get_particles(), system->get_num_particles());
#endif
        }

        cached_potential_energy = force_calc ? force_calc->get_potential_energy() : 0.0;
    }
}

void MDSimulation::reverse_halo_for_bonded_if_needed() {
    if (!comm_backend_->is_distributed()) {
        return;
    }

    const auto& cfg = system->get_force_manager().get_config();
    const bool has_bonded = cfg.enable_bonds || cfg.enable_angles || cfg.enable_dihedrals;
    if (!has_bonded) {
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
        cuda_calc->update_reference_positions_on_device(system->get_local_particles());
#else
        neighbor_builder->build(system->get_particles(), system->get_box(),
                                system->get_neighbor_list(), system->get_local_particles());
#endif
    } else {
        neighbor_builder->build(system->get_particles(), system->get_box(),
                                system->get_neighbor_list(), system->get_local_particles());
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
            return;
#else
#endif
        }
        integrator->complete_velocity_update(system->get_particles(), dt, system->get_local_particles());
    });
}

void MDSimulation::accumulate_step_timing(double integrate1_time, double nlist_build_time, double comm_time,
                                          double migrate_time,
                                          double force_time, double integrate2_time,
                                          double thermostat_time, double ke_time,
                                          double barostat_time, double modify_time,
                                          double total_time) {
    // Keep per-step phase timing in milliseconds for optional diagnostics.
    last_timing.domain_ms = 0.0;
    last_timing.nlist_build_ms = nlist_build_time * 1000.0;
    last_timing.comm_ms = comm_time * 1000.0;
    last_timing.migrate_ms = migrate_time * 1000.0;
    last_timing.force_ms = force_time * 1000.0;
    last_timing.integrate_ms = (integrate1_time + integrate2_time) * 1000.0;
    last_timing.thermostat_ms = thermostat_time * 1000.0;
    last_timing.ke_ms = ke_time * 1000.0;
    last_timing.barostat_ms = barostat_time * 1000.0;
    last_timing.total_ms = total_time * 1000.0;

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
        cuda_calc->update_reference_positions_on_device(system->get_local_particles());
#else
        neighbor_builder->build(system->get_particles(), system->get_box(),
                                system->get_neighbor_list(), system->get_local_particles());
#endif
    } else {
        neighbor_builder->build(system->get_particles(), system->get_box(),
                                system->get_neighbor_list(), system->get_local_particles());
    }

    ++neighbor_list_builds;
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


void MDSimulation::run() {
    timing_enabled = env_flag_enabled("MD_TIMING");
    timing_every_step = env_flag_enabled("MD_TIMING_EVERY_STEP");

   
    if (system->get_rank() == 0) {
        std::cout << "Starting MD simulation for " << total_steps << " steps" << std::endl;
        if (timing_enabled) {
            std::cout << "Per-step timing output enabled (MD_TIMING=1)" << std::endl;
            if (timing_every_step) {
                std::cout << "Timing print mode: every step (MD_TIMING_EVERY_STEP=1)" << std::endl;
            } else {
                std::cout << "Timing print mode: output steps only" << std::endl;
            }
        }
    }
    
    // Reset state flags
    current_step = 0;
    neighbor_rebuild_forced = false;
    
    // Prepare initial forces at t=0 before starting MD loop
    prepare_first_step();

    // Output initial state (step 0)
    output_results(0);
    
    // Main simulation loop - current_step is already 1 from initialization
    for (int step = 1; step <= total_steps; step++) {
        single_step();

        if (timing_every_step && should_print_timing_for_step(step)) {
            print_step_timing(step);
        }

        handle_step_output(step);
    }
    
    // Output neighbor list statistics at end
    if (system->get_rank() == 0) {
        std::cout << "Neighbor list builds = " << neighbor_list_builds << std::endl;
    }
}

void MDSimulation::print_step_timing(int step) {
    if (!timing_enabled || system->get_rank() != 0) {
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
              << " io_ms=" << last_timing.io_ms
              << std::endl;
}

bool MDSimulation::should_output_step(int step) const {
    return (output_freq > 0) && (step % output_freq == 0);
}

bool MDSimulation::needs_ke_update_for_step(int step) const {
    return should_output_step(step) || (step == total_steps);
}

bool MDSimulation::should_write_trajectory_step(int step) const {
    return (trajectory_output_freq > 0) && (step % trajectory_output_freq == 0);
}

bool MDSimulation::should_print_timing_for_step(int step) const {
    if (!timing_enabled) {
        return false;
    }
    if (timing_every_step) {
        return true;
    }
    return should_output_step(step);
}

void MDSimulation::handle_step_output(int step) {
    if (!should_output_step(step)) {
        return;
    }

    if (should_print_timing_for_step(step) && !timing_every_step) {
        print_step_timing(step);
    }

    // OPTIMIZATION: Removed MPI_Barrier before output.
    // The MPI_Allreduce calls in calculate_*_energy already synchronize enough.
    output_results(step);
}

void MDSimulation::compute_output_energies(const ParticleData& particles, OutputMetrics& metrics) {
    if (force_calc) {
        metrics.total_ke = force_calc->get_kinetic_energy() * mvv2e;

        if (enable_python_forces_ && !enable_builtin_forces_ && !enable_libtorch_forces_) {
            metrics.total_pe = cached_potential_energy;
        } else if (enable_builtin_forces_) {
            // Always report the force calculator's own accumulated PE. It
            // already sums every enabled contribution (bonds/angles/
            // dihedrals/LJ/electrostatics) consistently for CPU, SYCL and
            // CUDA. The previous CPU/SYCL branch called calculate_potential_energy(),
            // a separate LJ-only re-implementation that silently ignored
            // bonded and electrostatic energy -- see force_calc for the
            // authoritative computation.
            metrics.total_pe = force_calc->get_potential_energy();
        } else {
            metrics.total_pe = force_calc->get_potential_energy();
        }
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

    if (should_write_trajectory_step(step)) {
        write_trajectory(step);
    }
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

void MDSimulation::output_results(int step) {
    double output_time = measure_seconds([&]() {
        sync_output_state_to_host_if_needed();

        OutputMetrics metrics;
        ParticleData& particles = system->get_particles();

        compute_output_energies(particles, metrics);
        compute_output_force_speed_stats(particles, metrics);
        compute_output_thermodynamics(particles, metrics);

        emit_rank0_output(step, metrics);
    });

    last_timing.io_ms = output_time * 1000.0;
    cumulative_timing.output_time += output_time;
}

void MDSimulation::finalize() {
    // Print LAMMPS-style timing breakdown
    // CRITICAL: All ranks must call this function because it uses MPI_Reduce
    if (cumulative_timing.step_count > 0) {
        print_timing_breakdown();
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
    
    if (neighbor_builder) {
        neighbor_builder->finalize();
    }
    
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
        } else if (key == "coordinate_file") {
            coordinate_file = value;
            found_coordinate_file = true;
        } else if (key == "topology_file") {
            topology_file = value;
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
        } else if (key == "neighbor_list_type") {
            neighbor_list_type_str = value;
        } else if (key == "neighbor_list_update_freq") {
            neighbor_list_update_freq = std::stoi(value);
        } else if (key == "verlet_skin") {
            verlet_skin = std::stod(value);
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
    
    // Validate required parameters
    std::vector<std::string> missing_params;
    if (!found_coordinate_file) missing_params.push_back("coordinate_file");
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
    // "builtin, python, libtorch".  The legacy `mlip_runtime` token remains
    // an input-compatible alias for `libtorch`.
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
    if (neighbor_list_update_freq <= 0) {
        throw std::runtime_error("Invalid neighbor_list_update_freq: must be positive");
    }
    
    // Set the physical cutoff in the system for force calculations
    system->set_physical_cutoff(cutoff_distance);
    system->set_verlet_skin(verlet_skin);
    rebuild_output_info_items_cache();
    if (system->get_rank() == 0) {
        std::cout << "Configuration loaded successfully:" << std::endl;
        std::cout << "  Coordinate file: " << coordinate_file << std::endl;
        std::cout << "  Target temperature: " << target_temperature << " K" << std::endl;
        std::cout << "  Timestep: " << dt << " fs" << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;
        std::cout << "  Integrator: " << integrator_type << std::endl;
        std::cout << "  Thermostat: " << thermostat_type << std::endl;
        std::cout << "  Barostat: " << barostat_type << std::endl;
        std::cout << "  Box size: " << system->get_box().lx << " x " << system->get_box().ly << " x " << system->get_box().lz << std::endl;
        std::cout << "  Periodic BC: " << (system->get_box().periodic_x ? "T" : "F") 
                  << " " << (system->get_box().periodic_y ? "T" : "F") 
                  << " " << (system->get_box().periodic_z ? "T" : "F") << std::endl;
        std::cout << "  Cutoff distance: " << cutoff_distance << std::endl;
        std::cout << "  Neighbor list type: " << neighbor_list_type_str << std::endl;
        std::cout << "  Neighbor list update frequency: " << neighbor_list_update_freq << " steps" << std::endl;
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
void MDSimulation::write_trajectory(int step) {
    if (trajectory_format == "xyz") {
        write_xyz_trajectory(step);
    } else if (trajectory_format == "dump") {
        write_lammps_dump(step);
    } else if (trajectory_format == "both") {
        write_xyz_trajectory(step);
        write_lammps_dump(step);
    }
}

void MDSimulation::write_xyz_trajectory(int step) {
    if (system->get_rank() == 0) {
        std::string filename = output_prefix + "_" + std::to_string(step) + ".xyz";
        std::ofstream traj_file(filename);
        
        if (!traj_file.is_open()) {
            std::cerr << "Warning: Cannot open trajectory file " << filename << std::endl;
            return;
        }
        
        ParticleData& particles = system->get_particles();
        
        // XYZ format header
        traj_file << particles.num_particles << "\n";
        traj_file << "Step " << step << " Time " << std::fixed << std::setprecision(3) 
                  << step * dt << " fs";
        
        // Add energy information to comment line
        if (force_calc) {
            Real ke = force_calc->get_kinetic_energy();
            Real pe = force_calc->get_potential_energy();
            traj_file << " KE=" << std::setprecision(6) << ke 
                      << " PE=" << pe << " Total=" << (ke + pe);
        }
        traj_file << "\n";
        
        // Particle coordinates
        for (size_t i = 0; i < particles.num_particles; i++) {
            // Use atom type or default to "C"
            std::string atom_type = "C";  // Could be enhanced to read from topology
            
            traj_file << atom_type << " " 
                      << std::fixed << std::setprecision(6) 
                      << particles.x[i] << " " 
                      << particles.y[i] << " " 
                      << particles.z[i];
            
            // Optionally include velocities and forces
            if (trajectory_include_velocities) {
                traj_file << " " << particles.vx[i] << " " 
                          << particles.vy[i] << " " << particles.vz[i];
            }
            
            if (trajectory_include_forces) {
                traj_file << " " << particles.fx[i] << " " 
                          << particles.fy[i] << " " << particles.fz[i];
            }
            
            traj_file << "\n";
        }
        
        traj_file.close();
    }
}

void MDSimulation::write_lammps_dump(int step) {
    if (system->get_rank() == 0) {
        std::string filename = output_prefix + "_dump." + std::to_string(step);
        std::ofstream dump_file(filename);
        
        if (!dump_file.is_open()) {
            std::cerr << "Warning: Cannot open dump file " << filename << std::endl;
            return;
        }
        
        ParticleData& particles = system->get_particles();
        SimulationBox& box = system->get_box();
        
        // LAMMPS dump format header
        dump_file << "ITEM: TIMESTEP\n" << step << "\n";
        dump_file << "ITEM: NUMBER OF ATOMS\n" << particles.num_particles << "\n";
        
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
        for (size_t i = 0; i < particles.num_particles; i++) {
            dump_file << (i + 1) << " " << particles.type[i] << " "
                      << std::fixed << std::setprecision(6)
                      << particles.x[i] << " " 
                      << particles.y[i] << " " 
                      << particles.z[i];
            
            if (trajectory_include_velocities) {
                dump_file << " " << particles.vx[i] << " " 
                          << particles.vy[i] << " " << particles.vz[i];
            }
            
            if (trajectory_include_forces) {
                dump_file << " " << particles.fx[i] << " " 
                          << particles.fy[i] << " " << particles.fz[i];
            }
            
            dump_file << "\n";
        }
        
        dump_file.close();
    }
}

// print_timing_breakdown() is now defined in timing_breakdown.cpp
