#include <mpi.h>
#include "md_simulation.h"
#include "neighbor_list.h"
#include "force_calculator_cuda.h"
#include "force_calculator_cpu.h"
#include "thermostat.h"
#include "barostat.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>


MDSimulation::MDSimulation() : 
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
    timing_enabled(false) {
    const char* env = std::getenv("MD_TIMING");
    if (env && std::string(env) == "1") timing_enabled = true;
}

MDSimulation::~MDSimulation() = default;

void MDSimulation::initialize(const std::string& config_file) {
    // Initialize MPI
    system = std::make_unique<MDSystem>();
    system->initialize_mpi();
    
    if (system->get_rank() == 0) {
        std::cout << "Starting MD simulation initialization..." << std::endl;
    }
    
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
    
    // Initialize force calculator based on compile-time flags
#ifdef USE_CUDA
    try {
        force_calc = std::make_unique<ForceCalculatorCUDA>();
        force_calc->initialize();
        if (system->get_rank() == 0) {
            std::cout << "Using CUDA for force calculations" << std::endl;
        }
    } catch (const std::exception& e) {
        if (system->get_rank() == 0) {
            std::cout << "CUDA initialization failed: " << e.what() << std::endl;
            std::cout << "Falling back to CPU force calculations" << std::endl;
        }
        force_calc = std::make_unique<ForceCalculatorCPU>();
        force_calc->initialize();
    }
#elif defined(USE_SYCL)
    force_calc = std::make_unique<ForceCalculatorSYCL>();
    if (system->get_rank() == 0) {
        std::cout << "Using SYCL for force calculations" << std::endl;
    }
    force_calc->initialize();
#else
    // CPU fallback
    force_calc = std::make_unique<ForceCalculatorCPU>();
    if (system->get_rank() == 0) {
        std::cout << "Using CPU for force calculations" << std::endl;
    }
    force_calc->initialize();
#endif
    // Initialize Python interface for custom forces (only if python mode is enabled)
    if (system->get_rank() == 0 && force_calculation_mode == "python") {
        python_interface = std::make_unique<PythonForceInterface>();
        python_interface->initialize_python(python_force_path);
        if (system->get_rank() == 0) {
            std::cout << "Python force calculation interface initialized" << std::endl;
        }
    }
    
    // Initialize neighbor list builder
    Real cutoff = cutoff_distance; // Use configured cutoff distance
    
    // Parse neighbor list type
    NeighborListType nlist_type = NeighborListType::VERLET_CELL_LIST; // Use verlet cell list by default
    if (neighbor_list_type_str == "morton_z_order") {
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
    
    // Set update frequency information for proper logging
    VerletCellListNeighborBuilder* verlet_builder = dynamic_cast<VerletCellListNeighborBuilder*>(neighbor_builder.get());
    if (verlet_builder) {
        verlet_builder->set_update_info(neighbor_list_update_freq, 0);
        // Enable automatic rebuild check for better performance (LAMMPS-style)
        verlet_builder->enable_check(true);
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

void MDSimulation::single_step() {
    // Performance timing (enabled via MD_TIMING=1 environment variable)
    auto step_start = std::chrono::high_resolution_clock::now();
    
    double comm_time = 0.0;  // Track communication time

    // ==================== Neighbor List Management ====================
    auto nlist_start = std::chrono::high_resolution_clock::now();
    bool should_rebuild = false;
    
    if (neighbor_rebuild_forced) {
        should_rebuild = true;
        neighbor_rebuild_forced = false;  // Reset flag after use
        
    } else if (current_step % neighbor_list_update_freq == 0) {
        VerletCellListNeighborBuilder* verlet_builder = dynamic_cast<VerletCellListNeighborBuilder*>(neighbor_builder.get());
        if (verlet_builder) {
            // This call uses MPI_Allreduce internally, so all ranks get the same result
            should_rebuild = verlet_builder->needs_rebuild(system->get_particles(), system->get_box(), system->get_local_particles());
        } else {
            // Other neighbor list types: always rebuild at specified frequency
            should_rebuild = true;
        }
    }
    
    // Perform halo exchange ONLY when rebuild is needed (optimization)
    if (should_rebuild) {
#ifdef USE_MPI
        {
            auto comm_start = std::chrono::high_resolution_clock::now();
            system->perform_halo_exchange();
            system->assemble_ghost_particles();
            auto comm_end = std::chrono::high_resolution_clock::now();
            comm_time += std::chrono::duration<double>(comm_end - comm_start).count();
        }
#endif
        // build() contains MPI_Barrier and MPI_Bcast internally
        neighbor_builder->build(system->get_particles(), system->get_box(),
                                system->get_neighbor_list(), system->get_local_particles());
        neighbor_list_builds++;
    }
    
    auto nlist_end = std::chrono::high_resolution_clock::now();
    double nlist_time = std::chrono::duration<double>(nlist_end - nlist_start).count();
    
    
    auto integrate_start = std::chrono::high_resolution_clock::now();
    

    // Step 1: Position update and half-step velocity update
    // Uses F(t) computed in previous step (or initial force for first step)
    integrator->integrate(system->get_particles(), system->get_box(), dt, system->get_local_particles());
    
    auto integrate1_end = std::chrono::high_resolution_clock::now();
    double integrate1_time = std::chrono::duration<double>(integrate1_end - integrate_start).count();
    
#ifdef USE_MPI
    // After position update, perform halo exchange for force calculation
    // This is the MAIN halo exchange for every step
    {
        auto comm_start = std::chrono::high_resolution_clock::now();
        system->perform_halo_exchange();
        system->assemble_ghost_particles();  // Assemble ghost particles into ParticleData
        MPI_Barrier(MPI_COMM_WORLD);
        auto comm_end = std::chrono::high_resolution_clock::now();
        comm_time += std::chrono::duration<double>(comm_end - comm_start).count();
    }
#endif
    
    // ==================== Force Calculation at t+dt ====================
    auto force_start = std::chrono::high_resolution_clock::now();
    
    // Zero forces for ALL particles (local + ghost)
    zero_forces();

    // Step 2: Compute forces at new positions F(t+dt)
    if (force_calculation_mode == "python" && python_interface) {
        python_interface->compute_forces(system->get_particles(), system->get_ff_params(), 
                                        system->get_neighbor_list());
        cached_potential_energy = 0.0;
    } else if (force_calculation_mode == "builtin" && force_calc) {
        force_calc->compute_forces(*system);
    } else {
        if (force_calc) {
            force_calc->compute_forces(*system);
        }
    }

#ifdef USE_MPI
    // Perform reverse halo exchange to accumulate ghost forces back to local particles
    {
        auto comm_start = std::chrono::high_resolution_clock::now();
        system->perform_reverse_halo_exchange();
        MPI_Barrier(MPI_COMM_WORLD);
        auto comm_end = std::chrono::high_resolution_clock::now();
        comm_time += std::chrono::duration<double>(comm_end - comm_start).count();
    }
#endif
    
    auto force_end = std::chrono::high_resolution_clock::now();
    double force_time = std::chrono::duration<double>(force_end - force_start).count();
    
    auto integrate2_start = std::chrono::high_resolution_clock::now();
    
    // Step 3: Complete velocity update with new forces
    // Implementation varies by integrator type (VV, Leapfrog, r-RESPA)
    // VelocityVerlet: v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
    // Leapfrog: v(t+3dt/2) = v(t+dt/2) + a(t+dt)*dt
    integrator->complete_velocity_update(system->get_particles(), dt, system->get_local_particles());
    
    auto integrate2_end = std::chrono::high_resolution_clock::now();
    double integrate2_time = std::chrono::duration<double>(integrate2_end - integrate2_start).count();
    
    // ==================== Apply thermostat and barostat ====================
    auto modify_start = std::chrono::high_resolution_clock::now();
    if (thermostat) {
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        thermostat->apply(system->get_particles(), dt, system->get_local_particles());
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    
    // Update kinetic energy after velocity update and thermostat application
    if (force_calc) {
        force_calc->calculate_kinetic_energy(*system);
    }
    
    // Apply barostat
    if (barostat) {
        // Store old box dimensions to detect significant changes
        Real old_lx = system->get_box().lx;
        Real old_ly = system->get_box().ly;
        Real old_lz = system->get_box().lz;
        
        barostat->apply(system->get_particles(), system->get_box(), dt, system->get_local_particles());
        
        // CRITICAL FIX 3: Force neighbor rebuild if box changed significantly
        Real dlx = std::abs(system->get_box().lx - old_lx) / old_lx;
        Real dly = std::abs(system->get_box().ly - old_ly) / old_ly;
        Real dlz = std::abs(system->get_box().lz - old_lz) / old_lz;
        
        const Real box_change_threshold = 0.01;  // 1% relative change
        if (dlx > box_change_threshold || dly > box_change_threshold || dlz > box_change_threshold) {
            neighbor_rebuild_forced = true;  // Will trigger rebuild in next step
        }
    }
    auto modify_end = std::chrono::high_resolution_clock::now();
    double modify_time = std::chrono::duration<double>(modify_end - modify_start).count();
    
#ifdef USE_MPI
    // Remove ghost particles after force/thermo/barostat application to restore local-only arrays
    system->remove_ghost_particles();
#endif
    
    auto step_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(step_end - step_start).count();
    
    // Accumulate timing statistics
    cumulative_timing.pair_time += force_time;
    cumulative_timing.neigh_time += nlist_time;
    cumulative_timing.comm_time += comm_time;
    cumulative_timing.modify_time += (integrate1_time + integrate2_time + modify_time);
    cumulative_timing.total_time += total_time;
    cumulative_timing.step_count++;
        
    current_step++;
}

void MDSimulation::prepare_first_step() {

#ifdef USE_MPI
    // Initial halo exchange to populate ghost particles
    system->perform_halo_exchange();
    system->assemble_ghost_particles();
#endif
    
    // Build initial neighbor list at t=0 (with ghost particles)
    neighbor_builder->build(system->get_particles(), system->get_box(),
                            system->get_neighbor_list(), system->get_local_particles());
    neighbor_list_builds++;
    
    // Zero forces for ALL particles (local + ghost)
    zero_forces();
    
    // Compute initial forces F(t=0)
    if (force_calculation_mode == "python" && python_interface) {
        python_interface->compute_forces(system->get_particles(), system->get_ff_params(), 
                                        system->get_neighbor_list());
        cached_potential_energy = 0.0;
    } else if (force_calculation_mode == "builtin" && force_calc) {
        force_calc->compute_forces(*system);
    } else {
        if (force_calc) {
            force_calc->compute_forces(*system);
        }
    }
    
#ifdef USE_MPI
    // Accumulate ghost forces back to local particles
    system->perform_reverse_halo_exchange();
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    // Special initialization for Leapfrog integrator
    if (integrator_type == "leap_frog") {
        integrator->initialize_velocities(system->get_particles(), dt, system->get_local_particles());
    }
    
    // Calculate initial kinetic energy after velocity initialization
    if (force_calc) {
        force_calc->calculate_kinetic_energy(*system);
    }
    
#ifdef USE_MPI
    // Clean up ghost particles after initial force calculation
    system->remove_ghost_particles();
#endif

}

void MDSimulation::zero_forces() {
    // Zero forces for ALL particles (local + ghost)
    // This is necessary because force calculation writes to both local and ghost particle forces
    ParticleData& particles = system->get_particles();
    size_t total_count = system->get_total_particles();  // local + ghost
    
    for (size_t i = 0; i < total_count; i++) {
        particles.fx[i] = 0.0;
        particles.fy[i] = 0.0;
        particles.fz[i] = 0.0;
    }
}

void MDSimulation::calculate_potential_energy() {
    Real local_pe = 0.0;
    const NeighborList& nlist = system->get_neighbor_list();
    const ForceFieldParams& ff_params = system->get_ff_params();
    const SimulationBox& box = system->get_box();
    ParticleData& particles = system->get_particles();
    
#ifdef USE_MPI
    // In MPI mode, only calculate PE for local particles to avoid double counting
    size_t num_local = system->get_local_particles();
#else
    // In serial mode, calculate for all particles
    size_t num_local = particles.num_particles;
#endif
    
    // Simple LJ potential energy calculation with PBC
    if (!ff_params.epsilon.empty() && !ff_params.sigma.empty()) {
        for (size_t i = 0; i < num_local; i++) {
            size_t start = (i == 0) ? 0 : nlist.num_neighbors[i-1];
            size_t end = nlist.num_neighbors[i];
            
            for (size_t idx = start; idx < end; idx++) {
                size_t j = nlist.neighbors[idx];
                
                // In MPI mode: only count pairs where i is local
                // Count each pair only once (i < j for local-local, all for local-ghost)
#ifdef USE_MPI
                if (j < num_local && i >= j) continue; // Skip local-local pairs to avoid double counting
#else
                if (i >= j) continue; // In serial mode, avoid double counting
#endif
                
                Real dx = particles.x[i] - particles.x[j];
                Real dy = particles.y[i] - particles.y[j];
                Real dz = particles.z[i] - particles.z[j];
                
                // Apply minimum image convention (periodic boundary conditions)
                box.apply_minimum_image(dx, dy, dz);
                
                Real r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 >= cutoff_distance * cutoff_distance || r2 < 1e-10) continue;
                
                // Use pre-computed mixing rules from lookup table (consistent with force calculation)
                int type_i = particles.type[i];
                int type_j = particles.type[j];
                Real epsilon = ff_params.epsilon_mixed[type_i][type_j];
                Real sigma = ff_params.sigma_mixed[type_i][type_j];
                
                Real sig2 = sigma * sigma;
                Real sig6 = sig2 * sig2 * sig2;
                Real r2inv = 1.0 / r2;
                Real r6inv = r2inv * r2inv * r2inv;
                Real sig6_r6 = sig6 * r6inv;
                Real sig12_r12 = sig6_r6 * sig6_r6;
                
                local_pe += 4.0 * epsilon * (sig12_r12 - sig6_r6);
            }
        }
    }
    
#ifdef USE_MPI
    // Sum potential energy across all ranks
    MPI_Allreduce(&local_pe, &cached_potential_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    cached_potential_energy = local_pe;
#endif
}

void MDSimulation::run() {
   
    if (system->get_rank() == 0) {
        std::cout << "Starting MD simulation for " << total_steps << " steps" << std::endl;
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
        
        // Output results based on output_freq
        if (step % output_freq == 0) {
#ifdef USE_MPI
            // Synchronize before output to ensure consistent state across ranks
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            output_results(step);  // Pass step number
        }
    }
    
    // Output neighbor list statistics at end
    if (system->get_rank() == 0) {
        std::cout << "Neighbor list builds = " << neighbor_list_builds << std::endl;
    }
}

void MDSimulation::output_results(int step) {
    // Track output timing
    auto output_start = std::chrono::high_resolution_clock::now();
    
    // Calculate all possible values
    Real total_ke = 0.0;
    Real max_speed = 0.0;
    Real total_speed = 0.0;
    Real total_pe = 0.0;  // Potential Energy
    Real max_force = 0.0; // Maximum force magnitude
    ParticleData& particles = system->get_particles();
    
    if (force_calc) {
        total_ke = force_calc->get_kinetic_energy();
        total_ke *= mvv2e;

        total_pe = force_calc->get_potential_energy();
    } else {
        // Fallback: calculate KE manually if no force calculator
        for (size_t i = 0; i < particles.num_particles; i++) {
            Real v2 = particles.vx[i] * particles.vx[i] + 
                      particles.vy[i] * particles.vy[i] + 
                      particles.vz[i] * particles.vz[i];
            total_ke += 0.5 * particles.mass[i] * v2;
        }
        total_ke *= mvv2e;
        total_pe = cached_potential_energy; // Use cached value as fallback
    }
    
    // Calculate maximum force magnitude - ONLY for LOCAL particles to avoid ghost particle issues
#ifdef USE_MPI
    size_t num_particles_for_fmax = system->get_local_particles();
#else
    size_t num_particles_for_fmax = particles.num_particles;
#endif
    
    size_t max_force_particle_id = 0;
    for (size_t i = 0; i < num_particles_for_fmax; i++) {
        Real f_mag = std::sqrt(particles.fx[i] * particles.fx[i] + 
                               particles.fy[i] * particles.fy[i] + 
                               particles.fz[i] * particles.fz[i]);
        if (f_mag > max_force) {
            max_force = f_mag;
            max_force_particle_id = i;
        }
    }
    
    Real avg_speed = (particles.num_particles > 0) ? (total_speed / particles.num_particles) : 0.0;
    Real temperature = 0.0;

#ifdef USE_MPI
    // In MPI mode, get global LOCAL particle count for accurate temperature
    // Do NOT count ghost particles - only owned particles contribute to N
    int global_particles = 0;
    int local_particle_count = static_cast<int>(system->get_local_particles());
    MPI_Allreduce(&local_particle_count, &global_particles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Get global maximum force across all ranks and which rank has it
    struct {
        Real value;
        int rank;
    } local_max, global_max;
    
    local_max.value = max_force;
    local_max.rank = system->get_rank();
    
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    max_force = global_max.value;
    
    // Broadcast the particle ID from the rank that has the maximum force
    int max_force_rank = global_max.rank;
    unsigned long long global_max_force_particle_id = static_cast<unsigned long long>(max_force_particle_id);
    MPI_Bcast(&global_max_force_particle_id, 1, MPI_UNSIGNED_LONG_LONG, max_force_rank, MPI_COMM_WORLD);
    max_force_particle_id = static_cast<size_t>(global_max_force_particle_id);
#else
    int global_particles = particles.num_particles;
#endif

    if (particles.num_particles > 0 && system->get_rank() == 0) {
        // Temperature calculation for 'real' units (kcal/mol, LAMMPS convention)
        // T = (2 * KE) / (3 * N * kB), KE in kcal/mol, kB = 0.0019872041 kcal/(mol·K)
        temperature = (2.0 * total_ke) / (3.0 * global_particles * kB);
    }
    
    if (system->get_rank() == 0) {
        // Output logic remains on rank 0
        std::stringstream output_line;
        output_line << std::fixed << std::setprecision(6);

        // Split output_info string by comma
        std::stringstream ss(output_info);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // Trim whitespace
            item.erase(0, item.find_first_not_of(" \t\r\n"));
            item.erase(item.find_last_not_of(" \t\r\n") + 1);

            if (item == "STEP") output_line << "Step: " << step;
            else if (item == "T") output_line << ", T: " << temperature;
            else if (item == "KE") output_line << ", KE: " << total_ke;
            else if (item == "PE") output_line << ", PE: " << total_pe;
            else if (item == "ETOTAL") output_line << ", Etot: " << total_ke + total_pe;
            else if (item == "MAX_SPEED") output_line << ", MaxSpd: " << max_speed;
            else if (item == "AVG_SPEED") output_line << ", AvgSpd: " << avg_speed;
            else if (item == "FMAX") output_line << ", Fmax: " << max_force << " (particle " << max_force_particle_id << ")";
        }
        std::cout << output_line.str() << std::endl;

        // Trajectory writing
        if (step % trajectory_output_freq == 0) {
            write_trajectory(step);
        }
    }
    
    // Accumulate output timing
    auto output_end = std::chrono::high_resolution_clock::now();
    double output_time = std::chrono::duration<double>(output_end - output_start).count();
    cumulative_timing.output_time += output_time;
}

void MDSimulation::finalize() {
    // Print LAMMPS-style timing breakdown
    if (system->get_rank() == 0 && cumulative_timing.step_count > 0) {
        print_timing_breakdown();
    }
    
    if (force_calc) {
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
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Skip empty lines after trimming
        if (line.empty()) continue;
        
        // Check for YAML sections
        if (line.find(':') != std::string::npos && line[0] != ' ' && line[0] != '\t') {
            current_section = line.substr(0, line.find(':'));
            // Trim whitespace
            current_section.erase(0, current_section.find_first_not_of(" \t"));
            current_section.erase(current_section.find_last_not_of(" \t") + 1);
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
            if (key == "mode") {
                force_calculation_mode = value;
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
        } else if (key == "barostat") {
            barostat_type = value;
        } else if (key == "target_pressure") {
            target_pressure = std::stod(value);
        } else if (key == "tau_p") {
            tau_p = std::stod(value);
        } else if (key == "python_force_path") {
            python_force_path = value;
        } else if (key == "python_force_enabled") {
            python_force_enabled = (value == "true");
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
    if (neighbor_list_update_freq <= 0) {
        throw std::runtime_error("Invalid neighbor_list_update_freq: must be positive");
    }
    
    // Set the physical cutoff in the system for force calculations
    system->set_physical_cutoff(cutoff_distance);
    
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
        if (force_calculation_mode == "python") {
            std::cout << "  Python force path: " << python_force_path << std::endl;
        }
        std::cout << "  Python force interface: " << (python_force_enabled ? "enabled" : "disabled") << std::endl;
        if (python_force_enabled) {
            std::cout << "  Python force path: " << python_force_path << std::endl;
        }
    }
}

void MDSimulation::initialize_integrator() {
    // Determine backend based on build flags and create appropriate integrator
    if (integrator_type == "velocity_verlet") {
#ifdef USE_SYCL
        integrator = std::make_unique<VelocityVerletIntegratorSYCL>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Velocity-Verlet integrator" << std::endl;
        }
#elif defined(USE_CUDA)
        integrator = std::make_unique<VelocityVerletIntegratorCUDA>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Velocity-Verlet integrator" << std::endl;
        }
#else
        integrator = std::make_unique<VelocityVerletIntegrator>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Velocity-Verlet integrator" << std::endl;
        }
#endif

    } else if (integrator_type == "r_respa") {
#ifdef USE_SYCL
        integrator = std::make_unique<RRESPAIntegratorSYCL>(dt, 4);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL r-RESPA integrator" << std::endl;
        }
#elif defined(USE_CUDA)
        integrator = std::make_unique<RRESPAIntegratorCUDA>(dt, 4);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA r-RESPA integrator" << std::endl;
        }
#else
        integrator = std::make_unique<RRESPAIntegrator>(dt, 4);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU r-RESPA integrator" << std::endl;
        }
#endif

    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
#ifdef USE_SYCL
        integrator = std::make_unique<LeapFrogIntegratorSYCL>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Leap-frog integrator" << std::endl;
        }
#elif defined(USE_CUDA)
        integrator = std::make_unique<LeapFrogIntegratorCUDA>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Leap-frog integrator" << std::endl;
        }
#else
        integrator = std::make_unique<LeapFrogIntegrator>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Leap-frog integrator" << std::endl;
        }
#endif
    } else {
        // Default to velocity-verlet
#ifdef USE_SYCL
        integrator = std::make_unique<VelocityVerletIntegratorSYCL>(dt);
#elif defined(USE_CUDA)
        integrator = std::make_unique<VelocityVerletIntegratorCUDA>(dt);
#else
        integrator = std::make_unique<VelocityVerletIntegrator>(dt);
#endif
        if (system->get_rank() == 0) {
            std::cout << "Unknown integrator type '" << integrator_type 
                      << "', using Velocity-Verlet as default" << std::endl;
        }
    }
}

void MDSimulation::initialize_thermostat() {
    if (thermostat_type == "none" || thermostat_type.empty()) {
        thermostat = nullptr;
        if (system->get_rank() == 0) {
            std::cout << "No thermostat initialized" << std::endl;
        }
        return;
    }
    
    if (thermostat_type == "nose_hoover") {
        Real Q = 1.0; // thermostat mass, could be made configurable
#ifdef USE_SYCL
        thermostat = std::make_unique<NoseHooverThermostatSYCL>(target_temperature, Q);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Nose-Hoover thermostat (T=" << target_temperature << " K)" << std::endl;
        }
#elif defined(USE_CUDA)
        thermostat = std::make_unique<NoseHooverThermostatCUDA>(target_temperature, Q);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Nose-Hoover thermostat (T=" << target_temperature << " K)" << std::endl;
        }
#else
        thermostat = std::make_unique<NoseHooverThermostat>(target_temperature, Q);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Nose-Hoover thermostat (T=" << target_temperature << " K)" << std::endl;
        }
#endif
    } else if (thermostat_type == "langevin") {
        Real gamma = 1.0 / tau_t; // friction coefficient from time constant
#ifdef USE_SYCL
        thermostat = std::make_unique<LangevinThermostatSYCL>(target_temperature, gamma);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Langevin thermostat (T=" << target_temperature 
                      << " K, tau=" << tau_t << " ps)" << std::endl;
        }
#elif defined(USE_CUDA)
        thermostat = std::make_unique<LangevinThermostatCUDA>(target_temperature, gamma);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Langevin thermostat (T=" << target_temperature 
                      << " K, tau=" << tau_t << " ps)" << std::endl;
        }
#else
        thermostat = std::make_unique<LangevinThermostat>(target_temperature, gamma);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Langevin thermostat (T=" << target_temperature 
                      << " K, tau=" << tau_t << " ps)" << std::endl;
        }
#endif
    } else if (thermostat_type == "andersen") {
        Real collision_freq = 1.0 / tau_t; // collision frequency from time constant
#ifdef USE_SYCL
        thermostat = std::make_unique<AndersenThermostatSYCL>(target_temperature, collision_freq);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Andersen thermostat (T=" << target_temperature 
                      << " K, freq=" << collision_freq << " ps^-1)" << std::endl;
        }
#elif defined(USE_CUDA)
        thermostat = std::make_unique<AndersenThermostatCUDA>(target_temperature, collision_freq);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Andersen thermostat (T=" << target_temperature 
                      << " K, freq=" << collision_freq << " ps^-1)" << std::endl;
        }
#else
        thermostat = std::make_unique<AndersenThermostat>(target_temperature, collision_freq);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Andersen thermostat (T=" << target_temperature 
                      << " K, freq=" << collision_freq << " ps^-1)" << std::endl;
        }
#endif
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown thermostat type '" << thermostat_type 
                      << "', no thermostat initialized" << std::endl;
        }
    }
}

void MDSimulation::initialize_barostat() {
    if (barostat_type == "none" || barostat_type.empty()) {
        barostat = nullptr;
        if (system->get_rank() == 0 ) {
            std::cout << "No barostat initialized" << std::endl;
        }
        return;
    }
    
    if (barostat_type == "parrinello_rahman") {
        Real W = 1.0; // barostat mass, could be made configurable
#ifdef USE_SYCL
        barostat = std::make_unique<ParrinelloRahmanBarostatSYCL>(target_pressure, W);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Parrinello-Rahman barostat (P=" << target_pressure 
                      << " bar)" << std::endl;
        }
#elif defined(USE_CUDA)
        barostat = std::make_unique<ParrinelloRahmanBarostatCUDA>(target_pressure, W);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Parrinello-Rahman barostat (P=" << target_pressure 
                      << " bar)" << std::endl;
        }
#else
        barostat = std::make_unique<ParrinelloRahmanBarostat>(target_pressure, W);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Parrinello-Rahman barostat (P=" << target_pressure 
                      << " bar)" << std::endl;
        }
#endif
    } else if (barostat_type == "andersen") {
        Real compressibility = 1e-5; // could be made configurable
#ifdef USE_SYCL
        barostat = std::make_unique<AndersenBarostatSYCL>(target_pressure, compressibility);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Andersen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " ps)" << std::endl;
        }
#elif defined(USE_CUDA)
        barostat = std::make_unique<AndersenBarostatCUDA>(target_pressure, compressibility);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Andersen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " ps)" << std::endl;
        }
#else
        barostat = std::make_unique<AndersenBarostat>(target_pressure, compressibility);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Andersen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " ps)" << std::endl;
        }
#endif
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

// Print LAMMPS-style timing breakdown
void MDSimulation::print_timing_breakdown() {
#ifdef USE_MPI
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Gather timing statistics from all ranks
    double local_times[6] = {
        cumulative_timing.pair_time,
        cumulative_timing.neigh_time,
        cumulative_timing.comm_time,
        cumulative_timing.output_time,
        cumulative_timing.modify_time,
        cumulative_timing.other_time
    };
    
    double min_times[6], max_times[6], sum_times[6];
    
    MPI_Reduce(local_times, min_times, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_times, sum_times, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (system->get_rank() == 0) {
        double avg_times[6];
        for (int i = 0; i < 6; i++) {
            avg_times[i] = sum_times[i] / mpi_size;
        }
        
        std::cout << "\n";
        std::cout << "Loop time of " << std::fixed << std::setprecision(5) 
                  << cumulative_timing.total_time << " on " << mpi_size 
                  << " procs for " << cumulative_timing.step_count << " steps\n";
        std::cout << "\n";
        std::cout << "MPI task timing breakdown:\n";
        std::cout << "Section |  min time  |  avg time  |  max time  |%varavg| %total\n";
        std::cout << "---------------------------------------------------------------\n";
        
        auto print_section = [&](const char* name, int idx) {
            double pct = (avg_times[idx] / cumulative_timing.total_time) * 100.0;
            double variance = 0.0;
            if (avg_times[idx] > 1e-10) {
                variance = ((max_times[idx] - min_times[idx]) / avg_times[idx]) * 100.0;
            }
            
            std::cout << std::left << std::setw(8) << name << "| "
                      << std::right << std::fixed << std::setprecision(4) << std::setw(10) << min_times[idx] << " | "
                      << std::setw(10) << avg_times[idx] << " | "
                      << std::setw(10) << max_times[idx] << " | "
                      << std::setw(5) << std::setprecision(1) << variance << " | "
                      << std::setw(6) << std::setprecision(2) << pct << "\n";
        };
        
        print_section("Pair", 0);
        print_section("Neigh", 1);
        print_section("Comm", 2);
        print_section("Output", 3);
        print_section("Modify", 4);
        
        // Calculate "Other" time
        double other_time = cumulative_timing.total_time - 
                           (avg_times[0] + avg_times[1] + avg_times[2] + avg_times[3] + avg_times[4]);
        double other_pct = (other_time / cumulative_timing.total_time) * 100.0;
        
        std::cout << std::left << std::setw(8) << "Other" << "|            | "
                  << std::right << std::fixed << std::setprecision(4) << std::setw(10) << other_time << " |"
                  << "            |       | "
                  << std::setw(6) << std::setprecision(2) << other_pct << "\n";
        
        std::cout << "\n";
        
        // Performance metrics
        double natoms = system->get_num_particles();
        
        std::cout << "Nlocal:    " << std::fixed << std::setprecision(1) << natoms 
                  << " ave " << natoms << " max " << natoms << " min\n";
        std::cout << "Ave neighs/atom = " << std::setprecision(2) << 0.0 << "\n";
        std::cout << "Neighbor list builds = " << neighbor_list_builds << "\n";
        std::cout << "\n";
    }
#else
    // Single processor version
    double total = cumulative_timing.total_time;
    
    std::cout << "\n";
    std::cout << "Loop time of " << std::fixed << std::setprecision(5) 
              << total << " on 1 procs for " << cumulative_timing.step_count << " steps\n";
    std::cout << "\n";
    std::cout << "Section timing breakdown:\n";
    std::cout << "Section |    time    | %total\n";
    std::cout << "--------------------------------\n";
    
    auto print_section = [&](const char* name, double time) {
        double pct = (time / total) * 100.0;
        std::cout << std::left << std::setw(8) << name << "| "
                  << std::right << std::fixed << std::setprecision(4) << std::setw(10) << time << " | "
                  << std::setw(6) << std::setprecision(2) << pct << "\n";
    };
    
    print_section("Pair", cumulative_timing.pair_time);
    print_section("Neigh", cumulative_timing.neigh_time);
    print_section("Comm", cumulative_timing.comm_time);
    print_section("Output", cumulative_timing.output_time);
    print_section("Modify", cumulative_timing.modify_time);
    
    double other_time = total - (cumulative_timing.pair_time + cumulative_timing.neigh_time + 
                                 cumulative_timing.comm_time + cumulative_timing.output_time + 
                                 cumulative_timing.modify_time);
    print_section("Other", other_time);
    
    std::cout << "\n";
    
    // Performance metrics
    double nsteps = cumulative_timing.step_count;
    double natoms = system->get_num_particles();
    
    std::cout << "Neighbor list builds = " << neighbor_list_builds << "\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(3) 
              << (nsteps * natoms / total / 1e6) << " million atom-steps/sec\n";
    std::cout << "\n";
#endif
}
