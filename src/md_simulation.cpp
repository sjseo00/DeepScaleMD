#define OMPI_SKIP_MPICXX 1
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
    neighbor_list_type_str("cell_list"),
    neighbor_list_tile_size(256),
    neighbor_list_update_freq(10),  // Default update frequency: every 10 steps
    verlet_skin(2.0),               // Default Verlet skin distance
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
    cached_potential_energy(0.0) {}

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
    
    // Set up domain decomposition after particles are loaded
    system->setup_domain_decomposition();
    
    if (system->get_rank() == 0 && debug_level(2)) {
        std::cout << "Domain decomposition completed" << std::endl;
    }
    
    // Load topology file from configuration and assign properties to particles
    if (!topology_file.empty()) {
        system->load_topology(topology_file);
        // Set up force field parameters after topology is loaded
        system->setup_force_field_params();
        
        if (system->get_rank() == 0 && debug_level(2)) {
            std::cout << "Topology and force field parameters loaded" << std::endl;
        }
    }
    
    // Initialize velocities with Maxwell-Boltzmann distribution using target temperature
    system->initialize_velocities(target_temperature);
    
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
    NeighborListType nlist_type = NeighborListType::CELL_LIST; // Use basic cell list by default
    if (neighbor_list_type_str == "morton_hilbert") {
#ifdef USE_CUDA
        nlist_type = NeighborListType::MORTON_HILBERT_CSR;
        if (system->get_rank() == 0) {
            std::cout << "Using Morton/Hilbert neighbor list with CUDA optimization" << std::endl;
        }
#else
        if (system->get_rank() == 0) {
            std::cout << "Morton/Hilbert neighbor list requires CUDA. Falling back to cell list." << std::endl;
        }
        nlist_type = NeighborListType::CELL_LIST;
#endif
    } else if (neighbor_list_type_str == "cell_list") {
        nlist_type = NeighborListType::CELL_LIST;
        if (system->get_rank() == 0) {
            std::cout << "Using cell list neighbor list" << std::endl;
        }
    } else if (neighbor_list_type_str == "verlet_list") {
        nlist_type = NeighborListType::VERLET_LIST;
        if (system->get_rank() == 0) {
            std::cout << "Using Verlet list neighbor list with skin distance: " << verlet_skin << std::endl;
        }
    } else if (neighbor_list_type_str == "verlet_cell_list") {
        nlist_type = NeighborListType::VERLET_CELL_LIST;
        if (system->get_rank() == 0) {
            std::cout << "Using Verlet-Cell hybrid neighbor list with skin distance: " << verlet_skin << std::endl;
        }
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown neighbor list type '" << neighbor_list_type_str 
                      << "', using cell list (default)" << std::endl;
        }
        nlist_type = NeighborListType::CELL_LIST;
    }
    
    // Set up neighbor list configuration based on type
    std::string nlist_config;
    if (nlist_type == NeighborListType::CELL_LIST) {
        // For cell list, use verlet_skin for stability (like verlet_cell_list)
        nlist_config = std::to_string(verlet_skin);
    } else if (nlist_type == NeighborListType::VERLET_LIST) {
        // For Verlet list, use verlet_skin distance
        nlist_config = std::to_string(verlet_skin);
    } else if (nlist_type == NeighborListType::VERLET_CELL_LIST) {
        // For Verlet-Cell hybrid, use verlet_skin distance
        nlist_config = std::to_string(verlet_skin);
    } else {
        // For Morton-Hilbert, use verlet_skin for stability
        nlist_config = std::to_string(verlet_skin);
    }
    
    neighbor_builder = create_neighbor_list_builder(nlist_type, cutoff, nlist_config, system->get_rank());
    neighbor_builder->initialize();
    
    // Initialize components based on configuration
    initialize_integrator();
    initialize_thermostat();
    initialize_barostat();
    
}

void MDSimulation::single_step() {
    // Build neighbor list (periodically)
    if (current_step % neighbor_list_update_freq == 0) {
        /*
        if (system->get_rank() == 0) {
            std::cout << "Building neighbor list at step " << current_step << std::endl;
        }
        */
        neighbor_builder->build(system->get_particles(), system->get_box(), 
                               system->get_neighbor_list());
    }
    // Check if we're using Velocity-Verlet integrator
    VelocityVerletIntegrator* vv_integrator = dynamic_cast<VelocityVerletIntegrator*>(integrator.get());
    
    if (!vv_integrator) {
        // For non-Velocity-Verlet integrators: Force calculation first
        // Compute forces based on calculation mode
        if (force_calculation_mode == "python" && python_interface) {
            // Use Python for all force calculations
            python_interface->compute_forces(system->get_particles(), system->get_ff_params(), 
                                            system->get_neighbor_list());
            // For now, PE calculation with Python forces is not implemented
            cached_potential_energy = 0.0;
        } else if (force_calculation_mode == "builtin" && force_calc) {
            // Use built-in force calculator
            force_calc->compute_forces(*system);
        } else {
            // Fallback to built-in if mode is invalid or interface not available
            if (force_calc) {
                force_calc->compute_forces(*system);
            }
            if (system->get_rank() == 0 && force_calculation_mode == "python" && !python_interface) {
                static bool warned = false;
                if (!warned) {
                    std::cout << "Warning: Python force calculation requested but interface not available. Using built-in forces." << std::endl;
                    warned = true;
                }
            }
        }
        
        // Then integrate equations of motion
        integrator->integrate(system->get_particles(), system->get_box(), dt, system->get_local_particles());
    } else {
        // For Velocity-Verlet: Position update first, then force calculation, then velocity completion
        // Step 1: Position update (stores old forces for later velocity update)
        vv_integrator->integrate(system->get_particles(), system->get_box(), dt, system->get_local_particles());
        
        // Step 2: Compute forces at new positions
        if (force_calculation_mode == "python" && python_interface) {
            // Use Python for all force calculations
            python_interface->compute_forces(system->get_particles(), system->get_ff_params(), 
                                            system->get_neighbor_list());
            // For now, PE calculation with Python forces is not implemented
            cached_potential_energy = 0.0;
        } else if (force_calculation_mode == "builtin" && force_calc) {
            // Use built-in force calculator
            force_calc->compute_forces(*system);
        } else {
            // Fallback to built-in if mode is invalid or interface not available
            if (force_calc) {
                force_calc->compute_forces(*system);
            }
            if (system->get_rank() == 0 && force_calculation_mode == "python" && !python_interface) {
                static bool warned = false;
                if (!warned) {
                    std::cout << "Warning: Python force calculation requested but interface not available. Using built-in forces." << std::endl;
                    warned = true;
                }
            }
        }
        
        // Step 3: Complete velocity update with new forces
        vv_integrator->complete_velocity_update(system->get_particles(), dt, system->get_local_particles());
    }

    // Check for numerical instabilities and adjust timestep if needed
    // check_energy_stability();
    
    // Apply thermostat
    if (thermostat) {
        thermostat->apply(system->get_particles(), dt, system->get_local_particles());
    }
    
    // Update kinetic energy after velocity update and thermostat application
    if (force_calc) {
        force_calc->calculate_kinetic_energy(*system);
    }
    
    // Apply barostat
    if (barostat) {
        barostat->apply(system->get_particles(), system->get_box(), dt, system->get_local_particles());
    }
    
    current_step++;
}

void MDSimulation::calculate_potential_energy() {
    cached_potential_energy = 0.0;
    const NeighborList& nlist = system->get_neighbor_list();
    const ForceFieldParams& ff_params = system->get_ff_params();
    const SimulationBox& box = system->get_box();
    ParticleData& particles = system->get_particles();
    
    // Simple LJ potential energy calculation with PBC
    if (!ff_params.epsilon.empty() && !ff_params.sigma.empty()) {
        for (size_t i = 0; i < nlist.num_neighbors.size(); i++) {
            size_t start = (i == 0) ? 0 : nlist.num_neighbors[i-1];
            size_t end = nlist.num_neighbors[i];
            
            for (size_t idx = start; idx < end; idx++) {
                size_t j = nlist.neighbors[idx];
                if (i >= j) continue; // Avoid double counting
                
                Real dx = particles.x[i] - particles.x[j];
                Real dy = particles.y[i] - particles.y[j];
                Real dz = particles.z[i] - particles.z[j];
                
                // Apply periodic boundary conditions
                if (box.periodic_x && std::abs(dx) > box.lx * 0.5) {
                    dx = (dx > 0) ? dx - box.lx : dx + box.lx;
                }
                if (box.periodic_y && std::abs(dy) > box.ly * 0.5) {
                    dy = (dy > 0) ? dy - box.ly : dy + box.ly;
                }
                if (box.periodic_z && std::abs(dz) > box.lz * 0.5) {
                    dz = (dz > 0) ? dz - box.lz : dz + box.lz;
                }
                
                Real r2 = dx*dx + dy*dy + dz*dz;
                Real r = std::sqrt(r2);
                
                // Use average parameters (simple approximation)
                Real epsilon = (ff_params.epsilon[i] + ff_params.epsilon[j]) * 0.5;
                Real sigma = (ff_params.sigma[i] + ff_params.sigma[j]) * 0.5;
                
                if (r > 0.1 && r < cutoff_distance) { // Avoid singularity and cutoff
                    Real sr = sigma / r;
                    Real sr6 = sr * sr * sr * sr * sr * sr;
                    Real sr12 = sr6 * sr6;
                    cached_potential_energy += 4.0 * epsilon * (sr12 - sr6);
                }
            }
        }
    }
}

void MDSimulation::run() {
   
    if (system->get_rank() == 0) {
        std::cout << "Starting MD simulation for " << total_steps << " steps" << std::endl;
    }
    
    for (int step = 0; step < total_steps; step++) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        single_step();
        
        auto step_end = std::chrono::high_resolution_clock::now();
        auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
        
        // Output results based on output_freq
        if (step % output_freq == 0) {
            output_results(step, step_duration.count() / 1000.0);  // Pass step time in milliseconds
            // Only synchronize when outputting results to ensure consistent output
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

void MDSimulation::output_results(int step, double step_time_ms) {
    if (system->get_rank() == 0) {
        // Calculate all possible values
        Real total_ke = 0.0;
        Real max_speed = 0.0;
        Real total_speed = 0.0;
        Real max_force = 0.0;
        Real total_pe = 0.0;  // Potential Energy
        ParticleData& particles = system->get_particles();
        
        if (force_calc) {
            total_ke = force_calc->get_kinetic_energy();
            total_pe = force_calc->get_potential_energy();
        } else {
            // Fallback: calculate KE manually if no force calculator
            for (size_t i = 0; i < particles.num_particles; i++) {
                Real v2 = particles.vx[i] * particles.vx[i] + 
                          particles.vy[i] * particles.vy[i] + 
                          particles.vz[i] * particles.vz[i];
                total_ke += 0.5 * particles.mass[i] * v2;
            }
            total_pe = cached_potential_energy; // Use cached value as fallback
        }
        

        Real avg_speed = total_speed / particles.num_particles;
        Real temperature = 0.0;
        if (particles.num_particles > 0) {
                // Temperature calculation for 'real' units (kcal/mol, LAMMPS convention)
                // T = (2 * KE) / (3 * N * kB), KE in kcal/mol, kB = 0.0019872041 kcal/(mol·K)
                constexpr double kB_kcal = 0.0019872041; // Boltzmann constant in kcal/(mol·K)
                temperature = (2.0 * total_ke) / (3.0 * particles.num_particles * kB_kcal);
        }
        
        // Determine when to output based on debug level and frequency
        bool should_output = false;
        should_output = (step % output_freq == 0);
        
        if (should_output) {
            // Parse output_info and build output string
            std::stringstream output;
            std::stringstream ss(output_info);
            std::string item;
            bool first = true;
            
            while (std::getline(ss, item, ',')) {
                // Trim all whitespace (space, tab, newline, carriage return) from both ends
                auto trim = [](std::string& s) {
                    const char* ws = " \t\n\r";
                    size_t start = s.find_first_not_of(ws);
                    if (start == std::string::npos) {
                        s.clear();
                        return;
                    }
                    size_t end = s.find_last_not_of(ws);
                    s = s.substr(start, end - start + 1);
                };
                trim(item);
                
                if (!first) output << ", ";
                first = false;
                
                if (item == "STEP") {
                    output << "Step " << step;
                } else if (item == "T") {
                    output << "T = " << std::fixed << std::setprecision(3) << temperature << " K";
                } else if (item == "KE") {
                    output << "KE = " << std::setprecision(8) << total_ke;
                } else if (item == "PE") {
                    output << "PE = " << std::setprecision(8) << total_pe;
                } else if (item == "AVG_SPD") {
                    output << "Avg Speed = " << std::setprecision(3) << avg_speed;
                } else if (item == "MAX_SPD") {
                    output << "Max Speed = " << std::setprecision(3) << max_speed;
                } else if (item == "MAX_FORCE") {
                    output << "Max Force = " << std::setprecision(3) << max_force;
                } else if (item == "SAMPLE_POS") {
                    output << "Sample pos = (" << std::setprecision(3) << particles.x[0] << "," 
                           << particles.y[0] << "," << particles.z[0] << ")";
                } else if (item == "PARTICLES") {
                    output << "Particles = " << particles.num_particles;
                } else if (item == "STEP_TIME") {
                    output << "Step time = " << std::setprecision(3) << step_time_ms << " ms";
                }
            }
            
            std::cout << output.str() << std::endl;
        }
        
        // Write trajectory at specified frequency
        if (step % trajectory_output_freq == 0) {
            write_trajectory(step);
        }
        
    }
}

void MDSimulation::finalize() {
    if (force_calc) {
        force_calc->finalize();
    }
    
    if (python_interface) {
        python_interface->finalize_python();
    }
    
    if (neighbor_builder) {
        neighbor_builder->finalize();
    }
    
    MPI_Finalize();
    
    if (system->get_rank() == 0 && debug_level(2)) {
        std::cout << "MD simulation finalized" << std::endl;
    }
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
    } else if (integrator_type == "verlet") {
#ifdef USE_SYCL
        integrator = std::make_unique<VerletIntegratorSYCL>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Verlet integrator" << std::endl;
        }
#elif defined(USE_CUDA)
        integrator = std::make_unique<VerletIntegratorCUDA>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Verlet integrator" << std::endl;
        }
#else
        integrator = std::make_unique<VerletIntegrator>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Verlet integrator" << std::endl;
        }
#endif
    } else if (integrator_type == "leapfrog") {
#ifdef USE_SYCL
        integrator = std::make_unique<LeapfrogIntegratorSYCL>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized SYCL Leapfrog integrator" << std::endl;
        }
#elif defined(USE_CUDA)
        integrator = std::make_unique<LeapfrogIntegratorCUDA>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CUDA Leapfrog integrator" << std::endl;
        }
#else
        integrator = std::make_unique<LeapfrogIntegrator>(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized CPU Leapfrog integrator" << std::endl;
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
        if (system->get_rank() == 0 && debug_level(2)) {
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
        if (system->get_rank() == 0 && debug_level(2)) {
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
        if (system->get_rank() == 0 && debug_level(2)) {
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
