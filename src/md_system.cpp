#define OMPI_SKIP_MPICXX 1
#include "md_system.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <random>
#include <map>
#include <unistd.h>  // For getcwd
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// External debug level variable
extern int g_debug_level;
inline bool debug_level(int level) { return g_debug_level >= level; }

MDSystem::MDSystem() : rank(0), size(1), local_start(0), local_end(0), d_particles(nullptr), use_gpu(false) {
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    topology = std::make_unique<Topology>();
    force_manager = std::make_unique<ForceManager>();
}

MDSystem::~MDSystem() {
    if (d_particles) {
        // Free device memory - implementation depends on backend
    }
}

void MDSystem::initialize_mpi() {
#ifdef USE_MPI
    // MPI is already initialized in main.cpp
    // Just check thread support level
    int provided;
    MPI_Query_thread(&provided);
    
    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            std::cerr << "Warning: MPI does not support multithreading\n";
        }
    }
#else
    std::cout << "MPI not available - running in single process mode" << std::endl;
#endif
}

void MDSystem::initialize() {
    // Default initialization - subclasses can override
}

void MDSystem::finalize() {
    // Default finalization - subclasses can override
}

void MDSystem::load_config(const std::string& config_file) {
    if (rank == 0) {
        std::cout << "Loading configuration from " << config_file << std::endl;
    }
    
    // Simple YAML-like parser for basic functionality
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_file);
    }
    
    std::map<std::string, std::string> config_map;
    std::string line;
    std::string current_section = "";
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);
        
        // Remove quotes if present
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }
        
        // Handle section headers (keys without values or with empty values)
        if (value.empty() || key == "builtin_forcefield" || key == "python_force_interface") {
            current_section = key;
            continue;
        }
        
        // Add section prefix for nested keys
        std::string full_key = key;
        if (current_section == "builtin_forcefield") {
            full_key = "builtin_" + key;
        } else if (current_section == "python_force_interface") {
            full_key = "python_" + key;
        } else {
            // Reset section for top-level keys
            current_section = "";
        }
        
        config_map[full_key] = value;
        
        // Parse box dimensions  
        if (key == "box_x") box.lx = std::stod(value);
        else if (key == "box_y") box.ly = std::stod(value);
        else if (key == "box_z") box.lz = std::stod(value);
        else if (key == "periodic_x") box.periodic_x = (value == "true");
        else if (key == "periodic_y") box.periodic_y = (value == "true");
        else if (key == "periodic_z") box.periodic_z = (value == "true");
    }
    
    // Initialize force manager with parsed configuration
    initialize_force_manager(config_map);
    
    if (rank == 0 && debug_level(1)) {
        std::cout << "Box dimensions: " << box.lx << " x " << box.ly << " x " << box.lz << std::endl;
        // Backend information is printed in initialize_force_manager
    }
}

void MDSystem::load_coordinates(const std::string& coord_file) {
    if (rank == 0 && debug_level(1)) {
        std::cout << "Loading coordinates from " << coord_file << std::endl;
    }
    
    // Use standard XYZ format loading
    load_standard_xyz(coord_file);
}

void MDSystem::initialize_velocities(Real temperature) {
    if (rank == 0) {
        std::cout << "Initializing velocities at T = " << temperature << " K for " << particles.num_particles << " particles" << std::endl;
    }
    
    // Generate random velocities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Real> dist(0.0, 1.0);
    
    // Initial velocity assignment - will be scaled later
    for (size_t i = 0; i < particles.num_particles; i++) {
        particles.vx[i] = dist(gen);
        particles.vy[i] = dist(gen);
        particles.vz[i] = dist(gen);
    }
    
    // Remove center-of-mass motion
    Real total_vx = 0.0, total_vy = 0.0, total_vz = 0.0;
    Real total_mass = 0.0;
    
    for (size_t i = 0; i < particles.num_particles; i++) {
        total_vx += particles.mass[i] * particles.vx[i];
        total_vy += particles.mass[i] * particles.vy[i];
        total_vz += particles.mass[i] * particles.vz[i];
        total_mass += particles.mass[i];
    }
    
    Real cm_vx = total_vx / total_mass;
    Real cm_vy = total_vy / total_mass;
    Real cm_vz = total_vz / total_mass;
    
    for (size_t i = 0; i < particles.num_particles; i++) {
        particles.vx[i] -= cm_vx;
        particles.vy[i] -= cm_vy;
        particles.vz[i] -= cm_vz;
    }
    
    // Scale velocities to match target temperature
    Real current_ke = 0.0;
    for (size_t i = 0; i < particles.num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        current_ke += 0.5 * particles.mass[i] * v2;
    }
    
    // Target kinetic energy: KE = (3/2) * N * kB * T
    // kB = 0.0019872041 kcal/(mol·K) for units real
    Real kB = 0.0019872041; // kcal/(mol·K)
    Real target_ke = 1.5 * particles.num_particles * kB * temperature;
    
    Real scale_factor = std::sqrt(target_ke / current_ke);
    
    for (size_t i = 0; i < particles.num_particles; i++) {
        particles.vx[i] *= scale_factor;
        particles.vy[i] *= scale_factor;
        particles.vz[i] *= scale_factor;
    }
    
    if (rank == 0 && debug_level(1)) {
        std::cout << "Velocity initialization: target T = " << temperature 
                  << " K, target KE = " << target_ke << ", scale factor = " << scale_factor << std::endl;
        if (debug_level(3)) {
            std::cout << "Sample velocities: particle 0: (" << particles.vx[0] 
                      << ", " << particles.vy[0] << ", " << particles.vz[0] << ")" << std::endl;
        }
    }
}

void MDSystem::load_topology(const std::string& topo_file) {
    if (rank == 0) {
        std::cout << "Loading topology from " << topo_file << std::endl;
    }
    
    // Clean the filename - remove quotes and trim whitespace/control characters
    std::string clean_filename = topo_file;
    
    // First, trim whitespace and control characters from both ends
    auto start = clean_filename.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        clean_filename = "";
    } else {
        auto end = clean_filename.find_last_not_of(" \t\r\n");
        clean_filename = clean_filename.substr(start, end - start + 1);
    }
    
    // Then remove quotes if present
    if (!clean_filename.empty() && clean_filename.front() == '"' && clean_filename.back() == '"') {
        clean_filename = clean_filename.substr(1, clean_filename.length() - 2);
    }
    
    if (!topology->load_topology(clean_filename, rank)) {
        throw std::runtime_error("Failed to load topology file: " + clean_filename);
    }
    
    // After loading topology, assign masses and charges to particles
    assign_masses_from_topology();
    assign_charges_from_topology();
}

void MDSystem::setup_domain_decomposition() {
    size_t total_particles = particles.num_particles;
    size_t particles_per_proc = total_particles / size;
    size_t remainder = total_particles % size;
    
    local_start = rank * particles_per_proc;
    local_end = local_start + particles_per_proc;
    
    if (static_cast<size_t>(rank) < remainder) {
        local_start += rank;
        local_end += rank + 1;
    } else {
        local_start += remainder;
        local_end += remainder;
    }
    
    if (rank == 0) {
        std::cout << "Domain decomposition setup for " << size << " processes" << std::endl;
    }
}

void MDSystem::allocate_device_memory() {
    if (!use_gpu) return;
    
    // Device memory allocation - implementation depends on backend (CUDA/SYCL)
    // This is handled in the specific force calculator implementations
}

void MDSystem::copy_to_device() {
    if (!use_gpu) return;
    
    // Copy data to device - implementation depends on backend
}

void MDSystem::copy_from_device() {
    if (!use_gpu) return;
    
    // Copy data from device - implementation depends on backend
}

void MDSystem::load_standard_xyz(const std::string& coord_file) {
    // Print current working directory for debugging (debug level 2)
    if (rank == 0 && debug_level(2)) {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            std::cout << "Current working directory: " << cwd << std::endl;
        }
        std::cout << "Attempting to load coordinate file: " << coord_file << std::endl;
    }
    
    // Remove quotes from filename if present
    std::string clean_filename = coord_file;
    
    // First, trim whitespace and control characters from both ends
    auto start = clean_filename.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        clean_filename = "";
    } else {
        auto end = clean_filename.find_last_not_of(" \t\r\n");
        clean_filename = clean_filename.substr(start, end - start + 1);
    }
    
    // Then remove quotes if present
    if (!clean_filename.empty() && clean_filename.front() == '"' && clean_filename.back() == '"') {
        clean_filename = clean_filename.substr(1, clean_filename.length() - 2);
    }
    
    std::ifstream file(clean_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open coordinate file: " + clean_filename);
    }
    
    std::string line;
    
    // Read number of atoms
    std::getline(file, line);
    int num_atoms = std::stoi(line);
    
    // Skip comment line
    std::getline(file, line);
    
    // Resize particle arrays
    particles.x.resize(num_atoms);
    particles.y.resize(num_atoms);
    particles.z.resize(num_atoms);
    particles.vx.resize(num_atoms, 0.0);
    particles.vy.resize(num_atoms, 0.0);
    particles.vz.resize(num_atoms, 0.0);
    particles.fx.resize(num_atoms, 0.0);
    particles.fy.resize(num_atoms, 0.0);
    particles.fz.resize(num_atoms, 0.0);
    particles.mass.resize(num_atoms, 1.0);  // Default mass, will be set from topology
    particles.charge.resize(num_atoms, 0.0);  // Default charge, will be set from topology
    particles.type.resize(num_atoms, 0);
    particles.num_particles = num_atoms;  // Set the particle count
    
    // Read atom coordinates
    int atom_count = 0;
    while (std::getline(file, line) && atom_count < num_atoms) {
        std::istringstream iss(line);
        std::string element;
        Real x, y, z;
        
        if (iss >> element >> x >> y >> z) {
            particles.x[atom_count] = x;
            particles.y[atom_count] = y;
            particles.z[atom_count] = z;
            atom_count++;
        }
    }
    
    if (rank == 0 && debug_level(1)) {
        std::cout << "Loaded " << atom_count << " particles from " << clean_filename << std::endl;
        if (debug_level(3)) {
            std::cout << "Sample data - Particle 0: pos(" << particles.x[0] << "," << particles.y[0] << "," << particles.z[0] << ")" << std::endl;
        }
    }
    
    file.close();
}

void MDSystem::assign_masses_from_topology() {
    if (!topology) {
        throw std::runtime_error("Topology not loaded");
    }
    
    if (particles.mass.size() == 0) {
        if (rank == 0) {
            std::cerr << "Error: Particle mass vector is empty. Load coordinates first." << std::endl;
        }
        return;
    }
    
    for (int i = 0; i < static_cast<int>(particles.mass.size()); ++i) {
        try {
            particles.mass[i] = topology->get_atom_mass(i);
        } catch (const std::runtime_error& e) {
            if (rank == 0) {
                std::cerr << "Warning: Cannot get mass for atom " << i << ": " << e.what() << std::endl;
                std::cerr << "Using default mass 1.0" << std::endl;
            }
            particles.mass[i] = 1.0;  // Default mass
        }
    }
    
    if (rank == 0) {
        std::cout << "Assigned masses from topology" << std::endl;
        if (particles.mass.size() > 0) {
            std::cout << "Sample mass: particle 0: " << particles.mass[0] << std::endl;
        }
    }
}

void MDSystem::assign_charges_from_topology() {
    if (!topology) {
        throw std::runtime_error("Topology not loaded");
    }
    
    for (int i = 0; i < static_cast<int>(particles.charge.size()); ++i) {
        try {
            particles.charge[i] = topology->get_atom_charge(i);
        } catch (const std::runtime_error& e) {
            if (rank == 0) {
                std::cerr << "Warning: Cannot get charge for atom " << i << ": " << e.what() << std::endl;
                std::cerr << "Using default charge 0.0" << std::endl;
            }
            particles.charge[i] = 0.0;  // Default charge
        }
    }
    
    if (rank == 0) {
        std::cout << "Assigned charges from topology" << std::endl;
    }
}

void MDSystem::setup_force_field_params() {
    if (!topology) {
        std::cerr << "Error: topology not loaded for force field setup" << std::endl;
        return;
    }
    
    // Resize force field parameter arrays
    ff_params.epsilon.resize(particles.num_particles);
    ff_params.sigma.resize(particles.num_particles);
    
    // Get LJ parameters for each atom from topology using public methods
    for (size_t i = 0; i < particles.num_particles; i++) {
        try {
            ff_params.epsilon[i] = topology->get_atom_epsilon(i);
            ff_params.sigma[i] = topology->get_atom_sigma(i);
        } catch (const std::runtime_error& e) {
            if (rank == 0) {
                std::cerr << "Warning: Cannot get LJ parameters for atom " << i << ": " << e.what() << std::endl;
                std::cerr << "Using default LJ parameters" << std::endl;
            }
            // Use default Argon parameters
            ff_params.epsilon[i] = 0.238; // kcal/mol
            ff_params.sigma[i] = 3.4;     // Å
        }
    }
    
    if (rank == 0 && debug_level(1)) {
        std::cout << "Force field setup: " << ff_params.epsilon.size() << " particles" << std::endl;
        if (debug_level(3) && !ff_params.epsilon.empty()) {
            std::cout << "Sample LJ parameters: epsilon=" << ff_params.epsilon[0] 
                      << " kcal/mol, sigma=" << ff_params.sigma[0] << " Å" << std::endl;
        }
    }
}

void MDSystem::initialize_force_manager(const std::map<std::string, std::string>& config_map) {
    // Parse force configuration from config map
    ForceManager::ForceConfig force_config = ForceManager::parse_config_from_map(config_map);

    // Determine backend based on compilation flags and device availability
    std::string backend = "cpu";  // Default fallback
    
#ifdef USE_CUDA
    // Check if CUDA devices are available
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaSuccess && device_count > 0) {
        backend = "cuda";
        std::cout << "Using CUDA backend (compiled with USE_CUDA, " << device_count << " devices found)" << std::endl;
    } else {
        backend = "cpu";
        std::cout << "CUDA compiled but no devices available (" << cudaGetErrorString(error) 
                  << "), falling back to CPU backend" << std::endl;
    }
#elif defined(USE_SYCL)
    backend = "sycl";
    std::cout << "Using SYCL backend (compiled with USE_SYCL)" << std::endl;
#else
    backend = "cpu";
    std::cout << "Using CPU backend" << std::endl;
#endif

    // Initialize force manager
    force_manager->initialize(force_config, backend);
    
    if (rank == 0 && debug_level(1)) {
        std::cout << "Force manager initialized with " << backend << " backend" << std::endl;
        if (debug_level(2)) {
            if (force_config.mode == ForceManager::ForceMode::BUILTIN) {
                std::cout << "Using built-in optimized force modules" << std::endl;
            } else {
                std::cout << "Using Python force interface" << std::endl;
            }
        }
    }
}

void MDSystem::compute_forces() {
    if (!force_manager) {
        throw std::runtime_error("Force manager not initialized");
    }
    
    force_manager->compute_forces(*this);
}
