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

MDSystem::MDSystem() : physical_cutoff(12.0), rank(0), size(1), local_start(0), local_end(0), d_particles(nullptr), use_gpu(false) {
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
    
    if (rank == 0) {
        std::cout << "Box dimensions: " << box.lx << " x " << box.ly << " x " << box.lz << std::endl;
        // Backend information is printed in initialize_force_manager
    }
}

void MDSystem::load_coordinates(const std::string& coord_file) {
    if (rank == 0) {
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
    
    current_ke = 0.0;
    for (size_t i = 0; i < particles.num_particles; i++) {
        Real v2 = particles.vx[i]*particles.vx[i] + particles.vy[i]*particles.vy[i] + particles.vz[i]*particles.vz[i];
        current_ke += 0.5 * particles.mass[i] * v2;
    }

    if (rank == 0) {
        std::cout << "Velocity initialization: target T = " << temperature 
                  << " K, target KE = " << target_ke << ", scale factor = " << scale_factor << std::endl;
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
    
    if (rank == 0) {
        std::cout << "Loaded " << atom_count << " particles from " << clean_filename << std::endl;
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
    
    if (rank == 0 && debug_level(2)) {
        std::cout << "Assigned masses from topology" << std::endl;
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
    
    if (rank == 0) {
        std::cout << "Force field setup: " << ff_params.epsilon.size() << " particles" << std::endl;
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
    
    if (rank == 0) {
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

// ============================================================================
// Halo Exchange Implementation
// ============================================================================

void MDSystem::setup_neighbor_ranks() {
#ifdef USE_MPI
    // Create a 3D Cartesian topology for the MPI processes
    int dims[3] = {0, 0, 0};
    int periods[3] = {box.periodic_x ? 1 : 0, box.periodic_y ? 1 : 0, box.periodic_z ? 1 : 0};
    
    // Let MPI decide the optimal grid dimensions
    MPI_Dims_create(size, 3, dims);
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    
    // Get the coordinates of this rank in the Cartesian grid
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    
    // Calculate local domain boundaries
    domain_min[0] = coords[0] * box.lx / dims[0];
    domain_min[1] = coords[1] * box.ly / dims[1];
    domain_min[2] = coords[2] * box.lz / dims[2];
    
    domain_max[0] = (coords[0] + 1) * box.lx / dims[0];
    domain_max[1] = (coords[1] + 1) * box.ly / dims[1];
    domain_max[2] = (coords[2] + 1) * box.lz / dims[2];
    
    // Identify neighbor ranks in all 26 directions
    num_neighbors = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue; // Skip self
                
                int neighbor_coords[3] = {coords[0] + i, coords[1] + j, coords[2] + k};
                int neighbor_rank;
                
                // MPI_Cart_rank returns MPI_PROC_NULL for out-of-bounds non-periodic
                MPI_Cart_rank(cart_comm, neighbor_coords, &neighbor_rank);
                
                if (neighbor_rank != MPI_PROC_NULL) {
                    neighbor_ranks[num_neighbors] = neighbor_rank;
                    num_neighbors++;
                }
            }
        }
    }
    
    // Free the Cartesian communicator
    MPI_Comm_free(&cart_comm);
    
#else
    num_neighbors = 0;
#endif
}

void MDSystem::identify_halo_particles() {
#ifdef USE_MPI
    halo_particles.clear();
    
    // Halo width = cutoff + verlet skin (if using Verlet list)
    Real halo_width = physical_cutoff + 2.0; // Add 2.0 Å as default skin
    
    // Identify particles near domain boundaries that need to be sent
    for (size_t i = 0; i < particles.num_particles; i++) {
        Real x = particles.x[i];
        Real y = particles.y[i];
        Real z = particles.z[i];
        
        // Check if particle is within halo width of any boundary
        bool in_halo = false;
        
        if (x < domain_min[0] + halo_width || x > domain_max[0] - halo_width) in_halo = true;
        if (y < domain_min[1] + halo_width || y > domain_max[1] - halo_width) in_halo = true;
        if (z < domain_min[2] + halo_width || z > domain_max[2] - halo_width) in_halo = true;
        
        if (in_halo) {
            halo_particles.push_back(i);
        }
    }
#endif
}

void MDSystem::exchange_halo_particles() {
#ifdef USE_MPI
    if (num_neighbors == 0) {
        return;  // No neighbors, nothing to exchange
    }
    
    // Resize neighbor_halos to store data from all neighbors
    neighbor_halos.resize(num_neighbors);
    
    std::vector<MPI_Request> send_count_requests(num_neighbors);
    std::vector<MPI_Request> recv_count_requests(num_neighbors);
    std::vector<MPI_Request> send_data_requests;
    std::vector<MPI_Request> recv_data_requests;
    
    std::vector<std::vector<Real>> send_buffers(num_neighbors);
    std::vector<std::vector<Real>> recv_buffers(num_neighbors);
    std::vector<int> send_counts(num_neighbors, 0);
    std::vector<int> recv_counts(num_neighbors, 0);  // Persistent storage for recv counts
    
    // Step 1: Pack data for all neighbors
    for (int n = 0; n < num_neighbors; n++) {
        pack_halo_data(neighbor_ranks[n], send_buffers[n]);
        send_counts[n] = send_buffers[n].size();
    }
    
    // Step 2: Exchange counts using non-blocking communication
    for (int n = 0; n < num_neighbors; n++) {
        // Post non-blocking receive for count
        MPI_Irecv(&recv_counts[n], 1, MPI_INT, neighbor_ranks[n], 
                  rank * 1000 + 0, MPI_COMM_WORLD, &recv_count_requests[n]);
        
        // Post non-blocking send for count
        MPI_Isend(&send_counts[n], 1, MPI_INT, neighbor_ranks[n], 
                  neighbor_ranks[n] * 1000 + 0, MPI_COMM_WORLD, &send_count_requests[n]);
    }
    
    // Step 3: Wait for count exchanges to complete
    if (num_neighbors > 0) {
        MPI_Waitall(num_neighbors, send_count_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(num_neighbors, recv_count_requests.data(), MPI_STATUSES_IGNORE);
    }
    
    // Step 4: Post receives for actual data
    for (int n = 0; n < num_neighbors; n++) {
        if (recv_counts[n] > 0) {
            recv_buffers[n].resize(recv_counts[n]);
            MPI_Request req;
            MPI_Irecv(recv_buffers[n].data(), recv_counts[n], MPI_DOUBLE, 
                     neighbor_ranks[n], rank * 1000 + 1, MPI_COMM_WORLD, &req);
            recv_data_requests.push_back(req);
        }
    }
    
    // Step 5: Send actual data
    for (int n = 0; n < num_neighbors; n++) {
        if (send_counts[n] > 0) {
            MPI_Request req;
            MPI_Isend(send_buffers[n].data(), send_counts[n], MPI_DOUBLE, 
                     neighbor_ranks[n], neighbor_ranks[n] * 1000 + 1, 
                     MPI_COMM_WORLD, &req);
            send_data_requests.push_back(req);
        }
    }
    
    // Step 6: Wait for all data exchanges to complete
    if (!recv_data_requests.empty()) {
        MPI_Waitall(recv_data_requests.size(), recv_data_requests.data(), MPI_STATUSES_IGNORE);
    }
    if (!send_data_requests.empty()) {
        MPI_Waitall(send_data_requests.size(), send_data_requests.data(), MPI_STATUSES_IGNORE);
    }
    
    // Step 7: Unpack received data
    for (int n = 0; n < num_neighbors; n++) {
        if (recv_counts[n] > 0) {
            unpack_halo_data(neighbor_ranks[n], recv_buffers[n]);
        }
    }
    
    if (debug_level(3)) {
        std::cout << "Rank " << rank << ": Halo exchange completed" << std::endl;
    }
#endif
}

void MDSystem::pack_halo_data(int neighbor_rank, std::vector<Real>& buffer) {
#ifdef USE_MPI
    buffer.clear();
    
    // Calculate which particles to send to this specific neighbor
    // For simplicity, we'll send all halo particles to all neighbors
    // (more sophisticated implementation would check spatial locality)
    
    int num_to_send = halo_particles.size();
    
    // Buffer format: [num_particles, x1, y1, z1, vx1, vy1, vz1, mass1, charge1, type1, x2, ...]
    // 9 values per particle (x, y, z, vx, vy, vz, mass, charge, type)
    buffer.resize(1 + num_to_send * 9);
    
    buffer[0] = static_cast<Real>(num_to_send);
    
    for (int i = 0; i < num_to_send; i++) {
        int idx = halo_particles[i];
        int offset = 1 + i * 9;
        
        buffer[offset + 0] = particles.x[idx];
        buffer[offset + 1] = particles.y[idx];
        buffer[offset + 2] = particles.z[idx];
        buffer[offset + 3] = particles.vx[idx];
        buffer[offset + 4] = particles.vy[idx];
        buffer[offset + 5] = particles.vz[idx];
        buffer[offset + 6] = particles.mass[idx];
        buffer[offset + 7] = particles.charge[idx];
        buffer[offset + 8] = static_cast<Real>(particles.type[idx]);
    }
    
#endif
}

void MDSystem::unpack_halo_data(int neighbor_rank, const std::vector<Real>& buffer) {
#ifdef USE_MPI
    if (buffer.empty()) return;
    
    int num_received = static_cast<int>(buffer[0]);
    
    // Find the neighbor index
    int neighbor_idx = -1;
    for (int n = 0; n < num_neighbors; n++) {
        if (neighbor_ranks[n] == neighbor_rank) {
            neighbor_idx = n;
            break;
        }
    }
    
    if (neighbor_idx < 0) return;
    
    // Resize the neighbor halo data structure
    neighbor_halos[neighbor_idx].resize(num_received);
    
    // Unpack particle data
    for (int i = 0; i < num_received; i++) {
        int offset = 1 + i * 9;
        
        neighbor_halos[neighbor_idx].x[i] = buffer[offset + 0];
        neighbor_halos[neighbor_idx].y[i] = buffer[offset + 1];
        neighbor_halos[neighbor_idx].z[i] = buffer[offset + 2];
        neighbor_halos[neighbor_idx].vx[i] = buffer[offset + 3];
        neighbor_halos[neighbor_idx].vy[i] = buffer[offset + 4];
        neighbor_halos[neighbor_idx].vz[i] = buffer[offset + 5];
        neighbor_halos[neighbor_idx].mass[i] = buffer[offset + 6];
        neighbor_halos[neighbor_idx].charge[i] = buffer[offset + 7];
        neighbor_halos[neighbor_idx].type[i] = static_cast<int>(buffer[offset + 8]);
    }
    
#endif
}

void MDSystem::update_ghost_positions() {
#ifdef USE_MPI
    // This function would update only positions of ghost particles
    // (used during MD integration when we don't need to re-exchange everything)
    
    // For now, this is a placeholder - typically called every timestep
    // while full exchange_halo_particles() is called less frequently
    
#endif
}
