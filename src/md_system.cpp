
#include "md_system.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <random>
#include <map>
#include <algorithm>  // For std::sort
#include <unistd.h>  // For getcwd
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

MDSystem::MDSystem() : physical_cutoff(12.0), rank(0), size(1), local_start(0), local_end(0), d_particles(nullptr), use_gpu(false), cuda_direct_halo_(false) {
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cart_comm = MPI_COMM_NULL;  // Initialize cart_comm
    cart_dims[0] = cart_dims[1] = cart_dims[2] = 0;
    cart_coords[0] = cart_coords[1] = cart_coords[2] = 0;
#endif
    topology = std::make_unique<Topology>();
    force_manager = std::make_unique<ForceManager>();
#ifdef USE_CUDA
    const char* env = std::getenv("MD_CUDA_DIRECT_HALO");
    if (env && std::string(env)=="1") cuda_direct_halo_ = true;
#endif
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
        if (value.empty() || key == "builtin_forcefield" || key == "python_force_interface" || key == "force_calculation") {
            current_section = key;
            continue;
        }
        
        // Add section prefix for nested keys
        std::string full_key = key;
        if (current_section == "builtin_forcefield") {
            full_key = "builtin_" + key;
        } else if (current_section == "python_force_interface") {
            full_key = "python_" + key;
        } else if (current_section == "force_calculation") {
            full_key = "force_calculation_" + key;
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
    
    // Initialize subdomain bounds for serial mode (will be updated by domain decomposition for MPI)
    // For serial: subdomain = entire box [0, L]
    box.xlo = 0.0;
    box.xhi = box.lx;
    box.ylo = 0.0;
    box.yhi = box.ly;
    box.zlo = 0.0;
    box.zhi = box.lz;
    
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
        std::cout << "Initializing velocities at T = " 
                  << temperature << " K for " 
                  << particles.num_particles << " particles" 
                  << std::endl;
    }

    // LAMMPS real units 상수
    // (g/mol)*(Å/fs)^2 -> kcal/mol

#ifdef USE_MPI
    if (rank == 0) {
#endif
        // 1) 가우시안 랜덤 속도 생성
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<Real> dist(0.0, 1.0);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            particles.vx[i] = dist(gen);
            particles.vy[i] = dist(gen);
            particles.vz[i] = dist(gen);
        }
        
        // 2) center-of-mass 속도 제거
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
        
        // 3) 현재 KE 계산 (먼저 g/mol·Å^2/fs^2 단위로)
        Real current_ke = 0.0; // 아직은 단위: (g/mol)*(Å/fs)^2
        for (size_t i = 0; i < particles.num_particles; i++) {
            Real v2 = particles.vx[i]*particles.vx[i]
                    + particles.vy[i]*particles.vy[i]
                    + particles.vz[i]*particles.vz[i];
            current_ke += 0.5 * particles.mass[i] * v2;
        }
        // LAMMPS real 유닛과 동일하게 kcal/mol 로 변환
        current_ke *= mvv2e;  // 이제 current_ke 단위: kcal/mol

        // 4) 타겟 KE (자유도 = 3N - 3 : COM 제거했으므로)
        const size_t dof = 3 * particles.num_particles - 3;
        Real target_ke = 0.5 * static_cast<Real>(dof) * kB * temperature;
        // = (dof/2) k_B T   (단위: kcal/mol)

        // 5) 스케일 팩터 계산 (둘 다 kcal/mol 이라 단위 일치)
        Real scale_factor = std::sqrt(target_ke / current_ke);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            particles.vx[i] *= scale_factor;
            particles.vy[i] *= scale_factor;
            particles.vz[i] *= scale_factor;
        }
        
        // 6) 체크용으로 다시 KE 계산
        current_ke = 0.0;
        for (size_t i = 0; i < particles.num_particles; i++) {
            Real v2 = particles.vx[i]*particles.vx[i]
                    + particles.vy[i]*particles.vy[i]
                    + particles.vz[i]*particles.vz[i];
            current_ke += 0.5 * particles.mass[i] * v2;
        }
        current_ke *= mvv2e;  // kcal/mol

        // 실제 온도도 한 번 출력해 보면 디버깅에 좋음
        Real current_T = (2.0 * current_ke) / (kB * static_cast<Real>(dof));

        if (rank == 0) {
            std::cout << "Velocity initialization: target T = " << temperature 
                      << " K, target KE = " << target_ke 
                      << " kcal/mol, scale factor = " << scale_factor
                      << ", resulting T = " << current_T << " K"
                      << std::endl;
        }
#ifdef USE_MPI
    }

    MPI_Bcast(particles.vx.data(), particles.num_particles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(particles.vy.data(), particles.num_particles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(particles.vz.data(), particles.num_particles, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Velocities broadcast to all ranks" << std::endl;
    }
#endif
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
    
    // For now, each rank handles its own subdomain = entire box
    // TODO: Implement spatial domain decomposition for MPI
    // Currently using particle-based decomposition, so subdomain = global box
    // This will be updated when spatial decomposition is fully implemented
    if (size > 1) {
        // MPI mode: for now, use global box as subdomain (will be refined later)
        box.xlo = 0.0;
        box.xhi = box.lx;
        box.ylo = 0.0;
        box.yhi = box.ly;
        box.zlo = 0.0;
        box.zhi = box.lz;
    }
    
    if (rank == 0) {
        std::cout << "Domain decomposition: " << size << " ranks, " 
                  << total_particles << " particles" << std::endl;
        std::cout << "  Rank 0: particles [" << local_start << ", " << local_end 
                  << ") = " << (local_end - local_start) << " local" << std::endl;
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
    
    // Resize particle arrays (also assigns sequential global IDs via ParticleData::resize later if used)
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
    particles.global_id.resize(num_atoms);
    particles.num_particles = num_atoms;  // Set the particle count
    for (int i = 0; i < num_atoms; ++i) particles.global_id[i] = i;
    
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
    
    // Calculate center of mass of loaded coordinates
    Real com_x = 0.0, com_y = 0.0, com_z = 0.0;
    for (int i = 0; i < atom_count; ++i) {
        com_x += particles.x[i];
        com_y += particles.y[i];
        com_z += particles.z[i];
    }
    com_x /= atom_count;
    com_y /= atom_count;
    com_z /= atom_count;
    
    // Shift coordinates to center in box
    Real target_x = box.lx / 2.0;
    Real target_y = box.ly / 2.0;
    Real target_z = box.lz / 2.0;
    
    for (int i = 0; i < atom_count; ++i) {
        particles.x[i] += (target_x - com_x);
        particles.y[i] += (target_y - com_y);
        particles.z[i] += (target_z - com_z);
    }
    
    if (rank == 0) {
        std::cout << "Loaded " << atom_count << " particles from " << clean_filename << std::endl;
        std::cout << "Shifted coordinates: COM (" << com_x << ", " << com_y << ", " << com_z 
                  << ") -> Box center (" << target_x << ", " << target_y << ", " << target_z << ")" << std::endl;
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
    
    // Extract unique atom types and build type-based parameter arrays
    std::map<int, std::pair<Real, Real>> type_params;  // type -> (epsilon, sigma)
    
    for (size_t i = 0; i < particles.num_particles; i++) {
        int type = particles.type[i];
        if (type_params.find(type) == type_params.end()) {
            try {
                Real eps = topology->get_atom_epsilon(i);
                Real sig = topology->get_atom_sigma(i);
                type_params[type] = std::make_pair(eps, sig);
            } catch (const std::runtime_error& e) {
                if (rank == 0) {
                    std::cerr << "Warning: Cannot get LJ parameters for atom " << i << " (type " << type << "): " << e.what() << std::endl;
                    std::cerr << "Using default LJ parameters" << std::endl;
                }
                // Use default Argon parameters
                type_params[type] = std::make_pair(0.238, 3.4);
            }
        }
    }
    
    // Build type-based arrays
    ff_params.num_atom_types = type_params.size();
    ff_params.epsilon.resize(ff_params.num_atom_types);
    ff_params.sigma.resize(ff_params.num_atom_types);
    
    int type_idx = 0;
    for (const auto& pair : type_params) {
        ff_params.epsilon[type_idx] = pair.second.first;
        ff_params.sigma[type_idx] = pair.second.second;
        type_idx++;
    }
    
    // Initialize pre-computed mixing rule lookup tables (type-based, not particle-based)
    // This avoids repeated sqrt and arithmetic operations during force calculation
    ff_params.initialize_mixing_tables();
    
    if (rank == 0) {
        std::cout << "Force field setup: " << particles.num_particles << " particles, " 
                  << ff_params.num_atom_types << " atom types" << std::endl;
        if (ff_params.num_atom_types > 0 && !ff_params.epsilon_mixed.empty()) {
            std::cout << "Mixing rule tables initialized: " << ff_params.num_atom_types 
                      << "x" << ff_params.num_atom_types << " elements" << std::endl;
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
    
    if (rank == 0) {
        std::cout << "Force manager initialized with " << backend << " backend" << std::endl;
    }
}

void MDSystem::compute_forces() {
    if (!force_manager) {
        throw std::runtime_error("Force manager not initialized");
    }
    
    force_manager->compute_forces(*this);
}

// NEW HALO EXCHANGE IMPLEMENTATION
void MDSystem::setup_neighbor_ranks() {
#ifdef USE_MPI
    cart_dims[0] = cart_dims[1] = cart_dims[2] = 0;
    MPI_Dims_create(size, 3, cart_dims);
    int periods[3] = {1,1,1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, cart_dims, periods, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, cart_coords);
    neighbor_info.clear();
    for (int dx=-1; dx<=1; ++dx) {
        for (int dy=-1; dy<=1; ++dy) {
            for (int dz=-1; dz<=1; ++dz) {
                if (dx==0 && dy==0 && dz==0) continue;
                int coords[3] = {cart_coords[0]+dx, cart_coords[1]+dy, cart_coords[2]+dz};
                int nrank; MPI_Cart_rank(cart_comm, coords, &nrank);
                neighbor_info.push_back({nrank, dx, dy, dz});
            }
        }
    }
    num_neighbors = static_cast<int>(neighbor_info.size());
    if (rank==0) {
        std::cout << "[Halo] Cartesian topology " << cart_dims[0] << "x" << cart_dims[1] << "x" << cart_dims[2]
                  << ", neighbors=" << num_neighbors << std::endl;
    }
    update_domain_bounds();
    halo_width_ = physical_cutoff + 2.0; // cutoff + skin (can be configurable)
    if (rank==0) {
        std::cout << "[Halo] sizeof(HaloParticle)=" << sizeof(HaloParticle) << " bytes" << std::endl;
    }
#endif
}

void MDSystem::update_domain_bounds() {
#ifdef USE_MPI
    // Simple spatial split: equal division; assumes global box already set
    Real lx = box.lx / cart_dims[0];
    Real ly = box.ly / cart_dims[1];
    Real lz = box.lz / cart_dims[2];
    domain_min[0] = cart_coords[0] * lx; domain_max[0] = domain_min[0] + lx;
    domain_min[1] = cart_coords[1] * ly; domain_max[1] = domain_min[1] + ly;
    domain_min[2] = cart_coords[2] * lz; domain_max[2] = domain_min[2] + lz;
#endif
}

void MDSystem::build_halo_send_buffers() {
#ifdef USE_MPI
    halo_send_buffers.clear();
    halo_send_buffers.resize(neighbor_info.size());
    // MPI-AWARE: Only send LOCAL particles, NOT ghost particles
    const size_t local_N = get_local_particles();
    
    for (size_t idx=0; idx<neighbor_info.size(); ++idx) {
        auto &nb = neighbor_info[idx];
        auto &buf = halo_send_buffers[idx];
        buf.clear();
        buf.reserve(128);
        
        // Loop over LOCAL particles only [0, local_N)
        for (size_t i=0; i<local_N; ++i) {
            Real x = particles.x[i];
            Real y = particles.y[i];
            Real z = particles.z[i];
            
            // Start with current (possibly wrapped) coordinates
            Real x_send = x, y_send = y, z_send = z;
            
            // Check each direction independently - ALL active directions must match
            bool send_x = true, send_y = true, send_z = true;
            
            // X-direction
            if (nb.dx < 0) {  // Left neighbor - need particles near left boundary
                if (x < domain_min[0] + halo_width_) {
                    send_x = true;
                } else if (box.periodic_x && x > box.lx - halo_width_) {
                    // Particle wrapped to far right after PBC
                    send_x = true;
                    x_send = x - box.lx;  // Shift to left periodic image
                } else {
                    send_x = false;
                }
            } else if (nb.dx > 0) {  // Right neighbor - need particles near right boundary
                if (x > domain_max[0] - halo_width_) {
                    send_x = true;
                } else if (box.periodic_x && x < halo_width_) {
                    // Particle wrapped to far left after PBC
                    send_x = true;
                    x_send = x + box.lx;  // Shift to right periodic image
                } else {
                    send_x = false;
                }
            }
            // else nb.dx == 0: don't care about x, send_x stays true
            
            // Y-direction
            if (nb.dy < 0) {  // Bottom neighbor
                if (y < domain_min[1] + halo_width_) {
                    send_y = true;
                } else if (box.periodic_y && y > box.ly - halo_width_) {
                    send_y = true;
                    y_send = y - box.ly;
                } else {
                    send_y = false;
                }
            } else if (nb.dy > 0) {  // Top neighbor
                if (y > domain_max[1] - halo_width_) {
                    send_y = true;
                } else if (box.periodic_y && y < halo_width_) {
                    send_y = true;
                    y_send = y + box.ly;
                } else {
                    send_y = false;
                }
            }
            
            // Z-direction
            if (nb.dz < 0) {  // Back neighbor
                if (z < domain_min[2] + halo_width_) {
                    send_z = true;
                } else if (box.periodic_z && z > box.lz - halo_width_) {
                    send_z = true;
                    z_send = z - box.lz;
                } else {
                    send_z = false;
                }
            } else if (nb.dz > 0) {  // Front neighbor
                if (z > domain_max[2] - halo_width_) {
                    send_z = true;
                } else if (box.periodic_z && z < halo_width_) {
                    send_z = true;
                    z_send = z + box.lz;
                } else {
                    send_z = false;
                }
            }
            
            // Send only if ALL active directions match
            if (!send_x || !send_y || !send_z) continue;
            
            // HaloParticle fields: x,y,z, vx,vy,vz, fx,fy,fz, mass,charge, type, global_id
            buf.push_back({x_send, y_send, z_send, 
                          particles.vx[i], particles.vy[i], particles.vz[i], 
                          0.0, 0.0, 0.0,
                          particles.mass[i], particles.charge[i], 
                          particles.type[i], particles.global_id[i]});
        }
    }
#endif
}

size_t MDSystem::get_total_halo_particles() const {
    size_t total=0; for (auto &b: halo_recv_buffers) total += b.size(); return total; }

void MDSystem::perform_halo_exchange() {
#ifdef USE_MPI
    build_halo_send_buffers();
    halo_recv_buffers.clear();
    halo_recv_buffers.resize(neighbor_info.size());
    std::vector<MPI_Request> reqs;
    reqs.reserve(neighbor_info.size()*2);
    // Phase 1: exchange counts
    std::vector<int> send_counts(neighbor_info.size()), recv_counts(neighbor_info.size());
    for (size_t n=0; n<neighbor_info.size(); ++n) {
        send_counts[n] = static_cast<int>(halo_send_buffers[n].size());
        MPI_Request r1,r2;
        MPI_Irecv(&recv_counts[n],1,MPI_INT,neighbor_info[n].rank, 500, MPI_COMM_WORLD,&r1);
        MPI_Isend(&send_counts[n],1,MPI_INT,neighbor_info[n].rank, 500, MPI_COMM_WORLD,&r2);
        reqs.push_back(r1); reqs.push_back(r2);
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();
    // Prepare MPI datatype for HaloParticle
    MPI_Datatype halo_type = get_halo_particle_type();
    // Phase 2: data exchange (derived datatype)
    for (size_t n=0; n<neighbor_info.size(); ++n) {
        int sc = send_counts[n]; int rc = recv_counts[n];
        if (rc > 0) halo_recv_buffers[n].resize(rc);
        HaloParticle* send_ptr = sc ? halo_send_buffers[n].data() : nullptr;
        HaloParticle* recv_ptr = rc ? halo_recv_buffers[n].data() : nullptr;
#ifdef USE_CUDA
        // CUDA-aware MPI path: allocate device buffers if requested and GPU enabled
        if (cuda_direct_halo_ && use_gpu) {
            if (sc) {
                void* dptr_send = nullptr;
                cudaMalloc(&dptr_send, sc * sizeof(HaloParticle));
                cudaMemcpy(dptr_send, send_ptr, sc * sizeof(HaloParticle), cudaMemcpyHostToDevice);
                d_halo_send.push_back(dptr_send);
                send_ptr = static_cast<HaloParticle*>(dptr_send); // use device pointer in MPI call
            } else {
                d_halo_send.push_back(nullptr);
            }
            if (rc) {
                void* dptr_recv = nullptr;
                cudaMalloc(&dptr_recv, rc * sizeof(HaloParticle));
                d_halo_recv.push_back(dptr_recv);
                recv_ptr = static_cast<HaloParticle*>(dptr_recv);
            } else {
                d_halo_recv.push_back(nullptr);
            }
        }
#endif
        MPI_Request r1,r2;
        if (rc) MPI_Irecv(recv_ptr, rc, halo_type, neighbor_info[n].rank, 600, MPI_COMM_WORLD, &r1);
        if (sc) MPI_Isend(send_ptr, sc, halo_type, neighbor_info[n].rank, 600, MPI_COMM_WORLD, &r2);
        if (rc) reqs.push_back(r1);
        if (sc) reqs.push_back(r2);
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
#ifdef USE_CUDA
    // Copy received device buffers back to host and free device memory
    if (cuda_direct_halo_ && use_gpu) {
        for (size_t n=0; n<neighbor_info.size(); ++n) {
            void* drecv = d_halo_recv.size()>n ? d_halo_recv[n] : nullptr;
            int rc = static_cast<int>(halo_recv_buffers[n].size());
            if (drecv && rc>0) {
                cudaMemcpy(halo_recv_buffers[n].data(), drecv, rc * sizeof(HaloParticle), cudaMemcpyDeviceToHost);
                cudaFree(drecv);
            }
            void* dsend = d_halo_send.size()>n ? d_halo_send[n] : nullptr;
            if (dsend) cudaFree(dsend);
        }
        d_halo_recv.clear();
        d_halo_send.clear();
    }
#endif
#endif
}

void MDSystem::perform_reverse_halo_exchange() {
#ifdef USE_MPI
    // Build reverse send buffers: pack ghost forces back to their original ranks
    std::vector<std::vector<HaloParticle>> reverse_send_buffers(neighbor_info.size());
    size_t local_N = particles.num_particles - ghost_count_;
    size_t ghost_start = local_N;
    size_t ghost_idx = 0;
    for (size_t n = 0; n < neighbor_info.size(); ++n) {
        auto& buf = halo_recv_buffers[n]; // original received buffer to match order
        reverse_send_buffers[n].resize(buf.size());
        for (size_t k = 0; k < buf.size(); ++k) {
            size_t global_idx = ghost_start + ghost_idx;
            reverse_send_buffers[n][k].fx = particles.fx[global_idx];
            reverse_send_buffers[n][k].fy = particles.fy[global_idx];
            reverse_send_buffers[n][k].fz = particles.fz[global_idx];
            reverse_send_buffers[n][k].global_id = particles.global_id[global_idx];
            ghost_idx++;
        }
    }

    // Exchange force updates
    std::vector<MPI_Request> reqs;
    reqs.reserve(neighbor_info.size() * 2);
    std::vector<int> send_counts(neighbor_info.size()), recv_counts(neighbor_info.size());
    for (size_t n = 0; n < neighbor_info.size(); ++n) {
        send_counts[n] = static_cast<int>(reverse_send_buffers[n].size());
        MPI_Request r1, r2;
        MPI_Irecv(&recv_counts[n], 1, MPI_INT, neighbor_info[n].rank, 700, MPI_COMM_WORLD, &r1);
        MPI_Isend(&send_counts[n], 1, MPI_INT, neighbor_info[n].rank, 700, MPI_COMM_WORLD, &r2);
        reqs.push_back(r1);
        reqs.push_back(r2);
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // Prepare buffers for force data exchange
    std::vector<std::vector<HaloParticle>> reverse_recv_buffers(neighbor_info.size());
    for (size_t n = 0; n < neighbor_info.size(); ++n) {
        int rc = recv_counts[n];
        if (rc > 0) reverse_recv_buffers[n].resize(rc);
        HaloParticle* send_ptr = send_counts[n] ? reverse_send_buffers[n].data() : nullptr;
        HaloParticle* recv_ptr = rc ? reverse_recv_buffers[n].data() : nullptr;
        MPI_Request r1, r2;
        if (rc) MPI_Irecv(recv_ptr, rc, get_halo_particle_type(), neighbor_info[n].rank, 800, MPI_COMM_WORLD, &r1);
        if (send_counts[n]) MPI_Isend(send_ptr, send_counts[n], get_halo_particle_type(), neighbor_info[n].rank, 800, MPI_COMM_WORLD, &r2);
        if (rc) reqs.push_back(r1);
        if (send_counts[n]) reqs.push_back(r2);
    }
    if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // Accumulate received forces back to local particles
    for (size_t n = 0; n < neighbor_info.size(); ++n) {
        for (auto& hp : reverse_recv_buffers[n]) {
            // Find the local particle by global_id
            for (size_t i = 0; i < local_N; ++i) {
                if (particles.global_id[i] == hp.global_id) {
                    particles.fx[i] += hp.fx;
                    particles.fy[i] += hp.fy;
                    particles.fz[i] += hp.fz;
                    break;
                }
            }
        }
    }
#endif
}

void MDSystem::assemble_ghost_particles() {
    // Append ghosts to ParticleData arrays (structure-of-arrays) so neighbor list can include them
    remove_ghost_particles(); // ensure clean state
    ghost_count_ = 0;
    size_t local_N = particles.num_particles;
    size_t total_ghosts = 0;
    for (auto &buf : halo_recv_buffers) total_ghosts += buf.size();
    if (total_ghosts == 0) return;
    particles.x.reserve(local_N + total_ghosts);
    particles.y.reserve(local_N + total_ghosts);
    particles.z.reserve(local_N + total_ghosts);
    particles.vx.reserve(local_N + total_ghosts);
    particles.vy.reserve(local_N + total_ghosts);
    particles.vz.reserve(local_N + total_ghosts);
    particles.fx.reserve(local_N + total_ghosts);
    particles.fy.reserve(local_N + total_ghosts);
    particles.fz.reserve(local_N + total_ghosts);
    particles.mass.reserve(local_N + total_ghosts);
    particles.charge.reserve(local_N + total_ghosts);
    particles.type.reserve(local_N + total_ghosts);
    
    for (auto &buf : halo_recv_buffers) {
        for (auto &hp : buf) {
            // Check distance to all local particles
            for (size_t i = 0; i < local_N; i++) {
                Real dx = hp.x - particles.x[i];
                Real dy = hp.y - particles.y[i];
                Real dz = hp.z - particles.z[i];
                
                // Apply minimum image
                box.apply_minimum_image(dx, dy, dz);
                
            }
            
            particles.x.push_back(hp.x);
            particles.y.push_back(hp.y);
            particles.z.push_back(hp.z);
            particles.vx.push_back(hp.vx);
            particles.vy.push_back(hp.vy);
            particles.vz.push_back(hp.vz);
            particles.fx.push_back(0.0); // ghost force starts at 0, not accumulated back
            particles.fy.push_back(0.0);
            particles.fz.push_back(0.0);
            particles.mass.push_back(hp.mass);
            particles.charge.push_back(hp.charge);
            particles.type.push_back(hp.type);
            particles.global_id.push_back(hp.global_id);
            ghost_count_++;
        }
    }
    particles.num_particles = local_N + ghost_count_;
}

void MDSystem::remove_ghost_particles() {
    if (ghost_count_ == 0) return;
    size_t new_size = particles.num_particles - ghost_count_;
    auto shrink = [new_size](auto &vec){ if (vec.size() > new_size) vec.resize(new_size); };
    shrink(particles.x); shrink(particles.y); shrink(particles.z);
    shrink(particles.vx); shrink(particles.vy); shrink(particles.vz);
    shrink(particles.fx); shrink(particles.fy); shrink(particles.fz);
    shrink(particles.mass); shrink(particles.charge); shrink(particles.type); shrink(particles.global_id);
    particles.num_particles = new_size;
    ghost_count_ = 0;
}

#ifdef USE_MPI
// Static MPI datatype instance
static MPI_Datatype g_halo_particle_type = MPI_DATATYPE_NULL;

MPI_Datatype MDSystem::get_halo_particle_type() {
    if (g_halo_particle_type != MPI_DATATYPE_NULL) return g_halo_particle_type;
    // Define HaloParticle layout
    
    // Actually we have: x,y,z,vx,vy,vz,fx,fy,fz,mass,charge,type,global_id -> 13 fields
    // We'll group reals and ints for simpler struct: use explicit offsets.
    HaloParticle dummy = {};  // Initialize to avoid uninitialized warning
    MPI_Aint base; MPI_Get_address(&dummy, &base);
    MPI_Aint disps[13];
    MPI_Get_address(&dummy.x, &disps[0]);
    MPI_Get_address(&dummy.y, &disps[1]);
    MPI_Get_address(&dummy.z, &disps[2]);
    MPI_Get_address(&dummy.vx, &disps[3]);
    MPI_Get_address(&dummy.vy, &disps[4]);
    MPI_Get_address(&dummy.vz, &disps[5]);
    MPI_Get_address(&dummy.fx, &disps[6]);
    MPI_Get_address(&dummy.fy, &disps[7]);
    MPI_Get_address(&dummy.fz, &disps[8]);
    MPI_Get_address(&dummy.mass, &disps[9]);
    MPI_Get_address(&dummy.charge, &disps[10]);
    MPI_Get_address(&dummy.type, &disps[11]);
    MPI_Get_address(&dummy.global_id, &disps[12]);
    for (int i=0;i<13;++i) disps[i] -= base;
    int bl[13] = {1,1,1,1,1,1,1,1,1,1,1,1,1};
    MPI_Datatype types[13] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_INT,MPI_INT};
    MPI_Type_create_struct(13, bl, disps, types, &g_halo_particle_type);
    MPI_Type_commit(&g_halo_particle_type);
    return g_halo_particle_type;
}

void MDSystem::free_halo_particle_type() {
    if (g_halo_particle_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&g_halo_particle_type);
        g_halo_particle_type = MPI_DATATYPE_NULL;
    }
}
#endif
