
#include "md_system.h"
#include "force_calculator_cuda.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <random>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>  // For std::sort
#include <numeric>    // For std::iota
#include <cstdint>
#include <unistd.h>  // For getcwd
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void MDSystem::resize_particle_storage(size_t count) {
    particles.x.resize(count); particles.y.resize(count); particles.z.resize(count);
    particles.vx.resize(count); particles.vy.resize(count); particles.vz.resize(count);
    particles.fx.resize(count); particles.fy.resize(count); particles.fz.resize(count);
    particles.mass.resize(count); particles.charge.resize(count);
    particles.type.resize(count); particles.global_id.resize(count);
    particles.num_particles = count;
}

MDSystem::MDSystem() : physical_cutoff(12.0), rank(0), size(1), local_start(0), local_end(0), d_particles(nullptr), use_gpu(false), cuda_direct_halo_(false) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    // MPI is already initialized in main.cpp
    // Just check thread support level
    int provided;
    MPI_Query_thread(&provided);
    
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
            // Keep backward compatibility with existing inputs that already
            // include the force_calculation_ prefix inside the section.
            if (key.rfind("force_calculation_", 0) == 0) {
                full_key = key;
            } else {
                full_key = "force_calculation_" + key;
            }
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

    if (rank == 0) {
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
    }

    if (rank == 0) {
        std::cout << "Velocities initialized on rank 0; owned slices will be scattered" << std::endl;
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
    
    int load_ok = 0;
    if (rank == 0) {
        load_ok = topology->load_topology(clean_filename, rank) ? 1 : 0;
    }
    MPI_Bcast(&load_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!load_ok) {
        throw std::runtime_error("Failed to load topology file: " + clean_filename);
    }

    topology->broadcast(MPI_COMM_WORLD, 0);

    // Only root performs the O(Nglobal) atom-property lookup.  The resulting
    // per-particle fields are distributed before dsDecomp compacts each rank.
    if (rank == 0) {
        assign_masses_from_topology();
        assign_charges_from_topology();
    }
}

void MDSystem::setup_domain_decomposition() {
    if (size == 1) {
        local_start = 0;
        local_end   = particles.num_particles;
        particles.reset_counts(particles.num_particles, 0);
        box.xlo = 0.0; box.xhi = box.lx;
        box.ylo = 0.0; box.yhi = box.ly;
        box.zlo = 0.0; box.zhi = box.lz;
        refresh_local_bonded_topology();
        return;
    }

    // dsDecomp owns the Cartesian topology and local-domain geometry.
    initialize_decomposition_runtime();
    // Root alone owns the global input arrays.  It classifies particles with
    // dsDecomp, groups them by destination rank, and scatters only the owned
    // slices.  Non-root ranks never allocate the global particle set.
    std::uint64_t total_particles_u64 =
        (rank == 0) ? static_cast<std::uint64_t>(particles.num_particles) : 0;
    MPI_Bcast(&total_particles_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const size_t total_particles = static_cast<size_t>(total_particles_u64);

    std::vector<int> counts(size, 0), displacements(size, 0);
    if (rank == 0) {
        std::vector<int> owners(total_particles);
        std::vector<size_t> sorted_indices(total_particles);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        for (size_t i = 0; i < total_particles; ++i) {
            owners[i] = decomp_runtime_->owner_rank(
                dsdecomp::Vec3d{particles.x[i], particles.y[i], particles.z[i]});
            ++counts[owners[i]];
        }
        for (int r = 1; r < size; ++r) displacements[r] = displacements[r - 1] + counts[r - 1];
        std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
            [&](size_t a, size_t b) { return owners[a] < owners[b]; });

        auto reorder = [&](auto& vec) {
            auto original = vec;
            for (size_t i = 0; i < total_particles; ++i) vec[i] = original[sorted_indices[i]];
        };
        reorder(particles.x); reorder(particles.y); reorder(particles.z);
        reorder(particles.vx); reorder(particles.vy); reorder(particles.vz);
        reorder(particles.fx); reorder(particles.fy); reorder(particles.fz);
        reorder(particles.mass); reorder(particles.charge);
        reorder(particles.type); reorder(particles.global_id);
    }

    int local_count = 0;
    MPI_Scatter(rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
                &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto scatter_real = [&](std::vector<Real>& values) {
        std::vector<Real> local(static_cast<size_t>(local_count));
        MPI_Scatterv(rank == 0 ? values.data() : nullptr,
                     rank == 0 ? counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_DOUBLE, local.data(), local_count, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        values.swap(local);
    };
    auto scatter_int = [&](std::vector<int>& values) {
        std::vector<int> local(static_cast<size_t>(local_count));
        MPI_Scatterv(rank == 0 ? values.data() : nullptr,
                     rank == 0 ? counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_INT, local.data(), local_count, MPI_INT,
                     0, MPI_COMM_WORLD);
        values.swap(local);
    };
    scatter_real(particles.x); scatter_real(particles.y); scatter_real(particles.z);
    scatter_real(particles.vx); scatter_real(particles.vy); scatter_real(particles.vz);
    scatter_real(particles.fx); scatter_real(particles.fy); scatter_real(particles.fz);
    scatter_real(particles.mass); scatter_real(particles.charge);
    scatter_int(particles.type); scatter_int(particles.global_id);

    num_local_particles_ = static_cast<size_t>(local_count);
    particles.num_particles = num_local_particles_;
    particles.reset_counts(num_local_particles_, 0);
    local_start = 0;
    local_end = static_cast<int>(num_local_particles_);
    refresh_local_bonded_topology();

    if (rank == 0) {
        const auto& dimensions = decomp_runtime_->topology().dimensions();
        std::cout << "Spatial domain decomposition: " << dimensions[0] << "x" << dimensions[1]
                  << "x" << dimensions[2] << " grid, " << total_particles << " particles" << std::endl;
        std::cout << "  Rank 0: " << num_local_particles_ << " local particles" << std::endl;
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
    // Root retains the global arrays until setup_domain_decomposition scatters
    // owned slices. Non-root ranks do not allocate or parse the global input.
    if (rank != 0) {
        resize_particle_storage(0);
        return;
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

    // Build string type name → integer type ID mapping
    // (type IDs are assigned in order of first occurrence, starting from 0)
    std::map<std::string, int> type_name_to_id;
    int next_type_id = 0;
    
    for (int i = 0; i < static_cast<int>(particles.mass.size()); ++i) {
        try {
            const std::string& type_name = topology->get_atom_type_name(i);
            
            // Assign integer type ID
            if (!type_name.empty()) {
                auto it = type_name_to_id.find(type_name);
                if (it == type_name_to_id.end()) {
                    type_name_to_id[type_name] = next_type_id++;
                }
                particles.type[i] = type_name_to_id[type_name];
            }
            
            particles.mass[i] = topology->get_atom_mass(i);
        } catch (const std::runtime_error& e) {
            if (rank == 0) {
                std::cerr << "Warning: Cannot get mass for atom " << i << ": " << e.what() << std::endl;
                std::cerr << "Using default mass 1.0" << std::endl;
            }
            particles.mass[i] = 1.0;  // Default mass
        }
    }
    
    if (rank == 0 && !type_name_to_id.empty()) {
        std::cout << "Atom type ID assignment:" << std::endl;
        for (const auto& kv : type_name_to_id) {
            std::cout << "  " << kv.first << " -> type " << kv.second << std::endl;
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
    
    if (rank == 0) {
        // Extract unique atom types and build type-based parameter arrays once.
        std::map<int, std::pair<Real, Real>> type_params;  // type -> (epsilon, sigma)
        for (size_t i = 0; i < particles.num_particles; i++) {
            int type = particles.type[i];
            if (type_params.find(type) == type_params.end()) {
                try {
                    Real eps = topology->get_atom_epsilon(static_cast<int>(i));
                    Real sig = topology->get_atom_sigma(static_cast<int>(i));
                    type_params[type] = std::make_pair(eps, sig);
                } catch (const std::runtime_error& e) {
                    std::cerr << "Warning: Cannot get LJ parameters for atom " << i
                              << " (type " << type << "): " << e.what() << std::endl;
                    std::cerr << "Using default LJ parameters" << std::endl;
                    type_params[type] = std::make_pair(0.238, 3.4);
                }
            }
        }

        ff_params.num_atom_types = static_cast<int>(type_params.size());
        ff_params.epsilon.resize(static_cast<size_t>(ff_params.num_atom_types));
        ff_params.sigma.resize(static_cast<size_t>(ff_params.num_atom_types));
        int type_idx = 0;
        for (const auto& pair : type_params) {
            ff_params.epsilon[type_idx] = pair.second.first;
            ff_params.sigma[type_idx] = pair.second.second;
            ++type_idx;
        }
    }

    MPI_Bcast(&ff_params.num_atom_types, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        ff_params.epsilon.resize(static_cast<size_t>(ff_params.num_atom_types));
        ff_params.sigma.resize(static_cast<size_t>(ff_params.num_atom_types));
    }
    MPI_Bcast(ff_params.epsilon.data(), ff_params.num_atom_types, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ff_params.sigma.data(), ff_params.num_atom_types, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

void MDSystem::initialize_decomposition_runtime() {
    dsdecomp::Config config;
    config.domain.lower = {0.0, 0.0, 0.0};
    config.domain.upper = {box.lx, box.ly, box.lz};
    // Preserve the existing algorithm: the MPI Cartesian topology has always
    // been periodic in all three dimensions, independently of force PBC flags.
    config.domain.periodic = {true, true, true};
    config.process_grid.dimensions = {0, 0, 0};
    config.process_grid.allow_rank_reorder = false;
    // Keep the established MPI_Dims_create behavior by default. The planner is
    // opt-in so upgrading the library cannot silently change rank ownership.
    if (const char* policy = std::getenv("DSDECOMP_GRID_POLICY")) {
        const std::string selected(policy);
        if (selected == "balanced") {
            config.grid_policy = dsdecomp::GridPolicy::Balanced;
        } else if (selected == "topology" || selected == "topology-aware") {
            config.grid_policy = dsdecomp::GridPolicy::TopologyAware;
            if (const char* weight = std::getenv("DSDECOMP_INTER_NODE_WEIGHT")) {
                const double parsed = std::atof(weight);
                if (parsed > 0.0) config.topology_info.inter_node_weight = parsed;
            }
            if (const char* weight = std::getenv("DSDECOMP_INTRA_NODE_WEIGHT")) {
                const double parsed = std::atof(weight);
                if (parsed > 0.0) config.topology_info.intra_node_weight = parsed;
            }
        }
    }
    config.halo.width = physical_cutoff + verlet_skin_ + 1.0;
    config.halo.include_edges_and_corners = true;
#ifdef USE_CUDA
    config.execution = dsdecomp::ExecutionMode::Cuda;
    // GPU-resident mode is mandatory for a CUDA DeepScaleMD build. A backend
    // that cannot consume device buffers must fail instead of staging them.
    config.advanced.allow_host_staging = false;
#else
    config.execution = dsdecomp::ExecutionMode::Host;
#endif

    decomp_runtime_ = std::make_unique<dsdecomp::Decomp>(MPI_COMM_WORLD, config);

    const auto& topology = decomp_runtime_->topology();
    if (rank==0) {
        const auto& dimensions = topology.dimensions();
        std::cout << "[Halo] Cartesian topology " << dimensions[0] << "x"
                  << dimensions[1] << "x" << dimensions[2]
                  << ", neighbors=" << topology.neighbors().size() << std::endl;
    }
    sync_decomposition_domain();
    if (rank==0) {
        std::cout << "[Halo] sizeof(ParticleRecord)=" << sizeof(dsdecomp::ParticleRecord)
                  << " bytes, halo_width=" << decomp_runtime_->halo_width() << std::endl;
    }
}

void MDSystem::sync_decomposition_domain() {
    const Real halo_width = physical_cutoff + verlet_skin_ + 1.0;
    // For single-rank MPI the full box is the domain, but the bounds still
    // need refreshing after an NPT box change.
    if (size == 1) {
        box.xlo = box.ylo = box.zlo = 0.0;
        box.xhi = box.lx; box.yhi = box.ly; box.zhi = box.lz;
        if (decomp_runtime_) {
            dsdecomp::GlobalDomain domain;
            domain.lower = {0.0, 0.0, 0.0};
            domain.upper = {box.lx, box.ly, box.lz};
            domain.periodic = {true, true, true};
            decomp_runtime_->update_domain(domain);
            decomp_runtime_->set_halo_width(halo_width);
        }
        return;
    }
    if (!decomp_runtime_) {
        throw std::runtime_error("dsDecomp topology is not initialized");
    }

    dsdecomp::GlobalDomain domain;
    domain.lower = {0.0, 0.0, 0.0};
    domain.upper = {box.lx, box.ly, box.lz};
    domain.periodic = {true, true, true};
    decomp_runtime_->update_domain(domain);

    const auto& local = decomp_runtime_->topology().local_domain();
    decomp_runtime_->set_halo_width(halo_width);
    // Update box subdomain bounds used by neighbor list builder and integrator
    box.xlo = local.lower[0]; box.xhi = local.upper[0];
    box.ylo = local.lower[1]; box.yhi = local.upper[1];
    box.zlo = local.lower[2]; box.zhi = local.upper[2];
    // NOTE: redistribute_particles_after_scale() is NOT called here.
    // It is called explicitly by md_simulation.cpp after each NPT box change.
}

bool MDSystem::migrate_particles_check() {
    if (size == 1) return false;
    if (!decomp_runtime_) return false;

    const size_t local_N = get_local_particles();
    particles.reset_counts(local_N, ghost_count_);
    if (!decomp_runtime_->has_out_of_domain_particles(particles)) {
        return false;
    }

    redistribute_particles_after_scale();
    invalidate_halo_cache();  // Particles moved; rebuild halo buffers from scratch next time
    return true;
}

void MDSystem::redistribute_particles_after_scale() {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecomp is not initialized");
    particles.reset_counts(get_local_particles(), ghost_count_);
    const auto result = decomp_runtime_->migrate(particles);
    const size_t new_local = result.new_owned_count;
    local_start = 0;
    local_end   = static_cast<int>(new_local);
    num_local_particles_ = new_local;
    ghost_count_ = 0;

}

void MDSystem::migrate_particles() {
    // Migrate particles that have drifted outside this rank's subdomain.
    // Same redistribution logic as redistribute_particles_after_scale() but without
    // any box-size change.  Called every neighbor-list rebuild step for NVE/NVT.
    if (size == 1) return;
    redistribute_particles_after_scale();   // reuses the full particle-migration logic
}

void MDSystem::invalidate_halo_cache() {
    if (decomp_runtime_) decomp_runtime_->invalidate_halo();
}

void MDSystem::refresh_local_bonded_topology() {
    const size_t local_count = get_local_particles();
    if (bonded_layout_local_count_ == local_count &&
        bonded_layout_global_ids_ == particles.global_id) {
        return;
    }

    local_bonds_.clear();
    local_angles_.clear();
    local_dihedrals_.clear();
    if (!topology) return;

    std::unordered_map<int, size_t> slot;
    slot.reserve(particles.global_id.size() * 2);
    // Owned slots take precedence over periodic duplicate ghost images.
    for (size_t i = 0; i < local_count && i < particles.global_id.size(); ++i) {
        slot.emplace(particles.global_id[i], i);
    }
    for (size_t i = local_count; i < particles.global_id.size(); ++i) {
        slot.emplace(particles.global_id[i], i);
    }

    auto owned_slot = [&](int global_id) {
        auto it = slot.find(global_id);
        return it != slot.end() && it->second < local_count;
    };

    local_bonds_.reserve(topology->get_bonds().size() / static_cast<size_t>(std::max(size, 1)) + 1);
    for (const Bond& bond : topology->get_bonds()) {
        const int owner = std::min(bond.atom1, bond.atom2);
        if (!owned_slot(owner)) continue;
        auto i = slot.find(bond.atom1), j = slot.find(bond.atom2);
        if (i == slot.end() || j == slot.end()) continue;
        local_bonds_.push_back({i->second, j->second, bond.type});
    }

    local_angles_.reserve(topology->get_angles().size() / static_cast<size_t>(std::max(size, 1)) + 1);
    for (const Angle& angle : topology->get_angles()) {
        if (!owned_slot(angle.atom2)) continue;
        auto i = slot.find(angle.atom1), j = slot.find(angle.atom2), k = slot.find(angle.atom3);
        if (i == slot.end() || j == slot.end() || k == slot.end()) continue;
        local_angles_.push_back({i->second, j->second, k->second, angle.type});
    }

    local_dihedrals_.reserve(topology->get_dihedrals().size() / static_cast<size_t>(std::max(size, 1)) + 1);
    for (const Dihedral& dihedral : topology->get_dihedrals()) {
        const int owner = std::min(dihedral.atom2, dihedral.atom3);
        if (!owned_slot(owner)) continue;
        auto i = slot.find(dihedral.atom1), j = slot.find(dihedral.atom2);
        auto k = slot.find(dihedral.atom3), l = slot.find(dihedral.atom4);
        if (i == slot.end() || j == slot.end() || k == slot.end() || l == slot.end()) continue;
        local_dihedrals_.push_back(
            {i->second, j->second, k->second, l->second, dihedral.type});
    }

    bonded_layout_global_ids_ = particles.global_id;
    bonded_layout_local_count_ = local_count;
}

#ifdef USE_CUDA
bool MDSystem::exchange_halo_on_device(ForceCalculatorCUDA& calculator,
                                       bool rebuild_halo) {
    if (size == 1) return false;
    const size_t owned = get_local_particles();
    cudaStream_t stream = calculator.get_force_stream();
    dsdecomp::ParticleInfo storage(
        owned, ghost_count_, calculator);
    const auto result = decomp_runtime_->exchange_halo(
        storage, dsdecomp::HaloExchangeOptions{rebuild_halo, stream});
    resize_particle_storage(owned + result.ghost_count);
    ghost_count_ = result.ghost_count;
    return result.storage_reallocated;
}

bool MDSystem::migrate_particles_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1) return false;
    const size_t owned = get_local_particles();
    cudaStream_t stream = calculator.get_force_stream();
    dsdecomp::ParticleInfo storage(
        owned, ghost_count_, calculator);
    const auto result = decomp_runtime_->migrate(storage, stream);
    resize_particle_storage(result.new_owned_count);
    local_start = 0;
    local_end = static_cast<int>(result.new_owned_count);
    num_local_particles_ = result.new_owned_count;
    ghost_count_ = 0;
    return result.ownership_changed;
}

void MDSystem::reverse_halo_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecomp is not initialized");
    const size_t owned = get_local_particles();
    dsdecomp::ParticleInfo storage(
        owned, ghost_count_, calculator);
    decomp_runtime_->reverse_halo(storage, calculator.get_force_stream());
}
#endif

dsdecomp::HaloResult MDSystem::perform_halo_exchange(bool rebuild) {
    // A single MPI rank already owns the full periodic box.  Creating 26
    // self-images here would duplicate interactions because the serial
    // neighbor builder also applies the minimum-image convention.
    if (size == 1) {
        return {};
    }
    if (!decomp_runtime_) throw std::runtime_error("dsDecomp is not initialized");
    particles.reset_counts(get_local_particles(), ghost_count_);
    const auto result = decomp_runtime_->exchange_halo(
        particles, dsdecomp::HaloExchangeOptions{rebuild, nullptr});
    ghost_count_ = result.ghost_count;
    particles.num_particles = get_local_particles() + ghost_count_;
    return result;
}

bool MDSystem::sync_halo_particles(bool rebuild_halo, NeighborList* nlist) {
    if (size == 1) return false;

    (void)nlist;
    const auto result = perform_halo_exchange(rebuild_halo);
    refresh_local_bonded_topology();
    return result.layout_changed;
}

bool MDSystem::sync_distributed_particles(bool local_rebuild_hint,
                                          NeighborList* nlist) {
    if (size == 1) {
        return local_rebuild_hint;
    }

    bool should_rebuild = unify_rebuild_decision(local_rebuild_hint);
    if (should_rebuild) {
        const bool migrated = migrate_particles_check();
        if (migrated) {
            should_rebuild = true;
        }
    }

    const bool layout_changed = sync_halo_particles(should_rebuild, nlist);
    if (layout_changed) should_rebuild = true;

    return should_rebuild;
}

bool MDSystem::unify_rebuild_decision(bool local_rebuild_flag) const {
    if (size == 1) return local_rebuild_flag;
    if (!decomp_runtime_) throw std::runtime_error("dsDecomp is not initialized");
    return decomp_runtime_->any_rank_true(local_rebuild_flag);
}

void MDSystem::perform_reverse_halo_exchange() {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecomp is not initialized");
    particles.reset_counts(get_local_particles(), ghost_count_);
    decomp_runtime_->reverse_halo(particles);
}

