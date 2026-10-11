
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
#include <cmath>
#include <cstdlib>
#include <atomic>
#include <unistd.h>  // For getcwd
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Reported by dsDecompX libraries that have it (see MDSystem::init, header/library check).
namespace dsdecompx { std::size_t halo_exchange_options_size() noexcept __attribute__((weak)); }

namespace {

// Fills `result` with what body(0, count, result, true) would produce, using several OpenMP
// threads for a long range. Each thread handles one contiguous block of [0, count) into its own
// vector and the blocks are appended in thread order, so the result is identical to the
// sequential pass. body(begin, end, out, strict) returns false when it met a term it cannot
// handle without throwing (strict == false); the pass is then repeated sequentially with
// strict == true so that the original exception surfaces on the calling thread.
template <typename T, typename Body>
void collect_in_order(size_t count, std::vector<T>& result, Body&& body) {
#ifdef _OPENMP
    const size_t blocks = std::min<size_t>({static_cast<size_t>(std::max(1, omp_get_max_threads())),
                                             size_t(16), count / 8192 + 1});
    if (blocks > 1) {
        std::vector<std::vector<T>> parts(blocks);
        std::atomic<bool> failed{false};
#pragma omp parallel num_threads(static_cast<int>(blocks))
        {
            const size_t team = static_cast<size_t>(omp_get_num_threads());
            const size_t id = static_cast<size_t>(omp_get_thread_num());
            if (!body(count * id / team, count * (id + 1) / team, parts[id], false)) failed.store(true);
        }
        if (!failed.load()) {
            size_t total = 0;
            for (const auto& part : parts) total += part.size();
            result.reserve(result.size() + total);
            for (auto& part : parts) result.insert(result.end(), part.begin(), part.end());
            return;
        }
        result.clear();
    }
#endif
    body(size_t(0), count, result, true);
}

}  // namespace

void MDSystem::resize_particle_storage(size_t count) {
    // Bypass ParticleData::resize(), which assigns sequential global IDs and
    // would corrupt IDs when storage grows or shrinks after halo/migration.
    static_cast<dsdecompx::ParticleInfo&>(particles).resize(count);
}

MDSystem::MDSystem() : physical_cutoff(12.0), rank(0), size(1) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    topology = std::make_unique<Topology>();
    force_manager = std::make_unique<ForceManager>();
}

MDSystem::~MDSystem() = default;

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

void MDSystem::initialize_distributed_velocities(Real temperature) {
    const size_t local_count = get_local_particles();
    std::random_device rd;
    std::seed_seq seed{rd(), rd(), static_cast<unsigned int>(rank)};
    std::mt19937 gen(seed);
    std::normal_distribution<Real> normal(0.0, 1.0);

    for (size_t i = 0; i < local_count; ++i) {
        particles.vx[i] = normal(gen);
        particles.vy[i] = normal(gen);
        particles.vz[i] = normal(gen);
    }

    Real local_moments[4] = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < local_count; ++i) {
        const Real mass = particles.mass[i];
        local_moments[0] += mass * particles.vx[i];
        local_moments[1] += mass * particles.vy[i];
        local_moments[2] += mass * particles.vz[i];
        local_moments[3] += mass;
    }
    Real global_moments[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_moments, global_moments, 4, MPI_DOUBLE, MPI_SUM, pp_comm_);
    if (global_moments[3] <= 0.0) {
        throw std::runtime_error("Cannot initialize velocities: total mass is not positive");
    }

    const Real cm_vx = global_moments[0] / global_moments[3];
    const Real cm_vy = global_moments[1] / global_moments[3];
    const Real cm_vz = global_moments[2] / global_moments[3];
    Real local_ke_unscaled = 0.0;
    for (size_t i = 0; i < local_count; ++i) {
        particles.vx[i] -= cm_vx;
        particles.vy[i] -= cm_vy;
        particles.vz[i] -= cm_vz;
        const Real v2 = particles.vx[i] * particles.vx[i] +
                        particles.vy[i] * particles.vy[i] +
                        particles.vz[i] * particles.vz[i];
        local_ke_unscaled += 0.5 * particles.mass[i] * v2;
    }

    Real global_ke_unscaled = 0.0;
    MPI_Allreduce(&local_ke_unscaled, &global_ke_unscaled, 1,
                  MPI_DOUBLE, MPI_SUM, pp_comm_);
    unsigned long long local_atoms = static_cast<unsigned long long>(local_count);
    unsigned long long global_atoms = 0;
    MPI_Allreduce(&local_atoms, &global_atoms, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, pp_comm_);

    const Real dof = global_atoms > 1 ? static_cast<Real>(3 * global_atoms - 3) : 0.0;
    const Real target_ke = 0.5 * dof * kB * temperature;
    const Real current_ke = global_ke_unscaled * mvv2e;
    const Real scale = (target_ke > 0.0 && current_ke > 0.0)
        ? std::sqrt(target_ke / current_ke) : 0.0;
    for (size_t i = 0; i < local_count; ++i) {
        particles.vx[i] *= scale;
        particles.vy[i] *= scale;
        particles.vz[i] *= scale;
    }
    for (size_t i = local_count; i < particles.vx.size(); ++i) {
        particles.vx[i] = particles.vy[i] = particles.vz[i] = 0.0;
    }

    if (rank == 0) {
        std::cout << "Reinitialized post-minimization velocities at T = "
                  << temperature << " K" << std::endl;
    }
}

void MDSystem::load_topology(const std::string& topo_file,
                             const std::string& parameter_file) {
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
        load_ok = parameter_file.empty()
            ? (topology->load_topology(clean_filename, rank) ? 1 : 0)
            : (topology->load_canonical(clean_filename, parameter_file, rank) ? 1 : 0);
    }
    MPI_Bcast(&load_ok, 1, MPI_INT, 0, pp_comm_);
    if (!load_ok) {
        throw std::runtime_error("Failed to load topology file: " + clean_filename);
    }

    topology->broadcast(pp_comm_, 0);

    int atom_count_matches = 1;
    if (rank == 0) {
        atom_count_matches = topology->get_num_atoms() == static_cast<int>(particles.num_particles) ? 1 : 0;
    }
    MPI_Bcast(&atom_count_matches, 1, MPI_INT, 0, pp_comm_);
    if (!atom_count_matches) {
        throw std::runtime_error("Coordinate/topology atom-count mismatch: XYZ has " +
            std::to_string(particles.num_particles) + ", topology has " +
            std::to_string(topology->get_num_atoms()));
    }

    int element_mismatch = -1;
    if (rank == 0 && topology->is_canonical()) {
        for (size_t i = 0; i < coordinate_elements_.size(); ++i) {
            const std::string& expected = topology->get_atom_element(static_cast<int>(i));
            if (!expected.empty() && coordinate_elements_[i] != expected) {
                element_mismatch = static_cast<int>(i);
                std::cerr << "Coordinate/topology element mismatch at atom " << i
                          << ": XYZ has '" << coordinate_elements_[i]
                          << "', topology has '" << expected << "'" << std::endl;
                break;
            }
        }
    }
    MPI_Bcast(&element_mismatch, 1, MPI_INT, 0, pp_comm_);
    if (element_mismatch >= 0) {
        throw std::runtime_error("Coordinate/topology element-order mismatch at atom " +
                                 std::to_string(element_mismatch));
    }

    // Only root performs the O(Nglobal) atom-property lookup.  The resulting
    // per-particle fields are distributed before dsDecompX compacts each rank.
    if (rank == 0) {
        assign_masses_from_topology();
        assign_charges_from_topology();
    }
}

void MDSystem::setup_domain_decomposition() {
    if (size == 1) {
        particles.reset_counts(particles.num_particles, 0);
        box.xlo = 0.0; box.xhi = box.lx;
        box.ylo = 0.0; box.yhi = box.ly;
        box.zlo = 0.0; box.zhi = box.lz;
        refresh_local_bonded_topology();
        return;
    }

    // dsDecompX owns the Cartesian topology and local-domain geometry.
    initialize_decomposition_runtime();
    // Root alone owns the global input arrays.  It classifies particles with
    // dsDecompX, groups them by destination rank, and scatters only the owned
    // slices.  Non-root ranks never allocate the global particle set.
    std::uint64_t total_particles_u64 =
        (rank == 0) ? static_cast<std::uint64_t>(particles.num_particles) : 0;
    MPI_Bcast(&total_particles_u64, 1, MPI_UINT64_T, 0, pp_comm_);
    const size_t total_particles = static_cast<size_t>(total_particles_u64);

    std::vector<int> counts(size, 0), displacements(size, 0);
    if (rank == 0) {
        std::vector<int> owners(total_particles);
        std::vector<size_t> sorted_indices(total_particles);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        for (size_t i = 0; i < total_particles; ++i) {
            owners[i] = decomp_runtime_->owner_rank(
                dsdecompx::Vec3d{particles.x[i], particles.y[i], particles.z[i]});
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
                &local_count, 1, MPI_INT, 0, pp_comm_);

    auto scatter_real = [&](std::vector<Real>& values) {
        std::vector<Real> local(static_cast<size_t>(local_count));
        MPI_Scatterv(rank == 0 ? values.data() : nullptr,
                     rank == 0 ? counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_DOUBLE, local.data(), local_count, MPI_DOUBLE,
                     0, pp_comm_);
        values.swap(local);
    };
    auto scatter_int = [&](std::vector<int>& values) {
        std::vector<int> local(static_cast<size_t>(local_count));
        MPI_Scatterv(rank == 0 ? values.data() : nullptr,
                     rank == 0 ? counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_INT, local.data(), local_count, MPI_INT,
                     0, pp_comm_);
        values.swap(local);
    };
    scatter_real(particles.x); scatter_real(particles.y); scatter_real(particles.z);
    scatter_real(particles.vx); scatter_real(particles.vy); scatter_real(particles.vz);
    scatter_real(particles.fx); scatter_real(particles.fy); scatter_real(particles.fz);
    scatter_real(particles.mass); scatter_real(particles.charge);
    scatter_int(particles.type); scatter_int(particles.global_id);

    const size_t owned_count = static_cast<size_t>(local_count);
    particles.num_particles = owned_count;
    particles.reset_counts(owned_count, 0);
    refresh_local_bonded_topology();

    if (rank == 0) {
        const auto& dimensions = decomp_runtime_->topology().dimensions();
        std::cout << "Spatial domain decomposition: " << dimensions[0] << "x" << dimensions[1]
                  << "x" << dimensions[2] << " grid, " << total_particles << " particles" << std::endl;
        std::cout << "  Rank 0: " << owned_count << " local particles" << std::endl;
    }
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
    coordinate_elements_.assign(static_cast<size_t>(num_atoms), "");
    for (int i = 0; i < num_atoms; ++i) particles.global_id[i] = i;
    
    // Read atom coordinates
    int atom_count = 0;
    while (std::getline(file, line) && atom_count < num_atoms) {
        std::istringstream iss(line);
        std::string element;
        Real x, y, z;
        
        if (iss >> element >> x >> y >> z) {
            coordinate_elements_[static_cast<size_t>(atom_count)] = element;
            particles.x[atom_count] = x;
            particles.y[atom_count] = y;
            particles.z[atom_count] = z;
            atom_count++;
        }
    }

    if (atom_count != num_atoms) {
        throw std::runtime_error("XYZ atom-count mismatch: header declares " +
            std::to_string(num_atoms) + " but " + std::to_string(atom_count) +
            " coordinate records were read");
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
            particles.type[i] = topology->get_atom_type_id(i);
            particles.mass[i] = topology->get_atom_mass(i);
        } catch (const std::runtime_error& e) {
            if (rank == 0) {
                std::cerr << "Warning: Cannot get mass for atom " << i << ": " << e.what() << std::endl;
                std::cerr << "Using default mass 1.0" << std::endl;
            }
            particles.mass[i] = 1.0;  // Default mass
        }
    }
    
    if (rank == 0) std::cout << "Assigned integer atom type IDs from topology" << std::endl;
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
    ff_params.mie_repulsive_exponent = topology->get_mie_repulsive_exponent();
    ff_params.mie_attractive_exponent = topology->get_mie_attractive_exponent();
    const Real n = static_cast<Real>(ff_params.mie_repulsive_exponent);
    const Real m = static_cast<Real>(ff_params.mie_attractive_exponent);
    ff_params.mie_prefactor = (n / (n - m)) * std::pow(n / m, m / (n - m));
    
    if (rank == 0) {
        if (topology->has_explicit_pair_parameters()) {
            ff_params.num_atom_types = static_cast<int>(topology->get_pair_sigma().size());
            ff_params.sigma_mixed = topology->get_pair_sigma();
            ff_params.epsilon_mixed = topology->get_pair_epsilon();
            ff_params.sigma.resize(static_cast<size_t>(ff_params.num_atom_types));
            ff_params.epsilon.resize(static_cast<size_t>(ff_params.num_atom_types));
            for (int i = 0; i < ff_params.num_atom_types; ++i) {
                ff_params.sigma[i] = ff_params.sigma_mixed[i][i];
                ff_params.epsilon[i] = ff_params.epsilon_mixed[i][i];
            }
            ff_params.explicit_pair_parameters = true;
        } else {
            std::map<int, std::pair<Real, Real>> type_params;
            for (size_t i = 0; i < particles.num_particles; i++) {
                const int type = particles.type[i];
                if (type_params.find(type) == type_params.end()) {
                    type_params[type] = {topology->get_atom_epsilon(static_cast<int>(i)),
                                         topology->get_atom_sigma(static_cast<int>(i))};
                }
            }
            ff_params.num_atom_types = static_cast<int>(type_params.size());
            ff_params.epsilon.resize(static_cast<size_t>(ff_params.num_atom_types));
            ff_params.sigma.resize(static_cast<size_t>(ff_params.num_atom_types));
            for (const auto& pair : type_params) {
                if (pair.first < 0 || pair.first >= ff_params.num_atom_types) {
                    throw std::runtime_error("Legacy atom type IDs are not contiguous");
                }
                ff_params.epsilon[pair.first] = pair.second.first;
                ff_params.sigma[pair.first] = pair.second.second;
            }
            ff_params.explicit_pair_parameters = false;
        }
    }

    MPI_Bcast(&ff_params.num_atom_types, 1, MPI_INT, 0, pp_comm_);
    if (rank != 0) {
        ff_params.epsilon.resize(static_cast<size_t>(ff_params.num_atom_types));
        ff_params.sigma.resize(static_cast<size_t>(ff_params.num_atom_types));
    }
    MPI_Bcast(ff_params.epsilon.data(), ff_params.num_atom_types, MPI_DOUBLE, 0, pp_comm_);
    MPI_Bcast(ff_params.sigma.data(), ff_params.num_atom_types, MPI_DOUBLE, 0, pp_comm_);

    int explicit_pairs = (rank == 0 && ff_params.explicit_pair_parameters) ? 1 : 0;
    MPI_Bcast(&explicit_pairs, 1, MPI_INT, 0, pp_comm_);
    ff_params.explicit_pair_parameters = explicit_pairs != 0;
    if (ff_params.explicit_pair_parameters) {
        if (rank != 0) {
            ff_params.sigma_mixed.assign(ff_params.num_atom_types,
                std::vector<Real>(ff_params.num_atom_types));
            ff_params.epsilon_mixed.assign(ff_params.num_atom_types,
                std::vector<Real>(ff_params.num_atom_types));
        }
        for (int i = 0; i < ff_params.num_atom_types; ++i) {
            MPI_Bcast(ff_params.sigma_mixed[i].data(), ff_params.num_atom_types,
                      MPI_DOUBLE, 0, pp_comm_);
            MPI_Bcast(ff_params.epsilon_mixed[i].data(), ff_params.num_atom_types,
                      MPI_DOUBLE, 0, pp_comm_);
        }
    } else {
        ff_params.initialize_mixing_tables();
    }
    
    if (rank == 0) {
        std::cout << "Force field setup: " << particles.num_particles << " particles, " 
                  << ff_params.num_atom_types << " atom types" << std::endl;
        std::cout << "Nonbonded Mie potential: (" << ff_params.mie_repulsive_exponent
                  << "," << ff_params.mie_attractive_exponent << "), prefactor="
                  << ff_params.mie_prefactor << std::endl;
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
    dsdecompx::Config config;
    config.domain.lower = {0.0, 0.0, 0.0};
    config.domain.upper = {box.lx, box.ly, box.lz};
    // Preserve the existing algorithm: the MPI Cartesian topology has always
    // been periodic in all three dimensions, independently of force PBC flags.
    config.domain.periodic = {true, true, true};
    config.process_grid.dimensions = {0, 0, 0};
    config.process_grid.allow_rank_reorder = false;
    // Keep the established MPI_Dims_create behavior by default. The planner is
    // opt-in so upgrading the library cannot silently change rank ownership.
    if (const char* policy = std::getenv("GRID_POLICY")) {
        const std::string selected(policy);
        if (selected == "balanced") {
            config.grid_policy = dsdecompx::GridPolicy::Balanced;
        } else if (selected == "topology" || selected == "topology-aware") {
            config.grid_policy = dsdecompx::GridPolicy::TopologyAware;
            if (const char* weight = std::getenv("INTER_NODE_WEIGHT")) {
                const double parsed = std::atof(weight);
                if (parsed > 0.0) config.topology_info.inter_node_weight = parsed;
            }
            if (const char* weight = std::getenv("INTRA_NODE_WEIGHT")) {
                const double parsed = std::atof(weight);
                if (parsed > 0.0) config.topology_info.intra_node_weight = parsed;
            }
        } else if (selected == "dynamic") {
            config.grid_policy = dsdecompx::GridPolicy::Dynamic;
            if (const char* value = std::getenv("DYNAMIC_BALANCE_INTERVAL")) {
                const long parsed = std::atol(value);
                if (parsed >= 0) config.dynamic.interval = static_cast<size_t>(parsed);
            }
            if (const char* value = std::getenv("DYNAMIC_IMBALANCE_THRESHOLD")) {
                const double parsed = std::atof(value);
                if (parsed >= 1.0) config.dynamic.imbalance_threshold = parsed;
            }
            if (const char* value = std::getenv("DYNAMIC_RELAXATION")) {
                const double parsed = std::atof(value);
                if (parsed > 0.0 && parsed <= 1.0) config.dynamic.relaxation = parsed;
            }
        }
    }
    config.halo.width = physical_cutoff + verlet_skin_ + 1.0;
    config.halo.include_edges_and_corners = true;
#ifdef USE_CUDA
    config.execution = dsdecompx::ExecutionMode::Cuda;
    // GPU-resident mode is mandatory for a CUDA DeepScaleMD build. A backend
    // that cannot consume device buffers must fail instead of staging them.
    config.advanced.allow_host_staging = false;
    // Decomp is created before ForceCalculatorCUDA::initialize() picks this
    // rank's GPU (node-local rank modulo visible devices), and NCCL/NVSHMEM bind
    // to the current device while initializing. Selecting the same device here
    // keeps backend, buffers and force kernels on one GPU per rank even when
    // every rank sees every GPU of the node.
    config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
#else
    config.execution = dsdecompx::ExecutionMode::Host;
#endif

    // HaloExchangeOptions crosses the library boundary by value. If the headers this file was compiled
    // with and the dsDecompX library come from different builds, the library reads garbage (for
    // example a null communication stream), so compare the layouts before using it.
    {
        const bool library_reports_size = &dsdecompx::halo_exchange_options_size != nullptr;
#ifdef DSDECOMPX_HALO_POSITIONS_ONLY
        const bool consistent = library_reports_size &&
            dsdecompx::halo_exchange_options_size() == sizeof(dsdecompx::HaloExchangeOptions);
#else
        const bool consistent = !library_reports_size ||
            dsdecompx::halo_exchange_options_size() == sizeof(dsdecompx::HaloExchangeOptions);
#endif
        if (!consistent) {
            std::string detail;
#ifdef DSDECOMPX_HALO_POSITIONS_ONLY
            detail = std::string("DeepScaleMD was compiled with ") + dsdecompx::header_path() +
                     " (HaloExchangeOptions " + std::to_string(sizeof(dsdecompx::HaloExchangeOptions)) +
                     " bytes)";
#else
            detail = "DeepScaleMD was compiled with an older dsdecompx.hpp (HaloExchangeOptions " +
                     std::to_string(sizeof(dsdecompx::HaloExchangeOptions)) + " bytes, no positions_only)";
#endif
            if (library_reports_size) {
                detail += "; the library reports " +
                          std::to_string(dsdecompx::halo_exchange_options_size()) + " bytes";
            } else {
                detail += "; the library has no size report (an older build)";
            }
            throw std::runtime_error(
                "dsDecompX headers and library do not match (different builds): " + detail +
                ". Another dsdecompx.hpp earlier on the include path (CPATH, ~/.local/include) or an old "
                "libdsdecompx.a is the usual cause; rebuild dsDecompX (bash build_cuda.sh) and DeepScaleMD.");
        }
    }
    decomp_runtime_ = std::make_unique<dsdecompx::Decomp>(pp_comm_, config);

    const auto& topology = decomp_runtime_->topology();
    if (rank==0) {
        const auto& dimensions = topology.dimensions();
        std::cout << "[Halo] Cartesian topology " << dimensions[0] << "x"
                  << dimensions[1] << "x" << dimensions[2]
                  << ", neighbors=" << topology.neighbors().size() << std::endl;
    }
    sync_decomposition_domain();
    if (rank==0) {
        std::cout << "[Halo] sizeof(ParticleRecord)=" << sizeof(dsdecompx::ParticleRecord)
                  << " bytes, halo_width=" << decomp_runtime_->halo_width() << std::endl;
        const char* backend_name = "mpi";
        switch (decomp_runtime_->backend_kind()) {
            case dsdecompx::BackendKind::Nccl: backend_name = "nccl"; break;
            case dsdecompx::BackendKind::Nvshmem: backend_name = "nvshmem"; break;
            default: break;
        }
        std::cout << "[Halo] data-plane backend: " << backend_name << std::endl;
    }
}

// With exactly two ranks along a periodic axis the left and right neighbor are
// the same rank. When the domain is narrower than two halo widths, a particle
// near the middle of the neighbor's domain lies inside both halo slabs and
// arrives twice, as two periodic images of one atom. The pair kernels apply the
// minimum image to every pair, so both copies interact with the same owned atom
// and the short-range energy and forces near the domain faces are counted twice.
void MDSystem::warn_if_domain_narrower_than_two_halos(Real halo_width) {
    static bool warned = false;
    if (warned || rank != 0 || !decomp_runtime_) return;
    const auto& dimensions = decomp_runtime_->topology().dimensions();
    const auto& local = decomp_runtime_->topology().local_domain();
    const char axis_name[3] = {'x', 'y', 'z'};
    for (int axis = 0; axis < 3; ++axis) {
        const Real width = local.upper[axis] - local.lower[axis];
        if (dimensions[axis] == 2 && width < 2.0 * halo_width) {
            warned = true;
            std::cerr << "[WARNING] Domain decomposition has 2 ranks along " << axis_name[axis]
                      << " with a domain width of " << width << " A, less than twice the halo width ("
                      << 2.0 * halo_width << " A). Atoms near the middle of a domain are received "
                      << "as ghosts twice, so short-range energies and forces near the domain "
                      << "faces are counted twice (PE and pressure are wrong). Use a larger box "
                      << "or a different number of ranks." << std::endl;
        }
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
            dsdecompx::GlobalDomain domain;
            domain.lower = {0.0, 0.0, 0.0};
            domain.upper = {box.lx, box.ly, box.lz};
            domain.periodic = {true, true, true};
            decomp_runtime_->update_domain(domain);
            decomp_runtime_->set_halo_width(halo_width);
        }
        return;
    }
    if (!decomp_runtime_) {
        throw std::runtime_error("dsDecompX topology is not initialized");
    }

    dsdecompx::GlobalDomain domain;
    domain.lower = {0.0, 0.0, 0.0};
    domain.upper = {box.lx, box.ly, box.lz};
    domain.periodic = {true, true, true};
    decomp_runtime_->update_domain(domain);

    const auto& local = decomp_runtime_->topology().local_domain();
    decomp_runtime_->set_halo_width(halo_width);
    warn_if_domain_narrower_than_two_halos(halo_width);
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

    const auto result = decomp_runtime_->migrate(particles);

    resize_particle_storage(result.new_owned_count);
    particles.reset_counts(result.new_owned_count, 0);

    const auto& local = decomp_runtime_->topology().local_domain();
    box.xlo = local.lower[0]; box.xhi = local.upper[0];
    box.ylo = local.lower[1]; box.yhi = local.upper[1];
    box.zlo = local.lower[2]; box.zhi = local.upper[2];

    if (result.ownership_changed) {
        // Particle ownership changed; halo selection must be rebuilt from
        // scratch before the next exchange.
        invalidate_halo_cache();
    }
    return result.ownership_changed;
}

void MDSystem::redistribute_particles_after_scale() {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecompX is not initialized");
    const auto result = decomp_runtime_->migrate(particles);
    // BUG FIX: see the identical fix/comment in migrate_particles_check().
    resize_particle_storage(result.new_owned_count);
    particles.reset_counts(result.new_owned_count, 0);

    const auto& local = decomp_runtime_->topology().local_domain();
    box.xlo = local.lower[0]; box.xhi = local.upper[0];
    box.ylo = local.lower[1]; box.yhi = local.upper[1];
    box.zlo = local.lower[2]; box.zhi = local.upper[2];
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

void MDSystem::ensure_bonded_parameter_cache() {
    if (bonded_param_cache_ready_ || !topology) return;
    auto fill = [](const auto& terms, const auto& lookup, auto&& store, std::vector<char>& ok) {
        ok.assign(terms.size(), 0);
        for (size_t t = 0; t < terms.size(); ++t) {
            try {
                store(t, lookup(terms[t].type));
                ok[t] = 1;
            } catch (const std::exception&) {
                // Unknown type: only an error if this term ever becomes local, where the
                // original lookup is repeated so the same exception is thrown.
            }
        }
    };
    const auto& bonds = topology->get_bonds();
    bond_k_all_.assign(bonds.size(), 0.0); bond_r0_all_.assign(bonds.size(), 0.0);
    fill(bonds, [&](const std::string& n) -> const BondType& { return topology->get_bond_type(n); },
         [&](size_t t, const BondType& p) { bond_k_all_[t] = p.k; bond_r0_all_[t] = p.r0; }, bond_ok_all_);
    const auto& angles = topology->get_angles();
    angle_k_all_.assign(angles.size(), 0.0); angle_theta0_all_.assign(angles.size(), 0.0);
    fill(angles, [&](const std::string& n) -> const AngleType& { return topology->get_angle_type(n); },
         [&](size_t t, const AngleType& p) { angle_k_all_[t] = p.k; angle_theta0_all_[t] = p.theta0; }, angle_ok_all_);
    const auto& dihedrals = topology->get_dihedrals();
    dihedral_k_all_.assign(dihedrals.size(), 0.0); dihedral_phi0_all_.assign(dihedrals.size(), 0.0);
    dihedral_n_all_.assign(dihedrals.size(), 0);
    fill(dihedrals, [&](const std::string& n) -> const DihedralType& { return topology->get_dihedral_type(n); },
         [&](size_t t, const DihedralType& p) {
             dihedral_k_all_[t] = p.k; dihedral_phi0_all_[t] = p.phi0; dihedral_n_all_[t] = p.n; },
         dihedral_ok_all_);
    const auto& impropers = topology->get_impropers();
    improper_k_all_.assign(impropers.size(), 0.0); improper_phi0_all_.assign(impropers.size(), 0.0);
    improper_n_all_.assign(impropers.size(), 0);
    fill(impropers, [&](const std::string& n) -> const DihedralType& { return topology->get_dihedral_type(n); },
         [&](size_t t, const DihedralType& p) {
             improper_k_all_[t] = p.k; improper_phi0_all_[t] = p.phi0; improper_n_all_[t] = p.n; },
         improper_ok_all_);
    bonded_param_cache_ready_ = true;
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
    local_impropers_.clear();
    local_special_pairs_.clear();
    local_constraints_.clear();
    if (!topology) return;
    ensure_bonded_parameter_cache();

    // Global id -> local slot in a flat array (it used to be an unordered_map, with one
    // hash lookup per topology term on every neighbor rebuild). Owned slots take
    // precedence over periodic duplicate ghost images, and the first ghost wins.
    int max_gid = -1;
    for (const int gid : particles.global_id) max_gid = std::max(max_gid, gid);
    std::vector<int>& slot = bonded_slot_scratch_;
    slot.assign(static_cast<size_t>(max_gid + 1), -1);
    for (size_t i = 0; i < local_count && i < particles.global_id.size(); ++i) {
        const int gid = particles.global_id[i];
        if (gid >= 0 && slot[gid] < 0) slot[gid] = static_cast<int>(i);
    }
    for (size_t i = local_count; i < particles.global_id.size(); ++i) {
        const int gid = particles.global_id[i];
        if (gid >= 0 && slot[gid] < 0) slot[gid] = static_cast<int>(i);
    }
    // Atoms that live on other ranks have ids above or between the local ones: absent.
    auto slot_of = [&](int gid) -> int {
        return (gid >= 0 && static_cast<size_t>(gid) < slot.size()) ? slot[gid] : -1;
    };
    auto owned_slot = [&](int gid) {
        const int s = slot_of(gid);
        return s >= 0 && static_cast<size_t>(s) < local_count;
    };
    const bool redundant = bonded_redundant_;

    const size_t ranks = static_cast<size_t>(std::max(size, 1));
    const auto& bonds = topology->get_bonds();
    local_bonds_.reserve(bonds.size() / ranks + 1);
    collect_in_order(bonds.size(), local_bonds_,
                     [&](size_t begin, size_t end, std::vector<LocalBond>& out, bool strict) {
        for (size_t t = begin; t < end; ++t) {
            const Bond& bond = bonds[t];
            const int owner = std::min(bond.atom1, bond.atom2);
            if (!(redundant ? (owned_slot(bond.atom1) || owned_slot(bond.atom2)) : owned_slot(owner))) continue;
            const int i = slot_of(bond.atom1), j = slot_of(bond.atom2);
            if (i < 0 || j < 0) continue;
            if (!bond_ok_all_[t]) {
                if (!strict) return false;
                topology->get_bond_type(bond.type);  // throws the original error
            }
            out.push_back({static_cast<size_t>(i), static_cast<size_t>(j), &bond.type,
                           bond_k_all_[t], bond_r0_all_[t]});
        }
        return true;
    });

    const auto& angles = topology->get_angles();
    local_angles_.reserve(angles.size() / ranks + 1);
    collect_in_order(angles.size(), local_angles_,
                     [&](size_t begin, size_t end, std::vector<LocalAngle>& out, bool strict) {
        for (size_t t = begin; t < end; ++t) {
            const Angle& angle = angles[t];
            if (!(redundant ? (owned_slot(angle.atom1) || owned_slot(angle.atom2) || owned_slot(angle.atom3))
                            : owned_slot(angle.atom2))) continue;
            const int i = slot_of(angle.atom1), j = slot_of(angle.atom2), k = slot_of(angle.atom3);
            if (i < 0 || j < 0 || k < 0) continue;
            if (!angle_ok_all_[t]) {
                if (!strict) return false;
                topology->get_angle_type(angle.type);
            }
            out.push_back({static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k),
                           &angle.type, angle_k_all_[t], angle_theta0_all_[t]});
        }
        return true;
    });

    const auto& dihedrals = topology->get_dihedrals();
    local_dihedrals_.reserve(dihedrals.size() / ranks + 1);
    collect_in_order(dihedrals.size(), local_dihedrals_,
                     [&](size_t begin, size_t end, std::vector<LocalDihedral>& out, bool strict) {
        for (size_t t = begin; t < end; ++t) {
            const Dihedral& dihedral = dihedrals[t];
            const int owner = std::min(dihedral.atom2, dihedral.atom3);
            if (!(redundant ? (owned_slot(dihedral.atom1) || owned_slot(dihedral.atom2) ||
                               owned_slot(dihedral.atom3) || owned_slot(dihedral.atom4))
                            : owned_slot(owner))) continue;
            const int i = slot_of(dihedral.atom1), j = slot_of(dihedral.atom2);
            const int k = slot_of(dihedral.atom3), l = slot_of(dihedral.atom4);
            if (i < 0 || j < 0 || k < 0 || l < 0) continue;
            if (!dihedral_ok_all_[t]) {
                if (!strict) return false;
                topology->get_dihedral_type(dihedral.type);
            }
            out.push_back({static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k),
                           static_cast<size_t>(l), &dihedral.type,
                           dihedral_k_all_[t], dihedral_phi0_all_[t], dihedral_n_all_[t]});
        }
        return true;
    });

    const auto& impropers = topology->get_impropers();
    local_impropers_.reserve(impropers.size() / ranks + 1);
    collect_in_order(impropers.size(), local_impropers_,
                     [&](size_t begin, size_t end, std::vector<LocalDihedral>& out, bool strict) {
        for (size_t t = begin; t < end; ++t) {
            const Dihedral& improper = impropers[t];
            const int owner = std::min(improper.atom2, improper.atom3);
            if (!(redundant ? (owned_slot(improper.atom1) || owned_slot(improper.atom2) ||
                               owned_slot(improper.atom3) || owned_slot(improper.atom4))
                            : owned_slot(owner))) continue;
            const int i = slot_of(improper.atom1), j = slot_of(improper.atom2);
            const int k = slot_of(improper.atom3), l = slot_of(improper.atom4);
            if (i < 0 || j < 0 || k < 0 || l < 0) continue;
            if (!improper_ok_all_[t]) {
                if (!strict) return false;
                topology->get_dihedral_type(improper.type);
            }
            out.push_back({static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k),
                           static_cast<size_t>(l), &improper.type,
                           improper_k_all_[t], improper_phi0_all_[t], improper_n_all_[t]});
        }
        return true;
    });

    const auto& special_pairs = topology->get_special_pairs();
    local_special_pairs_.reserve(special_pairs.size() / ranks + 1);
    for (const SpecialPair& pair : special_pairs) {
        const int owner = std::min(pair.atom1, pair.atom2);
        if (!(redundant ? (owned_slot(pair.atom1) || owned_slot(pair.atom2)) : owned_slot(owner))) continue;
        const int i = slot_of(pair.atom1), j = slot_of(pair.atom2);
        if (i < 0 || j < 0) continue;
        local_special_pairs_.push_back(
            {static_cast<size_t>(i), static_cast<size_t>(j), pair.lj_scale, pair.coulomb_scale,
             pair.sigma, pair.epsilon});
    }

    const auto& constraints = topology->get_constraints();
    local_constraints_.reserve(constraints.size() / ranks + 1);
    for (const Constraint& c : constraints) {
        const int owner = std::min(c.atom1, c.atom2);
        if (!owned_slot(owner)) continue;
        const int i = slot_of(c.atom1), j = slot_of(c.atom2);
        if (i < 0 || j < 0) continue;
        local_constraints_.push_back({static_cast<size_t>(i), static_cast<size_t>(j), c.distance});
    }

    bonded_layout_global_ids_ = particles.global_id;
    bonded_layout_local_count_ = local_count;
}

#ifdef USE_CUDA
// A halo refresh on a reused plan only needs the new ghost positions: velocity, mass, charge, type
// and id are fixed between rebuilds, and nothing on the CUDA path reads a ghost's velocity. Sending
// the position triple instead of the full particle record cuts the halo traffic of a step to a
// quarter (96 -> 24 bytes per ghost), which matters once the ranks sit on different nodes.
// With a dsDecompX that predates the option (no DSDECOMPX_HALO_POSITIONS_ONLY) full records are sent.
static dsdecompx::HaloExchangeOptions make_halo_options(bool rebuild, cudaStream_t stream,
                                                        bool positions_only) {
    dsdecompx::HaloExchangeOptions options;
    options.rebuild = rebuild;
    options.stream = stream;
#ifdef DSDECOMPX_HALO_POSITIONS_ONLY
    options.positions_only = positions_only;
#else
    (void)positions_only;
#endif
    return options;
}

bool MDSystem::exchange_halo_on_device(ForceCalculatorCUDA& calculator,
                                       bool rebuild_halo) {
    if (size == 1) return false;
    const size_t owned = get_local_particles();
    cudaStream_t stream = calculator.get_force_stream();
    dsdecompx::ParticleInfo storage(
        owned, particles.ghost_count(), calculator);
    const auto result = decomp_runtime_->exchange_halo(
        storage, make_halo_options(rebuild_halo, stream, !rebuild_halo));
    resize_particle_storage(owned + result.ghost_count);
    particles.reset_counts(owned, result.ghost_count);
    last_device_halo_plan_reused_ = result.plan_reused;
    // A rebuilt plan (not a reused one) can reorder ghost slots even without a
    // reallocation, which invalidates ghost-indexed neighbor lists.
    return result.storage_reallocated || result.layout_changed;
}

bool MDSystem::can_overlap_halo_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1 || !decomp_runtime_) return false;
    const size_t owned = get_local_particles();
    dsdecompx::ParticleInfo storage(
        owned, particles.ghost_count(), calculator);
    return decomp_runtime_->can_refresh_halo(storage);
}

void MDSystem::begin_halo_exchange_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecompX is not initialized");
    const size_t owned = get_local_particles();
    dsdecompx::ParticleInfo storage(
        owned, particles.ghost_count(), calculator);
    decomp_runtime_->begin_halo_exchange(
        storage, make_halo_options(false, calculator.get_halo_stream(), true));
}

void MDSystem::end_halo_exchange_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1) return;
    const size_t owned = get_local_particles();
    dsdecompx::ParticleInfo storage(
        owned, particles.ghost_count(), calculator);
    const auto result = decomp_runtime_->end_halo_exchange(storage);
    resize_particle_storage(owned + result.ghost_count);
    particles.reset_counts(owned, result.ghost_count);
    last_device_halo_plan_reused_ = result.plan_reused;
    calculator.make_force_stream_wait_for_halo();
}

bool MDSystem::migrate_particles_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1) return false;
    const size_t owned = get_local_particles();
    cudaStream_t stream = calculator.get_force_stream();
    dsdecompx::ParticleInfo storage(
        owned, particles.ghost_count(), calculator);
    const auto result = decomp_runtime_->migrate(storage, stream);
    resize_particle_storage(result.new_owned_count);
    particles.reset_counts(result.new_owned_count, 0);
    const auto& local = decomp_runtime_->topology().local_domain();
    box.xlo = local.lower[0]; box.xhi = local.upper[0];
    box.ylo = local.lower[1]; box.yhi = local.upper[1];
    box.zlo = local.lower[2]; box.zhi = local.upper[2];
    return result.ownership_changed;
}

void MDSystem::reverse_halo_on_device(ForceCalculatorCUDA& calculator) {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecompX is not initialized");
    const size_t owned = get_local_particles();
    dsdecompx::ParticleInfo storage(
        owned, particles.ghost_count(), calculator);
    decomp_runtime_->reverse_halo(storage, calculator.get_force_stream());
}
#endif

dsdecompx::HaloResult MDSystem::perform_halo_exchange(bool rebuild) {
    // A single MPI rank already owns the full periodic box.  Creating 26
    // self-images here would duplicate interactions because the serial
    // neighbor builder also applies the minimum-image convention.
    if (size == 1) {
        return {};
    }
    if (!decomp_runtime_) throw std::runtime_error("dsDecompX is not initialized");
    const auto result = decomp_runtime_->exchange_halo(
        particles, dsdecompx::HaloExchangeOptions{rebuild, nullptr});
    return result;
}

bool MDSystem::sync_halo_particles(bool rebuild_halo) {
    if (size == 1) return false;

    const auto result = perform_halo_exchange(rebuild_halo);
    refresh_local_bonded_topology();
    return result.layout_changed;
}

bool MDSystem::sync_distributed_particles(bool local_rebuild_hint) {
    if (size == 1) {
        return local_rebuild_hint;
    }

    const bool should_rebuild = unify_rebuild_decision(local_rebuild_hint);
    if (should_rebuild) {
        migrate_particles_check();
    }
    sync_halo_particles(should_rebuild);

    return should_rebuild;
}

bool MDSystem::unify_rebuild_decision(bool local_rebuild_flag) const {
    if (size == 1) return local_rebuild_flag;
    if (!decomp_runtime_) throw std::runtime_error("dsDecompX is not initialized");
    return decomp_runtime_->any_rank_true(local_rebuild_flag);
}

void MDSystem::perform_reverse_halo_exchange() {
    if (size == 1) return;
    if (!decomp_runtime_) throw std::runtime_error("dsDecompX is not initialized");
    decomp_runtime_->reverse_halo(particles);
}
