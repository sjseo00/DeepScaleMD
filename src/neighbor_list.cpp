
#include "neighbor_list.h"
#include "md_types.h"
#include "mpi_utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <mpi.h>
#include <chrono>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

// Enhanced logging with rank information
inline void log_message(int level, const std::string& message) {
    // Use simple rank check - this matches existing code pattern
    int mpi_rank = 0;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    }
    if (mpi_rank == 0) {
        std::cout << "[NeighborList] " << message << std::endl;
    }
}

inline void log_warning(const std::string& message) {
    int mpi_rank = 0;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    }
    if (mpi_rank == 0) {
        std::cerr << "[NeighborList WARNING] " << message << std::endl;
    }
}

inline void log_error(const std::string& message) {
    int mpi_rank = 0;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    }
    if (mpi_rank == 0) {
        std::cerr << "[NeighborList ERROR] " << message << std::endl;
    }
}



// VerletCellListNeighborBuilder implementation
VerletCellListNeighborBuilder::VerletCellListNeighborBuilder(Real cutoff_val, Real skin) 
    : cutoff(cutoff_val), verlet_skin(skin), update_frequency(1), delay_steps(0), info_printed(false), check_enabled(true) {
    cell_size = cutoff + verlet_skin;  // Larger cell size for Verlet skin
    // Threshold: rebuild when any particle moves more than skin/2
    max_displacement_threshold = (verlet_skin * 0.5) * (verlet_skin * 0.5);
}

void VerletCellListNeighborBuilder::initialize() {
    // Initialization handled in build method
}

void VerletCellListNeighborBuilder::finalize() {
    cells.clear();
    last_build_x.clear();
    last_build_y.clear();
    last_build_z.clear();
}

bool VerletCellListNeighborBuilder::needs_rebuild(const ParticleData& particles, size_t num_local) {
    // If check is disabled, always return false (caller controls rebuild)
    if (!check_enabled) return false;
    
    // First build always needed
    if (last_build_x.empty() || last_build_y.empty() || last_build_z.empty()) {
        return true;
    }
    
    // Check if size changed
    if (last_build_x.size() != num_local) {
        return true;
    }
    
    // Find maximum displacement since last build (local particles only)
    Real local_max_disp_sq = 0.0;
    for (size_t i = 0; i < num_local; ++i) {
        Real dx = particles.x[i] - last_build_x[i];
        Real dy = particles.y[i] - last_build_y[i];
        Real dz = particles.z[i] - last_build_z[i];
        Real disp_sq = dx*dx + dy*dy + dz*dz;
        local_max_disp_sq = std::max(local_max_disp_sq, disp_sq);
    }
    
    // Rebuild if any local particle moved more than threshold (skin/2)
    // Note: In LAMMPS-style architecture, Comm layer will ensure ghost consistency
    // Each rank makes decision based on its own owned particles
    return local_max_disp_sq > max_displacement_threshold;
}

void VerletCellListNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local) {
    //auto start = std::chrono::high_resolution_clock::now();
    
    // LAMMPS-style: num_local = owned particles, num_particles = owned + ghost
    size_t local_N = (num_local > 0) ? num_local : particles.num_particles;
    size_t total_N = particles.num_particles;  // includes ghost particles

    // Extended cutoff for neighbor search (Verlet skin included)
    const Real extended_cutoff = cutoff + verlet_skin;
    const Real rcut2 = extended_cutoff * extended_cutoff;
    nlist.cutoff = extended_cutoff; // Store search cutoff; physical cutoff remains 'cutoff'
    nlist.list_type = "verlet_cell";

    // LAMMPS-style: Each rank builds neighbor list for its local subdomain
    // Use local subdomain dimensions (includes ghost layer)
    // Calculate subdomain extent from bounds (for MPI) or use full box (for serial)
    Real Lx = (box.xhi > box.xlo) ? (box.xhi - box.xlo) : box.lx;  // Local subdomain extent in x
    Real Ly = (box.yhi > box.ylo) ? (box.yhi - box.ylo) : box.ly;  // Local subdomain extent in y
    Real Lz = (box.zhi > box.zlo) ? (box.zhi - box.zlo) : box.lz;  // Local subdomain extent in z
    
    // Compute cell grid dimensions: use extended_cutoff/2 for optimal binning (LAMMPS style)
    // This ensures each bin is roughly half the neighbor cutoff distance
    Real target_bin_size = extended_cutoff / 2.0;
    
    // Calculate number of bins based on local subdomain size
    nx = std::max(1, static_cast<int>(Lx / target_bin_size + 0.5)); // Round to nearest
    ny = std::max(1, static_cast<int>(Ly / target_bin_size + 0.5));
    nz = std::max(1, static_cast<int>(Lz / target_bin_size + 0.5));

    // Calculate actual bin size and stencil range
    Real actual_bin_size_x = Lx / nx;
    Real actual_bin_size_y = Ly / ny;
    Real actual_bin_size_z = Lz / nz;
    Real max_bin_size = std::max({actual_bin_size_x, actual_bin_size_y, actual_bin_size_z});
    
    // Stencil range: how many bins we need to search in each direction
    // Must cover extended_cutoff distance
    int stencil_range = static_cast<int>(std::ceil(extended_cutoff / max_bin_size));
    stencil_range = std::max(1, stencil_range); // At least 1

    const Real inv_lx_cell = nx / Lx;
    const Real inv_ly_cell = ny / Ly;
    const Real inv_lz_cell = nz / Lz;

    const int total_cells = nx * ny * nz;
    cells.assign(total_cells, {});

    // Precompute particle cell indices for ALL particles (owned + ghost)
    // Transform particle coordinates to local subdomain frame by subtracting origin
    std::vector<int> cx(total_N);
    std::vector<int> cy(total_N);
    std::vector<int> cz(total_N);
    
    // Subdomain origin (for serial: xlo=ylo=zlo=0)
    Real origin_x = (box.xhi > box.xlo) ? box.xlo : 0.0;
    Real origin_y = (box.yhi > box.ylo) ? box.ylo : 0.0;
    Real origin_z = (box.zhi > box.zlo) ? box.zlo : 0.0;
    
    for (size_t i = 0; i < total_N; ++i) {
        // Transform to local subdomain coordinates by subtracting origin
        // This works whether coordinates are global (MPI) or already local (serial)
        Real rx = particles.x[i] - origin_x;
        Real ry = particles.y[i] - origin_y;
        Real rz = particles.z[i] - origin_z;
        
        int ix = static_cast<int>(rx * inv_lx_cell);
        int iy = static_cast<int>(ry * inv_ly_cell);
        int iz = static_cast<int>(rz * inv_lz_cell);

        // For local subdomain, clamp to valid cell range (no periodic wrapping within subdomain)
        // Comm layer handles periodicity via ghost exchange
        ix = std::clamp(ix, 0, nx - 1);
        iy = std::clamp(iy, 0, ny - 1);
        iz = std::clamp(iz, 0, nz - 1);

        cx[i] = ix; 
        cy[i] = iy; 
        cz[i] = iz;
        
        int cell_id = iz * nx * ny + iy * nx + ix;
        if (cell_id >= 0 && cell_id < total_cells) {
            cells[cell_id].push_back(static_cast<int>(i));
        }
    }

    // Initialize neighbor list arrays (sized for LOCAL particles only)
    nlist.neighbors.clear();
    nlist.num_neighbors.assign(local_N, 0);
    nlist.neighbor_offset.assign(local_N, 0);

    // Reserve approximate neighbor capacity to reduce reallocations
    if (local_N > 0) {
        Real cell_volume = (Lx * Ly * Lz) / static_cast<Real>(total_cells);
        Real sphere_volume = (4.0/3.0) * M_PI * extended_cutoff * extended_cutoff * extended_cutoff;
        size_t est_neighbors_per_particle = static_cast<size_t>(sphere_volume / cell_volume * 1.2);
        est_neighbors_per_particle = std::min<size_t>(est_neighbors_per_particle, 512);
        nlist.neighbors.reserve(local_N * est_neighbors_per_particle);
    }

    // Neighbors (j) can be from owned OR ghost particles (0..total_N-1)
    for (size_t i = 0; i < local_N; ++i) {
        std::vector<int> local_neighbors;
        local_neighbors.reserve(64);

        int ix = cx[i];
        int iy = cy[i];
        int iz = cz[i];

        for (int dz = -stencil_range; dz <= stencil_range; ++dz) {
            int zc = iz + dz;
            // Apply periodic wrapping for cells at boundaries
            bool wrapped = false;
            if (box.periodic_z) {
                if (zc < 0) { zc += nz; wrapped = true; }
                else if (zc >= nz) { zc -= nz; wrapped = true; }
            } else {
                if (zc < 0 || zc >= nz) continue;
            }
            
            for (int dy = -stencil_range; dy <= stencil_range; ++dy) {
                int yc = iy + dy;
                if (box.periodic_y) {
                    if (yc < 0) { yc += ny; wrapped = true; }
                    else if (yc >= ny) { yc -= ny; wrapped = true; }
                } else {
                    if (yc < 0 || yc >= ny) continue;
                }
                
                for (int dx = -stencil_range; dx <= stencil_range; ++dx) {
                    int xc = ix + dx;
                    if (box.periodic_x) {
                        if (xc < 0) { xc += nx; wrapped = true; }
                        else if (xc >= nx) { xc -= nx; wrapped = true; }
                    } else {
                        if (xc < 0 || xc >= nx) continue;
                    }
                    
                    int cell_id = zc * nx * ny + yc * nx + xc;
                    
                    // Bounds check
                    if (cell_id < 0 || cell_id >= total_cells) continue;
                    
                    for (int j : cells[cell_id]) {
                        if (j == static_cast<int>(i)) continue; // Skip self
                        
                        // Half-list with Newton on: only store j > i for owned-owned pairs
                        // For owned-ghost pairs (j >= local_N), always store
                        if (j < static_cast<int>(local_N) && j <= static_cast<int>(i)) continue;
                            
                        
                        Real dx_ = particles.x[i] - particles.x[j];
                        Real dy_ = particles.y[i] - particles.y[j];
                        Real dz_ = particles.z[i] - particles.z[j];
                        
                        // Apply minimum image convention
                        // Note: With proper ghost exchange, this should rarely trigger
                        box.apply_minimum_image(dx_, dy_, dz_);
                        Real r2 = dx_*dx_ + dy_*dy_ + dz_*dz_;
                        
                        if (r2 < rcut2) {
                            local_neighbors.push_back(j);
                        }
                    }
                }
            }
        }

        nlist.neighbor_offset[i] = static_cast<int>(nlist.neighbors.size());
        nlist.num_neighbors[i] = static_cast<int>(local_neighbors.size());
        
        if (!local_neighbors.empty()) {
            nlist.neighbors.insert(nlist.neighbors.end(), local_neighbors.begin(), local_neighbors.end());
        }

    }

    
    // Print neighbor list information (LAMMPS-style logging) - only once per rank
    // Note: Each rank will log its own statistics
    if (!info_printed) {
        // Calculate max neighbors per atom and average for LOCAL particles
        int max_neighbors = 0;
        size_t total_neighbors = 0;
        if (!nlist.num_neighbors.empty()) {
            max_neighbors = *std::max_element(nlist.num_neighbors.begin(), nlist.num_neighbors.end());
            for (size_t i = 0; i < nlist.num_neighbors.size(); ++i) {
                total_neighbors += nlist.num_neighbors[i];
            }
        }
        Real avg_neighbors = (local_N > 0) ? static_cast<Real>(total_neighbors) / static_cast<Real>(local_N) : 0.0;
        
        // Calculate actual bin size (should be approximately extended_cutoff/2)
        Real bin_size = std::min({Lx / nx, Ly / ny, Lz / nz});
        
        // Estimate memory page size (typical value for memory management)
        
        std::cout << "\nNeighbor list info ..." << std::endl;
        std::cout << "  update every " << update_frequency << " steps, delay " << delay_steps << " steps, check yes" << std::endl;
        std::cout << "  max neighbors/atom: " << max_neighbors <<  std::endl;
        std::cout << "  master list distance cutoff = " << extended_cutoff << std::endl;
        std::cout << "  ghost atom cutoff = " << extended_cutoff << std::endl;
        std::cout << "  binsize = " << bin_size << ", bins = " << nx << " " << ny << " " << nz << std::endl;
        std::cout << "  1 neighbor lists, perpetual/occasional/extra = 1 0 0" << std::endl;
        std::cout << "  (1) pair lj/cut, perpetual" << std::endl;
        std::cout << "      attributes: half, newton on (within rank), newton off (across MPI ranks)" << std::endl;
        std::cout << "      pair build: half/bin/cell" << std::endl;
        std::cout << "      stencil: half/bin/3d" << std::endl;
        std::cout << "      bin: standard" << std::endl;
        std::cout << std::endl;
        std::cout << "Total # of neighbors = " << total_neighbors << std::endl;
        std::cout << "Ave neighs/atom = " << avg_neighbors << std::endl;
        std::cout << std::endl;
        
        info_printed = true;  // Set flag to prevent future printing
    }

    // Save current positions for next rebuild check (local owned particles only)
    if (check_enabled) {
        last_build_x.resize(local_N);
        last_build_y.resize(local_N);
        last_build_z.resize(local_N);
        for (size_t i = 0; i < local_N; ++i) {
            last_build_x[i] = particles.x[i];
            last_build_y[i] = particles.y[i];
            last_build_z[i] = particles.z[i];
        }
    }
}

void VerletCellListNeighborBuilder::broadcast_neighbor_list(NeighborList& nlist, size_t n, int mpi_rank, int mpi_size) {
    if (mpi_size <= 1) return;

    struct NeighborListHeader {
        int total_neighbors;
        Real cutoff_val;
    } header;

    if (mpi_rank == 0) {
        header.total_neighbors = static_cast<int>(nlist.neighbors.size());
        header.cutoff_val = nlist.cutoff;
    }

    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        nlist.cutoff = header.cutoff_val;
        nlist.neighbors.resize(header.total_neighbors);
        nlist.num_neighbors.assign(n, 0);
        nlist.neighbor_offset.assign(n, 0);
    }

    if (header.total_neighbors > 0) {
        MPI_Bcast(nlist.neighbors.data(), header.total_neighbors, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(nlist.num_neighbors.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(nlist.neighbor_offset.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
    }
}

// Forward declaration for CUDA kernel launcher
#ifdef USE_CUDA
extern "C" void launch_cuda_neighbor_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int* d_neighbor_list,
    int* d_num_neighbors,
    int* d_neighbor_offset,
    int n_local, int n_total,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors_per_particle,
    cudaStream_t stream);
#endif

// Morton Z-order curve based neighbor list implementation
// Optimized for MPI parallelism and CUDA/SYCL GPU acceleration

MortonZOrderNeighborBuilder::MortonZOrderNeighborBuilder(Real cutoff_val, Real skin, int tile_size) 
    : cutoff(cutoff_val), verlet_skin(skin), morton_levels(10), update_frequency(1), delay_steps(0), info_printed(false) {
    tile_staging.tile_size = tile_size;
    // Initialize GPU memory pointers to nullptr
    d_neighbor_list = nullptr;
    d_tile_data = nullptr;
    gpu_memory_allocated = false;
    gpu_available = false;
    gpu_device_id = 0;
}

void MortonZOrderNeighborBuilder::initialize() {   
    // Initialize GPU context if available
    #ifdef USE_CUDA
    cudaError_t err = cudaGetDevice(&gpu_device_id);
    if (err == cudaSuccess) {
        gpu_available = true;
        log_message(1, "CUDA device " + std::to_string(gpu_device_id) + " detected for Morton Z-order neighbor list");
    } else {
        gpu_available = false;
        log_warning("CUDA not available for Morton Z-order, using CPU fallback");
    }
    #elif defined(USE_SYCL)
    try {
        sycl_queue = sycl::queue(sycl::gpu_selector_v);
        gpu_available = true;
        if (is_rank_zero()) {
            std::cout << "SYCL GPU queue created for Morton Z-order neighbor list" << std::endl;
        }
    } catch (const std::exception& e) {
        gpu_available = false;
        if (is_rank_zero()) {
            std::cout << "Warning: SYCL GPU not available, using CPU fallback: " << e.what() << std::endl;
        }
    }
    #else
    gpu_available = false;
    #endif
}

void MortonZOrderNeighborBuilder::finalize() {
    // Clean up GPU memory if allocated with proper error handling
    if (gpu_memory_allocated) {
        #ifdef USE_CUDA
        if (d_neighbor_list != nullptr) {
            cudaError_t err = cudaFree(d_neighbor_list);
            if (err != cudaSuccess && is_rank_zero()) {
                std::cout << "Warning: Failed to free d_neighbor_list: " << cudaGetErrorString(err) << std::endl;
            }
            d_neighbor_list = nullptr;
        }
        if (d_tile_data != nullptr) {
            cudaError_t err = cudaFree(d_tile_data);
            if (err != cudaSuccess && is_rank_zero()) {
                std::cout << "Warning: Failed to free d_tile_data: " << cudaGetErrorString(err) << std::endl;
            }
            d_tile_data = nullptr;
        }
        #elif defined(USE_SYCL)
        try {
            if (d_neighbor_list != nullptr) {
                sycl::free(d_neighbor_list, sycl_queue);
                d_neighbor_list = nullptr;
            }
            if (d_tile_data != nullptr) {
                sycl::free(d_tile_data, sycl_queue);
                d_tile_data = nullptr;
            }
        } catch (const std::exception& e) {
            if (is_rank_zero()) {
                std::cout << "Warning: SYCL memory deallocation failed: " << e.what() << std::endl;
            }
        }
        #endif
        gpu_memory_allocated = false;
    }
    
    // Clean up CPU data structures
    morton_codes.clear();
    particle_reorder.clear();
    inverse_reorder.clear();
    csr_nlist.neighbors.clear();
    csr_nlist.row_ptr.clear();
    csr_nlist.col_indices.clear();
    tile_staging.tiles.clear();
    
}

// Optimized Morton Z-order encoding using bit interleaving
// This creates a space-filling curve that preserves spatial locality
uint64_t MortonZOrderNeighborBuilder::morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    // Efficient bit interleaving using precomputed masks
    auto split_by_3 = [](uint32_t a) -> uint64_t {
        uint64_t x = a & 0x1fffff; // Use first 21 bits for high resolution
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8)  & 0x100f00f00f00f00f;
        x = (x | x << 4)  & 0x10c30c30c30c30c3;
        x = (x | x << 2)  & 0x1249249249249249;
        return x;
    };
    
    // Interleave bits: z,y,x pattern for better memory coalescing
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2);
}

void MortonZOrderNeighborBuilder::morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) {
    // Reverse bit interleaving
    auto compact_by_3 = [](uint64_t x) -> uint32_t {
        x = x & 0x1249249249249249;
        x = (x | (x >> 2))  & 0x10c30c30c30c30c3;
        x = (x | (x >> 4))  & 0x100f00f00f00f00f;
        x = (x | (x >> 8))  & 0x1f0000ff0000ff;
        x = (x | (x >> 16)) & 0x1f00000000ffff;
        x = (x | (x >> 32)) & 0x1fffff;
        return static_cast<uint32_t>(x);
    };
    
    x = compact_by_3(morton);
    y = compact_by_3(morton >> 1);
    z = compact_by_3(morton >> 2);
}

void MortonZOrderNeighborBuilder::reorder_particles(ParticleData& particles) {
    size_t n = particles.num_particles;
    
    // Create backup of original data
    std::vector<Real> orig_x = particles.x, orig_y = particles.y, orig_z = particles.z;
    std::vector<Real> orig_vx = particles.vx, orig_vy = particles.vy, orig_vz = particles.vz;
    std::vector<Real> orig_mass = particles.mass, orig_charge = particles.charge;
    std::vector<int> orig_type = particles.type;
    std::vector<int> orig_gid = particles.global_id;
    
    // Reorder according to Morton codes
    for (size_t i = 0; i < n; i++) {
        int new_idx = particle_reorder[i];
        particles.x[i] = orig_x[new_idx];
        particles.y[i] = orig_y[new_idx];
        particles.z[i] = orig_z[new_idx];
        particles.vx[i] = orig_vx[new_idx];
        particles.vy[i] = orig_vy[new_idx];
        particles.vz[i] = orig_vz[new_idx];
        particles.mass[i] = orig_mass[new_idx];
        particles.charge[i] = orig_charge[new_idx];
        particles.type[i] = orig_type[new_idx];
        particles.global_id[i] = orig_gid[new_idx];
    }
}

void MortonZOrderNeighborBuilder::restore_particle_order(ParticleData& particles) {
    size_t n = particles.num_particles;
    
    // Create backup of reordered data
    std::vector<Real> reord_x = particles.x, reord_y = particles.y, reord_z = particles.z;
    std::vector<Real> reord_vx = particles.vx, reord_vy = particles.vy, reord_vz = particles.vz;
    std::vector<Real> reord_mass = particles.mass, reord_charge = particles.charge;
    std::vector<int> reord_type = particles.type;
    std::vector<int> reord_gid = particles.global_id;
    
    // Restore original order
    for (size_t i = 0; i < n; i++) {
        int orig_idx = inverse_reorder[i];
        particles.x[orig_idx] = reord_x[i];
        particles.y[orig_idx] = reord_y[i];
        particles.z[orig_idx] = reord_z[i];
        particles.vx[orig_idx] = reord_vx[i];
        particles.vy[orig_idx] = reord_vy[i];
        particles.vz[orig_idx] = reord_vz[i];
        particles.mass[orig_idx] = reord_mass[i];
        particles.charge[orig_idx] = reord_charge[i];
        particles.type[orig_idx] = reord_type[i];
        particles.global_id[orig_idx] = reord_gid[i];
    }
}

void MortonZOrderNeighborBuilder::build_csr_neighbor_list(ParticleData& particles, SimulationBox& box, int max_neighbors_per_particle) {
    size_t n = particles.num_particles;
    Real extended_cutoff = cutoff + verlet_skin;
    Real cutoff_sq = extended_cutoff * extended_cutoff;
    
    // Clear CSR structures
    csr_nlist.neighbors.clear();
    csr_nlist.row_ptr.clear();
    csr_nlist.col_indices.clear();
    
    csr_nlist.row_ptr.resize(n + 1, 0);
    
    // Build neighbor list in CSR format
    for (size_t i = 0; i < n; i++) {
        csr_nlist.row_ptr[i] = csr_nlist.neighbors.size();
        
        for (size_t j = i + 1; j < n; j++) {
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply minimum image convention
            box.apply_minimum_image(dx, dy, dz);
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff_sq) {
                csr_nlist.neighbors.push_back(j);
                csr_nlist.col_indices.push_back(j);
                // Limit neighbors per particle
                if (static_cast<int>(csr_nlist.neighbors.size() - csr_nlist.row_ptr[i]) >= max_neighbors_per_particle) {
                    break;
                }
            }
        }
    }
    
    csr_nlist.row_ptr[n] = csr_nlist.neighbors.size();
}

void MortonZOrderNeighborBuilder::create_tile_staging() {
    size_t n_neighbors = csr_nlist.neighbors.size();
    int n_tiles = (n_neighbors + tile_staging.tile_size - 1) / tile_staging.tile_size;
    
    tile_staging.tiles.clear();
    tile_staging.tiles.resize(n_tiles);
    
    // Group neighbors into tiles for GPU memory coalescing
    for (size_t i = 0; i < n_neighbors; i++) {
        int tile_id = i / tile_staging.tile_size;
        tile_staging.tiles[tile_id].push_back(csr_nlist.neighbors[i]);
    }
}

// Optimized Morton Z-order neighbor list build with Cell List for O(n) complexity
void MortonZOrderNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local) {
    //auto start = std::chrono::high_resolution_clock::now();
    

    // LAMMPS-style: num_local = owned particles, num_particles = owned + ghost
    size_t n_local = (num_local > 0) ? num_local : particles.num_particles;
    size_t n_total = particles.num_particles;  // includes ghost particles
    
    // Initialize neighbor list structure - sized for LOCAL particles only
    nlist.neighbors.clear();
    nlist.num_neighbors.assign(n_local, 0);
    nlist.neighbor_offset.assign(n_local, 0);
    nlist.cutoff = cutoff + verlet_skin;
    nlist.list_type = "morton_z_order";
    
    Real extended_cutoff = cutoff + verlet_skin;
    Real cutoff_sq = extended_cutoff * extended_cutoff;
    
#ifdef USE_CUDA
    // === CUDA Path: Build neighbor list directly on GPU ===
    if (gpu_available) {
        const int max_neighbors_per_particle = 512;
        
        // Allocate GPU memory for particles
        Real *d_x, *d_y, *d_z;
        cudaMalloc(&d_x, n_total * sizeof(Real));
        cudaMalloc(&d_y, n_total * sizeof(Real));
        cudaMalloc(&d_z, n_total * sizeof(Real));
        
        // Copy particle positions to GPU
        cudaMemcpy(d_x, particles.x.data(), n_total * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, particles.y.data(), n_total * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, particles.z.data(), n_total * sizeof(Real), cudaMemcpyHostToDevice);
        
        // Allocate GPU memory for neighbor list
        int* d_neighbor_list_gpu;
        int* d_num_neighbors_gpu;
        int* d_neighbor_offset_gpu;
        
        cudaMalloc(&d_neighbor_list_gpu, n_local * max_neighbors_per_particle * sizeof(int));
        cudaMalloc(&d_num_neighbors_gpu, n_local * sizeof(int));
        cudaMalloc(&d_neighbor_offset_gpu, n_local * sizeof(int));
        
        // Launch CUDA kernel
        cudaStream_t stream = 0;
        launch_cuda_neighbor_list_build(
            d_x, d_y, d_z,
            d_neighbor_list_gpu, d_num_neighbors_gpu, d_neighbor_offset_gpu,
            n_local, n_total,
            cutoff, verlet_skin,
            box.lx, box.ly, box.lz,
            box.periodic_x, box.periodic_y, box.periodic_z,
            max_neighbors_per_particle,
            stream
        );
        
        cudaDeviceSynchronize();
        
        // Copy results back to host
        std::vector<int> h_num_neighbors(n_local);
        cudaMemcpy(h_num_neighbors.data(), d_num_neighbors_gpu, n_local * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Compact neighbor list (remove unused space)
        size_t total_neighbors = 0;
        for (size_t i = 0; i < n_local; i++) {
            total_neighbors += h_num_neighbors[i];
        }
        
        nlist.neighbors.resize(total_neighbors);
        nlist.num_neighbors = h_num_neighbors;
        
        // Copy neighbor data with offsets
        std::vector<int> h_neighbor_list_full(n_local * max_neighbors_per_particle);
        cudaMemcpy(h_neighbor_list_full.data(), d_neighbor_list_gpu, 
                   n_local * max_neighbors_per_particle * sizeof(int), cudaMemcpyDeviceToHost);
        
        size_t offset = 0;
        for (size_t i = 0; i < n_local; i++) {
            nlist.neighbor_offset[i] = offset;
            int n_neigh = h_num_neighbors[i];
            for (int j = 0; j < n_neigh; j++) {
                nlist.neighbors[offset + j] = h_neighbor_list_full[i * max_neighbors_per_particle + j];
            }
            offset += n_neigh;
        }
        
        // Cleanup
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_neighbor_list_gpu);
        cudaFree(d_num_neighbors_gpu);
        cudaFree(d_neighbor_offset_gpu);
        
        //auto end = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        /*
        if (mpi_rank == 0) {
            std::cout << "Morton Z-order + Cell List neighbor build (CUDA): " << duration << " ms "
                      << "(local particles: " << n_local << ", total: " << n_total 
                      << ", neighbors: " << total_neighbors << ")" << std::endl;
        }
                      */
        return;
    }
#endif
    
    // === CPU Path: Cell list + Morton Z-order for cache locality ===
    // LAMMPS-style: Use local subdomain dimensions
    Real Lx = box.lx;  // Local subdomain extent
    Real Ly = box.ly;
    Real Lz = box.lz;
    
    Real target_bin_size = extended_cutoff / 2.0;
    int nx = std::max(1, static_cast<int>(Lx / target_bin_size + 0.5));
    int ny = std::max(1, static_cast<int>(Ly / target_bin_size + 0.5));
    int nz = std::max(1, static_cast<int>(Lz / target_bin_size + 0.5));
    
    Real inv_cell_x = nx / Lx;
    Real inv_cell_y = ny / Ly;
    Real inv_cell_z = nz / Lz;
    
    int total_cells = nx * ny * nz;
    std::vector<std::vector<int>> cells(total_cells);
    
    // Assign all particles (local + ghost) to cells
    // Note: Ghost particles are already positioned in local subdomain by Comm layer
    std::vector<int> particle_cell_id(n_total);
    
    for (size_t i = 0; i < n_total; i++) {
        Real px = particles.x[i];
        Real py = particles.y[i];
        Real pz = particles.z[i];
        
        // For local subdomain: coordinates are already in local frame
        // When subdomain bounds (xlo, xhi, etc.) are added, shift by: px - box.xlo
        
        int cx = static_cast<int>(px * inv_cell_x);
        int cy = static_cast<int>(py * inv_cell_y);
        int cz = static_cast<int>(pz * inv_cell_z);
        
        // Clamp to valid local cell range (no periodic wrapping within subdomain)
        cx = std::max(0, std::min(nx - 1, cx));
        cy = std::max(0, std::min(ny - 1, cy));
        cz = std::max(0, std::min(nz - 1, cz));
        
        int cell_idx = cz * nx * ny + cy * nx + cx;
        particle_cell_id[i] = cell_idx;
    }
    
    // Populate cells (serial to avoid race condition)
    for (size_t i = 0; i < n_total; i++) {
        cells[particle_cell_id[i]].push_back(i);
    }
    
    // === Step 2: Morton Z-order encoding for spatial coherence (optional optimization) ===
    // This helps with cache locality within cells
    morton_codes.clear();
    morton_codes.reserve(n_total);
    
    const uint32_t grid_res = 1 << morton_levels;
    const Real inv_lx = grid_res / Lx;
    const Real inv_ly = grid_res / Ly;
    const Real inv_lz = grid_res / Lz;
    
    std::vector<MortonCode> local_codes(n_total);
    
    for (size_t i = 0; i < n_total; i++) {
        Real norm_x = particles.x[i];
        Real norm_y = particles.y[i];
        Real norm_z = particles.z[i];
        
        // For local subdomain: when bounds are added, shift by: norm_x - box.xlo
        // For now, assume coordinates are in [0, Lx) range
        
        uint32_t x = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), norm_x * inv_lx))) & (grid_res - 1);
        uint32_t y = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), norm_y * inv_ly))) & (grid_res - 1);
        uint32_t z = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), norm_z * inv_lz))) & (grid_res - 1);
        
        local_codes[i].code = morton_encode_3d(x, y, z);
        local_codes[i].particle_id = static_cast<int>(i);
    }
    
    morton_codes = std::move(local_codes);
    
    // Sort by Morton code for better cache locality
    std::sort(morton_codes.begin(), morton_codes.end());
    
    // === Step 3: Build neighbor list using Cell List (O(n) complexity) ===
    // LAMMPS-style: Build only for owned particles (0..n_local-1)
    size_t estimated_neighbors = n_local * 50;
    nlist.neighbors.reserve(estimated_neighbors);
    
    {
        std::vector<int> thread_local_neighbors;
        thread_local_neighbors.reserve(100);
        
        for (size_t i = 0; i < n_local; i++) {  // Only owned particles
            thread_local_neighbors.clear();
            
            Real xi = particles.x[i];
            Real yi = particles.y[i];
            Real zi = particles.z[i];
            
            // Get cell of particle i in local subdomain
            Real px = xi;  // When subdomain bounds added: xi - box.xlo
            Real py = yi;
            Real pz = zi;
            
            int cx = std::clamp(static_cast<int>(px * inv_cell_x), 0, nx - 1);
            int cy = std::clamp(static_cast<int>(py * inv_cell_y), 0, ny - 1);
            int cz = std::clamp(static_cast<int>(pz * inv_cell_z), 0, nz - 1);
            
            // Search in 3x3x3 neighboring cells (27 cells total)
            // Apply periodic boundary conditions for cells
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ncx = cx + dx;
                        int ncy = cy + dy;
                        int ncz = cz + dz;
                        
                        // Apply periodic wrapping for cells at boundaries
                        bool wrapped = false;
                        if (box.periodic_x) {
                            if (ncx < 0) { ncx += nx; wrapped = true; }
                            else if (ncx >= nx) { ncx -= nx; wrapped = true; }
                        } else {
                            if (ncx < 0 || ncx >= nx) continue;
                        }
                        
                        if (box.periodic_y) {
                            if (ncy < 0) { ncy += ny; wrapped = true; }
                            else if (ncy >= ny) { ncy -= ny; wrapped = true; }
                        } else {
                            if (ncy < 0 || ncy >= ny) continue;
                        }
                        
                        if (box.periodic_z) {
                            if (ncz < 0) { ncz += nz; wrapped = true; }
                            else if (ncz >= nz) { ncz -= nz; wrapped = true; }
                        } else {
                            if (ncz < 0 || ncz >= nz) continue;
                        }
                        
                        int neighbor_cell = ncz * nx * ny + ncy * nx + ncx;
                        
                        // Check all particles in this neighboring cell (owned + ghost)
                        for (int j : cells[neighbor_cell]) {
                            if (j == static_cast<int>(i)) continue;  // Skip self
                            
                            // Half-list logic: for owned-owned pairs (j < n_local), only store j > i
                            // For owned-ghost pairs (j >= n_local), always store
                            if (j < static_cast<int>(n_local) && j <= static_cast<int>(i)) continue;
                            
                            Real dx_ij = xi - particles.x[j];
                            Real dy_ij = yi - particles.y[j];
                            Real dz_ij = zi - particles.z[j];
                            
                            box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
                            
                            Real r2 = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;
                            
                            if (r2 < cutoff_sq) {
                                thread_local_neighbors.push_back(j);
                            }
                        }
                    }
                }
            }
            
            // Store neighbors (critical section)
            {
                nlist.neighbor_offset[i] = nlist.neighbors.size();
                nlist.num_neighbors[i] = thread_local_neighbors.size();
                nlist.neighbors.insert(nlist.neighbors.end(), 
                                      thread_local_neighbors.begin(), 
                                      thread_local_neighbors.end());
            }
        }
    }
    
    // === Step 4: Copy to GPU if available (async for efficiency) ===
    if (gpu_available) {
        copy_to_gpu(nlist);
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Print neighbor list information (LAMMPS-style logging) - only once per rank
    if (!info_printed) {
        // Calculate max neighbors per atom
        int max_neighbors = 0;
        if (!nlist.num_neighbors.empty()) {
            max_neighbors = *std::max_element(nlist.num_neighbors.begin(), nlist.num_neighbors.end());
        }
        
        
        // Calculate actual bin size (using local cell grid parameters)
        Real bin_size = std::min({Lx / nx, Ly / ny, Lz / nz});
        Real extended_cutoff = cutoff + verlet_skin;
        
        std::cout << "\nNeighbor list info ..." << std::endl;
        std::cout << "  update every " << update_frequency << " steps, delay " << delay_steps << " steps, check yes" << std::endl;
        std::cout << "  max neighbors/atom: " << max_neighbors << std::endl;
        std::cout << "  master list distance cutoff = " << extended_cutoff << std::endl;
        std::cout << "  ghost atom cutoff = " << extended_cutoff << std::endl;
        std::cout << "  binsize = " << bin_size << ", bins = " << nx << " " << ny << " " << nz << std::endl;
        std::cout << std::endl;
        
        info_printed = true;  // Set flag to prevent future printing
    }
    
}

void MortonZOrderNeighborBuilder::copy_to_gpu(NeighborList& nlist) {
    if (!gpu_available) {
        if (is_rank_zero()) {
            std::cout << "GPU not available, skipping GPU memory transfer" << std::endl;
        }
        return;
    }
    

    
    
    #ifdef USE_CUDA
    // Allocate GPU memory for neighbor list
    size_t neighbor_list_size = nlist.neighbors.size() * sizeof(int);
    size_t num_neighbors_size = nlist.num_neighbors.size() * sizeof(int);
    size_t offset_size = nlist.neighbor_offset.size() * sizeof(int);
    size_t total_size = neighbor_list_size + num_neighbors_size + offset_size;
    
    // Enhanced input validation
    if (total_size == 0) {
        if (is_rank_zero()) {
            std::cout << "Warning: Empty neighbor list, skipping GPU transfer" << std::endl;
        }
        return;
    }
    

    

    
    // Clean up existing GPU memory with proper error checking
    if (gpu_memory_allocated) {
        if (d_neighbor_list != nullptr) {
            cudaError_t err = cudaFree(d_neighbor_list);
            if (err != cudaSuccess && is_rank_zero()) {
                std::cout << "Warning: Failed to free d_neighbor_list: " << cudaGetErrorString(err) << std::endl;
            }
            d_neighbor_list = nullptr;
        }
        if (d_tile_data != nullptr) {
            cudaError_t err = cudaFree(d_tile_data);
            if (err != cudaSuccess && is_rank_zero()) {
                std::cout << "Warning: Failed to free d_tile_data: " << cudaGetErrorString(err) << std::endl;
            }
            d_tile_data = nullptr;
        }
        gpu_memory_allocated = false;
    }
    
    // Allocate new GPU memory
    cudaError_t err = cudaMalloc(&d_neighbor_list, total_size);
    if (err != cudaSuccess) {
        if (is_rank_zero()) {
            std::cout << "Failed to allocate GPU memory for neighbor list (" 
                      << total_size << " bytes): " << cudaGetErrorString(err) << std::endl;
        }
        d_neighbor_list = nullptr;
        return;
    }
    
    // Copy data to GPU with proper error handling
    char* gpu_ptr = static_cast<char*>(d_neighbor_list);
    
    if (neighbor_list_size > 0) {
        err = cudaMemcpy(gpu_ptr, nlist.neighbors.data(), neighbor_list_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (is_rank_zero()) {
                std::cout << "Failed to copy neighbor list to GPU: " << cudaGetErrorString(err) << std::endl;
            }
            cudaFree(d_neighbor_list);
            d_neighbor_list = nullptr;
            return;
        }
        gpu_ptr += neighbor_list_size;
    }
    
    if (num_neighbors_size > 0) {
        err = cudaMemcpy(gpu_ptr, nlist.num_neighbors.data(), num_neighbors_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (is_rank_zero()) {
                std::cout << "Failed to copy num_neighbors to GPU: " << cudaGetErrorString(err) << std::endl;
            }
            cudaFree(d_neighbor_list);
            d_neighbor_list = nullptr;
            return;
        }
        gpu_ptr += num_neighbors_size;
    }
    
    if (offset_size > 0) {
        err = cudaMemcpy(gpu_ptr, nlist.neighbor_offset.data(), offset_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (is_rank_zero()) {
                std::cout << "Failed to copy neighbor_offset to GPU: " << cudaGetErrorString(err) << std::endl;
            }
            cudaFree(d_neighbor_list);
            d_neighbor_list = nullptr;
            return;
        }
    }
    
    // Allocate and copy tile data if available
    if (!tile_staging.tiles.empty()) {
        size_t tile_data_size = tile_staging.tiles.size() * tile_staging.tile_size * sizeof(int);
        err = cudaMalloc(&d_tile_data, tile_data_size);
        if (err == cudaSuccess) {
            // Flatten tile data for GPU (optimized with reserve)
            std::vector<int> flat_tiles;
            flat_tiles.reserve(tile_staging.tiles.size() * tile_staging.tile_size);
            
            for (const auto& tile : tile_staging.tiles) {
                flat_tiles.insert(flat_tiles.end(), tile.begin(), tile.end());
                // Pad to tile_size if necessary
                size_t padding = tile_staging.tile_size - (tile.size() % tile_staging.tile_size);
                if (padding < tile_staging.tile_size) {
                    flat_tiles.insert(flat_tiles.end(), padding, -1); // Invalid neighbor marker
                }
            }
            
            err = cudaMemcpy(d_tile_data, flat_tiles.data(), flat_tiles.size() * sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                if (is_rank_zero()) {
                    std::cout << "Warning: Failed to copy tile data to GPU: " << cudaGetErrorString(err) << std::endl;
                }
                cudaFree(d_tile_data);
                d_tile_data = nullptr;
            }
        } else {
            if (is_rank_zero()) {
                std::cout << "Warning: Failed to allocate GPU memory for tile data: " << cudaGetErrorString(err) << std::endl;
            }
            d_tile_data = nullptr;
        }
    }
    
    gpu_memory_allocated = true;
    
    #elif defined(USE_SYCL)
    try {
        // Allocate GPU memory for neighbor list using SYCL
        size_t neighbor_list_size = nlist.neighbors.size();
        size_t num_neighbors_size = nlist.num_neighbors.size();
        size_t offset_size = nlist.neighbor_offset.size();
        size_t total_elements = neighbor_list_size + num_neighbors_size + offset_size;
        
        // Enhanced input validation
        if (total_elements == 0) {
            if (is_rank_zero()) {
                std::cout << "Warning: Empty neighbor list, skipping GPU transfer" << std::endl;
            }
            return;
        }
        

        
        // Clean up existing GPU memory with proper error handling
        if (gpu_memory_allocated) {
            if (d_neighbor_list != nullptr) {
                try {
                    sycl::free(d_neighbor_list, sycl_queue);
                } catch (const std::exception& e) {
                    if (is_rank_zero()) {
                        std::cout << "Warning: Failed to free d_neighbor_list: " << e.what() << std::endl;
                    }
                }
                d_neighbor_list = nullptr;
            }
            if (d_tile_data != nullptr) {
                try {
                    sycl::free(d_tile_data, sycl_queue);
                } catch (const std::exception& e) {
                    if (is_rank_zero()) {
                        std::cout << "Warning: Failed to free d_tile_data: " << e.what() << std::endl;
                    }
                }
                d_tile_data = nullptr;
            }
            gpu_memory_allocated = false;
        }
        
        // Allocate new GPU memory
        d_neighbor_list = sycl::malloc_device<int>(total_elements, sycl_queue);
        if (!d_neighbor_list) {
            if (is_rank_zero()) {
                std::cout << "Failed to allocate GPU memory for neighbor list (" 
                          << total_elements * sizeof(int) << " bytes)" << std::endl;
            }
            return;
        }
        
        // Copy data to GPU with proper error handling
        int* gpu_ptr = static_cast<int*>(d_neighbor_list);
        
        if (neighbor_list_size > 0) {
            sycl_queue.memcpy(gpu_ptr, nlist.neighbors.data(), neighbor_list_size * sizeof(int)).wait();
            gpu_ptr += neighbor_list_size;
        }
        
        if (num_neighbors_size > 0) {
            sycl_queue.memcpy(gpu_ptr, nlist.num_neighbors.data(), num_neighbors_size * sizeof(int)).wait();
            gpu_ptr += num_neighbors_size;
        }
        
        if (offset_size > 0) {
            sycl_queue.memcpy(gpu_ptr, nlist.neighbor_offset.data(), offset_size * sizeof(int)).wait();
        }
        
        // Allocate and copy tile data if available
        if (!tile_staging.tiles.empty()) {
            size_t tile_data_elements = tile_staging.tiles.size() * tile_staging.tile_size;
            
            d_tile_data = sycl::malloc_device<int>(tile_data_elements, sycl_queue);
            if (d_tile_data) {
                // Flatten tile data for GPU (optimized with reserve)
                std::vector<int> flat_tiles;
                flat_tiles.reserve(tile_data_elements);
                
                for (const auto& tile : tile_staging.tiles) {
                    flat_tiles.insert(flat_tiles.end(), tile.begin(), tile.end());
                    // Pad to tile_size if necessary
                    size_t padding = tile_staging.tile_size - (tile.size() % tile_staging.tile_size);
                    if (padding < tile_staging.tile_size) {
                        flat_tiles.insert(flat_tiles.end(), padding, -1);
                    }
                }
                
                sycl_queue.memcpy(d_tile_data, flat_tiles.data(), flat_tiles.size() * sizeof(int)).wait();
            } else {
                if (is_rank_zero()) {
                    std::cout << "Warning: Failed to allocate GPU memory for tile data" << std::endl;
                }
            }
        }
        
        gpu_memory_allocated = true;
        
    } catch (const std::exception& e) {
        if (is_rank_zero()) {
            std::cout << "SYCL GPU memory operation failed: " << e.what() << std::endl;
        }
        // Clean up on failure
        if (d_neighbor_list != nullptr) {
            try {
                sycl::free(d_neighbor_list, sycl_queue);
            } catch (...) {}
            d_neighbor_list = nullptr;
        }
        if (d_tile_data != nullptr) {
            try {
                sycl::free(d_tile_data, sycl_queue);
            } catch (...) {}
            d_tile_data = nullptr;
        }
        gpu_memory_allocated = false;
        return;
    }
    #endif
    
}





// Optimized MPI broadcast with chunked transfer for large neighbor lists
void MortonZOrderNeighborBuilder::optimized_mpi_broadcast(NeighborList& nlist, size_t n, int mpi_rank, int mpi_size) {
    if (mpi_size <= 1) return;
    
    // Broadcast metadata only
    struct NeighborListHeader {
        int total_neighbors;
        Real cutoff_val;
    } header;
    
    if (mpi_rank == 0) {
        header.total_neighbors = static_cast<int>(nlist.neighbors.size());
        header.cutoff_val = nlist.cutoff;
    }
    
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Non-root ranks allocate memory
    if (mpi_rank != 0) {
        nlist.cutoff = header.cutoff_val;
        nlist.neighbors.resize(header.total_neighbors);
        nlist.num_neighbors.assign(n, 0);
        nlist.neighbor_offset.assign(n, 0);
    }
    
    // Send data efficiently
    if (header.total_neighbors > 0) {
        MPI_Bcast(nlist.neighbors.data(), header.total_neighbors, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(nlist.num_neighbors.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(nlist.neighbor_offset.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
    }
}



// Factory function implementation
std::unique_ptr<NeighborListBuilder> create_neighbor_list_builder(
    NeighborListType type, Real cutoff, const std::string& config, int mpi_rank) {
    
    int rank = mpi_rank;
    
    switch (type) {
        case NeighborListType::VERLET_CELL_LIST: {
            Real skin = 1.0; // Default Verlet skin
            
            // Parse config string for skin distance if provided
            if (!config.empty()) {
                try {
                    skin = std::stod(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid skin distance in config, using default 1.0" << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating Verlet-Cell list neighbor builder (skin: " 
                          << skin << ")" << std::endl;
            }
            return std::make_unique<VerletCellListNeighborBuilder>(cutoff, skin);
        }
            
        case NeighborListType::MORTON_Z_ORDER_CSR: {
            Real skin = 2.0; // Default skin for stability
            int tile_size = 256; // Default tile size (optimized for GPU warps)
            
            // Parse config string for skin distance if provided
            if (!config.empty()) {
                try {
                    skin = std::stod(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid skin distance in config, using default 2.0" << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating Morton Z-order CSR neighbor builder (skin: " 
                          << skin << ", tile size: " << tile_size << ")" << std::endl;
            }
            return std::make_unique<MortonZOrderNeighborBuilder>(cutoff, skin, tile_size);
        }
        
        default:
            if (rank == 0) {
                std::cout << "Unknown neighbor list type, defaulting to verlet cell list" << std::endl;
            }
            return std::make_unique<VerletCellListNeighborBuilder>(cutoff, 2.0);
    }
}
