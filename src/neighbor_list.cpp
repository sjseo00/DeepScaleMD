
#include "neighbor_list.h"
#include "md_types.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <mpi.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif
// External debug level variable
extern int g_debug_level;
inline bool debug_level(int level) { return g_debug_level >= level; }

static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

// VerletCellListNeighborBuilder implementation
VerletCellListNeighborBuilder::VerletCellListNeighborBuilder(Real cutoff_val, Real skin) 
    : cutoff(cutoff_val), verlet_skin(skin) {
    cell_size = cutoff + verlet_skin;  // Larger cell size for Verlet skin
}

void VerletCellListNeighborBuilder::initialize() {
    // Initialization handled in build method
}

void VerletCellListNeighborBuilder::finalize() {
    cells.clear();
}

void VerletCellListNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local) {
    // MPI setup and data synchronization
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    int mpi_rank = 0, mpi_size = 1;
    
    // Only build neighbor lists for local particles
    // If num_local == 0, fall back to all particles (serial mode)
    size_t end_particle = (num_local > 0) ? num_local : particles.num_particles;
    
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        if (mpi_size > 1) {
            // Ensure all ranks have consistent particle data
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Broadcast particle count from rank 0
            size_t N = particles.num_particles;
            MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            
            // Resize arrays on non-root ranks if needed
            if (mpi_rank != 0) {
                particles.x.resize(N);
                particles.y.resize(N);
                particles.z.resize(N);
                particles.num_particles = N;
            }
            
            // Broadcast particle coordinates from rank 0
            if (N > 0) {
                MPI_Bcast(particles.x.data(), static_cast<int>(N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(particles.y.data(), static_cast<int>(N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(particles.z.data(), static_cast<int>(N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            
            // Broadcast simulation box parameters
            MPI_Bcast(&box.lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&box.ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&box.lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&box.periodic_x, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            MPI_Bcast(&box.periodic_y, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            MPI_Bcast(&box.periodic_z, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    // Extended physical cutoff used for search; physical pairwise force uses 'cutoff'
    const Real extended_cutoff = cutoff + verlet_skin;
    nlist.cutoff = extended_cutoff;
    nlist.list_type = "verlet_cell";
    const Real extended_cutoff_sq = extended_cutoff * extended_cutoff;

    // Enforce cell bin length >= physical cutoff (not extended) to retain 3x3x3 stencil validity
    const Real bin_length_target = std::max(cutoff, cell_size); // cell_size was preset to cutoff+skin in ctor
    // Choose number of cells via floor; guarantee at least 1 per dimension
    nx = static_cast<int>(std::floor(box.lx / bin_length_target)); if (nx < 1) nx = 1;
    ny = static_cast<int>(std::floor(box.ly / bin_length_target)); if (ny < 1) ny = 1;
    nz = static_cast<int>(std::floor(box.lz / bin_length_target)); if (nz < 1) nz = 1;

    const Real eff_lx = box.lx / nx;
    const Real eff_ly = box.ly / ny;
    const Real eff_lz = box.lz / nz;

    const int total_cells = nx * ny * nz;
    cells.assign(total_cells, {});

    // Assign particles to cells (periodic-aware indexing with robust modulo)
    // Use more robust cell index calculation to ensure consistency across MPI ranks
    for (size_t i = 0; i < particles.num_particles; ++i) {
        // Use floor to ensure consistent cell assignment across different systems
        int ix = static_cast<int>(std::floor(particles.x[i] / eff_lx));
        int iy = static_cast<int>(std::floor(particles.y[i] / eff_ly));
        int iz = static_cast<int>(std::floor(particles.z[i] / eff_lz));
        
        // Robust periodic boundary condition handling
        if (box.periodic_x) {
            ix = ((ix % nx) + nx) % nx;
        } else {
            ix = std::max(0, std::min(ix, nx - 1));
        }
        if (box.periodic_y) {
            iy = ((iy % ny) + ny) % ny;
        } else {
            iy = std::max(0, std::min(iy, ny - 1));
        }
        if (box.periodic_z) {
            iz = ((iz % nz) + nz) % nz;
        } else {
            iz = std::max(0, std::min(iz, nz - 1));
        }
        
        int cell_id = iz * nx * ny + iy * nx + ix;
        cells[cell_id].push_back(static_cast<int>(i));
    }

    // Initialize neighbor list arrays
    nlist.neighbors.clear();
    nlist.num_neighbors.assign(particles.num_particles, 0);
    nlist.neighbor_offset.assign(particles.num_particles, 0);

    // Only rank 0 computes the neighbor list to ensure consistency across MPI ranks
    if (mpi_rank == 0) {
        // Optional: track duplicates (we create half list only: j added only when j>i to reduce size)
        for (size_t i = 0; i < end_particle; ++i) {  // Only build for local particles
        nlist.neighbor_offset[i] = static_cast<int>(nlist.neighbors.size());

        // Use consistent cell index calculation (same as in cell assignment)
        int ix = static_cast<int>(std::floor(particles.x[i] / eff_lx));
        int iy = static_cast<int>(std::floor(particles.y[i] / eff_ly));
        int iz = static_cast<int>(std::floor(particles.z[i] / eff_lz));
        
        if (box.periodic_x) {
            ix = ((ix % nx) + nx) % nx;
        } else {
            ix = std::max(0, std::min(ix, nx - 1));
        }
        if (box.periodic_y) {
            iy = ((iy % ny) + ny) % ny;
        } else {
            iy = std::max(0, std::min(iy, ny - 1));
        }
        if (box.periodic_z) {
            iz = ((iz % nz) + nz) % nz;
        } else {
            iz = std::max(0, std::min(iz, nz - 1));
        }

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nix = ix + dx;
                    int niy = iy + dy;
                    int niz = iz + dz;
                    if (box.periodic_x) {
                        nix = ((nix % nx) + nx) % nx;
                    } else if (nix < 0 || nix >= nx) {
                        continue;
                    }
                    if (box.periodic_y) {
                        niy = ((niy % ny) + ny) % ny;
                    } else if (niy < 0 || niy >= ny) {
                        continue;
                    }
                    if (box.periodic_z) {
                        niz = ((niz % nz) + nz) % nz;
                    } else if (niz < 0 || niz >= nz) {
                        continue;
                    }
                    int ncell = niz * nx * ny + niy * nx + nix;
                    for (int j : cells[ncell]) {
                        // Half-list: only keep j>i
                        if (j <= static_cast<int>(i)) continue;
                        Real dx_ij = particles.x[i] - particles.x[j];
                        Real dy_ij = particles.y[i] - particles.y[j];
                        Real dz_ij = particles.z[i] - particles.z[j];
                        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
                        Real r2 = dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij;
                        if (r2 < extended_cutoff_sq) {
                            nlist.neighbors.push_back(j);
                            ++nlist.num_neighbors[i];
                            if (nlist.max_neighbors_per_atom <= 0) nlist.max_neighbors_per_atom = 500;
                            if (nlist.num_neighbors[i] >= nlist.max_neighbors_per_atom) goto next_particle_verlet_fixed;
                        }
                    }
                }
            }
            }
            next_particle_verlet_fixed:;
        }
    } // End of rank 0 only neighbor list computation
    
    // --- MPI synchronization: Broadcast neighbor list from rank 0 to all ranks ---
    if (mpi_initialized && mpi_size > 1) {
        // Broadcast neighbor list data from rank 0 to all other ranks
        Real cutoff_val = nlist.cutoff;
        MPI_Bcast(&cutoff_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        nlist.cutoff = cutoff_val;
        
        int max_cap = nlist.max_neighbors_per_atom;
        MPI_Bcast(&max_cap, 1, MPI_INT, 0, MPI_COMM_WORLD);
        nlist.max_neighbors_per_atom = max_cap;
        
        int N = static_cast<int>(particles.num_particles);
        
        // Prepare arrays on non-root ranks (rank 0 already has computed values)
        if (mpi_rank != 0) {
            nlist.num_neighbors.assign(N, 0);
            nlist.neighbor_offset.assign(N, 0);
        }
        
        // Broadcast neighbor counts and offsets
        if (N > 0) {
            MPI_Bcast(nlist.num_neighbors.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(nlist.neighbor_offset.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Broadcast total neighbor count and neighbor list
        int total_nbrs = static_cast<int>(nlist.neighbors.size());
        MPI_Bcast(&total_nbrs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (mpi_rank != 0) {
            nlist.neighbors.resize(total_nbrs);
        }
        
        if (total_nbrs > 0) {
            MPI_Bcast(nlist.neighbors.data(), total_nbrs, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Final synchronization barrier
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Morton/Hilbert curve based neighbor list implementation
// Compatible with CUDA, SYCL, and CPU backends

MortonHilbertNeighborBuilder::MortonHilbertNeighborBuilder(Real cutoff_val, Real skin, int tile_size) 
    : cutoff(cutoff_val), verlet_skin(skin), morton_levels(10) {
    tile_staging.tile_size = tile_size;
    // Initialize GPU memory pointers to nullptr
    d_neighbor_list = nullptr;
    d_tile_data = nullptr;
    gpu_memory_allocated = false;
    gpu_available = false;
    gpu_device_id = 0;
}

void MortonHilbertNeighborBuilder::initialize() {   
    // Initialize GPU context if available
    #ifdef USE_CUDA
    cudaError_t err = cudaGetDevice(&gpu_device_id);
    if (err == cudaSuccess) {
        gpu_available = true;
        if (is_rank_zero()) {
            std::cout << "CUDA device " << gpu_device_id << " detected and ready" << std::endl;
        }
    } else {
        gpu_available = false;
        if (is_rank_zero()) {
            std::cout << "Warning: CUDA not available, using CPU fallback" << std::endl;
        }
    }
    #elif defined(USE_SYCL)
    try {
        sycl_queue = sycl::queue(sycl::gpu_selector_v);
        gpu_available = true;
        if (is_rank_zero()) {
            std::cout << "SYCL GPU queue created successfully" << std::endl;
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

void MortonHilbertNeighborBuilder::finalize() {
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

uint64_t MortonHilbertNeighborBuilder::morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    // Interleave bits of x, y, z to create Morton code
    auto split_by_3 = [](uint32_t a) -> uint64_t {
        uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    };
    
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2);
}

void MortonHilbertNeighborBuilder::morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) {
    auto compact_by_3 = [](uint64_t x) -> uint32_t {
        x = x & 0x1249249249249249;
        x = (x | (x >> 2)) & 0x10c30c30c30c30c3;
        x = (x | (x >> 4)) & 0x100f00f00f00f00f;
        x = (x | (x >> 8)) & 0x1f0000ff0000ff;
        x = (x | (x >> 16)) & 0x1f00000000ffff;
        x = (x | (x >> 32)) & 0x1fffff;
        return static_cast<uint32_t>(x);
    };
    
    x = compact_by_3(morton);
    y = compact_by_3(morton >> 1);
    z = compact_by_3(morton >> 2);
}

uint64_t MortonHilbertNeighborBuilder::hilbert_encode_3d(uint32_t x, uint32_t y, uint32_t z, int level) {
    // Simplified Hilbert curve encoding - more sophisticated than Morton
    // This is a basic implementation; for production use, consider a full Hilbert curve library
    
    uint64_t h = 0;
    uint32_t s = 1 << (level - 1);
    
    // Rotate and fold coordinates based on Hilbert curve patterns
    for (int i = level - 1; i >= 0; i--) {
        uint32_t xi = (x >> i) & 1;
        uint32_t yi = (y >> i) & 1;
        uint32_t zi = (z >> i) & 1;
        
        // Calculate Hilbert index for this level
        uint32_t h_level = 0;
        
        // 3D Hilbert curve state transitions (simplified)
        if (zi == 0) {
            if (yi == 0) {
                h_level = xi;
                if (xi == 1) {
                    // Rotate coordinates
                    std::swap(x, y);
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
            } else {
                h_level = 2 + (1 - xi);
            }
        } else {
            if (yi == 0) {
                h_level = 4 + (1 - xi);
            } else {
                h_level = 6 + xi;
                if (xi == 0) {
                    // Rotate coordinates
                    std::swap(x, z);
                    x = s - 1 - x;
                    z = s - 1 - z;
                }
            }
        }
        
        h = (h << 3) | h_level;
        s >>= 1;
    }
    
    return h;
}

void MortonHilbertNeighborBuilder::reorder_particles(ParticleData& particles) {
    size_t n = particles.num_particles;
    
    // Create backup of original data
    std::vector<Real> orig_x = particles.x, orig_y = particles.y, orig_z = particles.z;
    std::vector<Real> orig_vx = particles.vx, orig_vy = particles.vy, orig_vz = particles.vz;
    std::vector<Real> orig_mass = particles.mass, orig_charge = particles.charge;
    std::vector<int> orig_type = particles.type;
    
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
    }
}

void MortonHilbertNeighborBuilder::restore_particle_order(ParticleData& particles) {
    size_t n = particles.num_particles;
    
    // Create backup of reordered data
    std::vector<Real> reord_x = particles.x, reord_y = particles.y, reord_z = particles.z;
    std::vector<Real> reord_vx = particles.vx, reord_vy = particles.vy, reord_vz = particles.vz;
    std::vector<Real> reord_mass = particles.mass, reord_charge = particles.charge;
    std::vector<int> reord_type = particles.type;
    
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
    }
}

void MortonHilbertNeighborBuilder::build_csr_neighbor_list(ParticleData& particles, SimulationBox& box, int max_cap) {
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
                if (csr_nlist.neighbors.size() - csr_nlist.row_ptr[i] >= max_cap) {
                    break;
                }
            }
        }
    }
    
    csr_nlist.row_ptr[n] = csr_nlist.neighbors.size();
}

void MortonHilbertNeighborBuilder::create_tile_staging() {
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

void MortonHilbertNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local) {
    // MPI setup
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    int mpi_rank = 0, mpi_size = 1;
    
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }
    
    size_t n = (num_local > 0) ? num_local : particles.num_particles;
    
    // Initialize neighbor list structure early
    nlist.neighbors.clear();
    nlist.num_neighbors.assign(n, 0);
    nlist.neighbor_offset.assign(n, 0);
    nlist.cutoff = cutoff + verlet_skin;
    if (nlist.max_neighbors_per_atom <= 0) nlist.max_neighbors_per_atom = 500;
    nlist.list_type = "morton_hilbert";
    
    // Only rank 0 computes the neighbor list to avoid MPI consistency issues
    if (mpi_rank == 0) {
        // Step 1: Generate Morton codes for all particles
        morton_codes.clear();
        morton_codes.reserve(n);
        
        // Calculate grid resolution based on box size
        uint32_t grid_res = 1 << morton_levels;
        Real scale_x = grid_res / box.lx;
        Real scale_y = grid_res / box.ly;
        Real scale_z = grid_res / box.lz;
        
        for (size_t i = 0; i < n; i++) {
            // Handle periodic boundary conditions properly
            Real norm_x = particles.x[i];
            Real norm_y = particles.y[i];
            Real norm_z = particles.z[i];
            
            // Normalize coordinates to [0, 1) with periodic boundaries
            if (box.periodic_x) {
                norm_x = norm_x - std::floor(norm_x / box.lx) * box.lx;
            }
            if (box.periodic_y) {
                norm_y = norm_y - std::floor(norm_y / box.ly) * box.ly;
            }
            if (box.periodic_z) {
                norm_z = norm_z - std::floor(norm_z / box.lz) * box.lz;
            }
            
            uint32_t x = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), norm_x * scale_x)));
            uint32_t y = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), norm_y * scale_y)));
            uint32_t z = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), norm_z * scale_z)));
            
            MortonCode mc;
            mc.code = morton_encode_3d(x, y, z);
            mc.particle_id = i;
            morton_codes.push_back(mc);
        }
        
        // Step 2: Sort particles by Morton code
        std::sort(morton_codes.begin(), morton_codes.end());
        
        // Step 3: Create reordering maps
        particle_reorder.resize(n);
        inverse_reorder.resize(n);
        for (size_t i = 0; i < n; i++) {
            particle_reorder[i] = morton_codes[i].particle_id;
            inverse_reorder[morton_codes[i].particle_id] = i;
        }
        
        // Step 4: Build neighbor list using Morton-sorted order for spatial locality
        Real extended_cutoff = cutoff + verlet_skin;
        Real cutoff_sq = extended_cutoff * extended_cutoff;
        
        for (size_t i = 0; i < n; i++) {
            int orig_i = morton_codes[i].particle_id;
            nlist.neighbor_offset[orig_i] = nlist.neighbors.size();
            
            // Search neighbors in Morton-sorted order for better cache performance
            for (size_t j = i + 1; j < n; j++) {
                int orig_j = morton_codes[j].particle_id;
                
                Real dx = particles.x[orig_i] - particles.x[orig_j];
                Real dy = particles.y[orig_i] - particles.y[orig_j];
                Real dz = particles.z[orig_i] - particles.z[orig_j];
                
                // Apply minimum image convention
                box.apply_minimum_image(dx, dy, dz);
                
                Real r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 < cutoff_sq) {
                    nlist.neighbors.push_back(orig_j);
                    nlist.num_neighbors[orig_i]++;
                    
                    // Limit neighbors per particle
                    if (nlist.num_neighbors[orig_i] >= nlist.max_neighbors_per_atom) {
                        break;
                    }
                }
            }
        }
    } // End of rank 0 computation
    
    // --- MPI synchronization: Broadcast neighbor list from rank 0 ---
    if (mpi_initialized && mpi_size > 1) {
        // Broadcast metadata
        Real cutoff_val = nlist.cutoff;
        MPI_Bcast(&cutoff_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        nlist.cutoff = cutoff_val;
        
        int max_cap = nlist.max_neighbors_per_atom;
        MPI_Bcast(&max_cap, 1, MPI_INT, 0, MPI_COMM_WORLD);
        nlist.max_neighbors_per_atom = max_cap;
        
        int N = static_cast<int>(n);
        
        // Prepare arrays on non-root ranks
        if (mpi_rank != 0) {
            nlist.num_neighbors.assign(N, 0);
            nlist.neighbor_offset.assign(N, 0);
        }
        
        // Broadcast per-particle data
        if (N > 0) {
            MPI_Bcast(nlist.num_neighbors.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(nlist.neighbor_offset.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Broadcast neighbor list
        int total_nbrs = (mpi_rank == 0) ? static_cast<int>(nlist.neighbors.size()) : 0;
        MPI_Bcast(&total_nbrs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (mpi_rank != 0) {
            nlist.neighbors.resize(total_nbrs);
        }
        
        if (total_nbrs > 0) {
            MPI_Bcast(nlist.neighbors.data(), total_nbrs, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
}

void MortonHilbertNeighborBuilder::copy_to_gpu(NeighborList& nlist) {
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
    
    // Validate input sizes
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
        
        // Validate input sizes
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
    
    if (is_rank_zero()) {
        std::cout << "Successfully copied neighbor list to GPU memory" << std::endl;
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
            
        case NeighborListType::MORTON_HILBERT_CSR: {
            Real skin = 2.0; // Default skin for stability
            int tile_size = 256; // Default tile size
            
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
                std::cout << "Creating Morton/Hilbert CSR neighbor builder (skin: " 
                          << skin << ", tile size: " << tile_size << ")" << std::endl;
            }
            return std::make_unique<MortonHilbertNeighborBuilder>(cutoff, skin, tile_size);
        }
        
        default:
            if (rank == 0) {
                std::cout << "Unknown neighbor list type, defaulting to verlet cell list" << std::endl;
            }
            return std::make_unique<VerletCellListNeighborBuilder>(cutoff, 2.0);
    }
}
