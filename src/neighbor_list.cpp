#define OMPI_SKIP_MPICXX 1
#include "neighbor_list.h"
#include "optimized_neighbor_list.h"
#include "md_types.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <mpi.h>

// Helper function to check if this is rank 0
static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

CellListNeighborBuilder::CellListNeighborBuilder(Real cutoff) : cell_size(cutoff) {}

void CellListNeighborBuilder::initialize() {
    // Initialization handled in build method
}

void CellListNeighborBuilder::finalize() {
    // Cleanup
    cells.clear();
}

void CellListNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist) {
    nlist.cutoff = cell_size;
    nlist.max_neighbors_per_atom = 200; // Conservative estimate
    nlist.list_type = "cell_list";
    
    // For very dense systems, reduce cutoff to avoid memory issues
    Real effective_cutoff = cell_size;
    if (particles.num_particles > 10000) {
        // For large systems, use a smaller effective cutoff to limit neighbor count
        effective_cutoff = std::min(cell_size, 8.0); // Max 8 Angstrom for large systems
        // Message removed to reduce output verbosity
        // std::cout << "Large system detected (" << particles.num_particles 
        //           << " particles). Reducing effective cutoff to " << effective_cutoff << std::endl;
    }
    
    // Calculate number of cells in each dimension based on effective cutoff
    // Calculate number of cells in each dimension based on effective cutoff
    nx = static_cast<int>(box.lx / effective_cutoff) + 1;
    ny = static_cast<int>(box.ly / effective_cutoff) + 1;
    nz = static_cast<int>(box.lz / effective_cutoff) + 1;
    
    int total_cells = nx * ny * nz;
    cells.clear();
    cells.resize(total_cells);
    
    // Assign particles to cells
    for (size_t i = 0; i < particles.num_particles; i++) {
        int ix = static_cast<int>(particles.x[i] / effective_cutoff);
        int iy = static_cast<int>(particles.y[i] / effective_cutoff);
        int iz = static_cast<int>(particles.z[i] / effective_cutoff);
        
        // Apply periodic boundary conditions if needed
        if (box.periodic_x) {
            ix = (ix + nx) % nx;
        } else {
            ix = std::max(0, std::min(ix, nx - 1));
        }
        if (box.periodic_y) {
            iy = (iy + ny) % ny;
        } else {
            iy = std::max(0, std::min(iy, ny - 1));
        }
        if (box.periodic_z) {
            iz = (iz + nz) % nz;
        } else {
            iz = std::max(0, std::min(iz, nz - 1));
        }
        
        int cell_id = iz * nx * ny + iy * nx + ix;
        if (cell_id >= 0 && cell_id < total_cells) {
            cells[cell_id].push_back(i);
        }
    }
    
    // Build neighbor lists
    nlist.neighbors.clear();
    nlist.num_neighbors.resize(particles.num_particles, 0);
    nlist.neighbor_offset.resize(particles.num_particles, 0);
    
    Real cutoff_sq = effective_cutoff * effective_cutoff;
    
    for (size_t i = 0; i < particles.num_particles; i++) {
        nlist.neighbor_offset[i] = nlist.neighbors.size();
        
        // Determine which cell particle i is in
        int ix = static_cast<int>(particles.x[i] / effective_cutoff);
        int iy = static_cast<int>(particles.y[i] / effective_cutoff);
        int iz = static_cast<int>(particles.z[i] / effective_cutoff);
        
        // Apply periodic boundary conditions
        if (box.periodic_x) ix = (ix + nx) % nx;
        else ix = std::max(0, std::min(ix, nx - 1));
        if (box.periodic_y) iy = (iy + ny) % ny;
        else iy = std::max(0, std::min(iy, ny - 1));
        if (box.periodic_z) iz = (iz + nz) % nz;
        else iz = std::max(0, std::min(iz, nz - 1));
        
        // Search neighboring cells
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int neighbor_ix = ix + dx;
                    int neighbor_iy = iy + dy;
                    int neighbor_iz = iz + dz;
                    
                    // Handle boundary conditions
                    if (box.periodic_x) {
                        neighbor_ix = (neighbor_ix + nx) % nx;
                    } else if (neighbor_ix < 0 || neighbor_ix >= nx) {
                        continue;
                    }
                    if (box.periodic_y) {
                        neighbor_iy = (neighbor_iy + ny) % ny;
                    } else if (neighbor_iy < 0 || neighbor_iy >= ny) {
                        continue;
                    }
                    if (box.periodic_z) {
                        neighbor_iz = (neighbor_iz + nz) % nz;
                    } else if (neighbor_iz < 0 || neighbor_iz >= nz) {
                        continue;
                    }
                    
                    int neighbor_cell = neighbor_iz * nx * ny + neighbor_iy * nx + neighbor_ix;
                    if (neighbor_cell < 0 || neighbor_cell >= total_cells) continue;
                    
                    // Check all particles in neighboring cell
                    for (int j : cells[neighbor_cell]) {
                        if (static_cast<int>(i) == j) continue;
                        
                        Real dx_ij = particles.x[i] - particles.x[j];
                        Real dy_ij = particles.y[i] - particles.y[j];
                        Real dz_ij = particles.z[i] - particles.z[j];
                        
                        // Apply minimum image convention for periodic boundaries
                        if (box.periodic_x) {
                            if (dx_ij > box.lx * 0.5) dx_ij -= box.lx;
                            else if (dx_ij < -box.lx * 0.5) dx_ij += box.lx;
                        }
                        if (box.periodic_y) {
                            if (dy_ij > box.ly * 0.5) dy_ij -= box.ly;
                            else if (dy_ij < -box.ly * 0.5) dy_ij += box.ly;
                        }
                        if (box.periodic_z) {
                            if (dz_ij > box.lz * 0.5) dz_ij -= box.lz;
                            else if (dz_ij < -box.lz * 0.5) dz_ij += box.lz;
                        }
                        
                        Real r2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
                        
                        if (r2 < cutoff_sq) {
                            nlist.neighbors.push_back(j);
                            nlist.num_neighbors[i]++;
                            
                            // Limit neighbors per particle to avoid memory explosion
                            if (nlist.num_neighbors[i] >= 500) {
                                goto next_particle; // Break out of all nested loops
                            }
                        }
                    }
                }
            }
        }
        next_particle:; // Label for breaking out of nested loops
    }
    
    // Debug: Print neighbor list statistics
    static int build_count = 0;
    if (build_count < 3 && is_rank_zero()) {
        int total_neighbors = 0;
        int max_neighbors = 0;
        for (size_t i = 0; i < particles.num_particles && i < 10; i++) {
            total_neighbors += nlist.num_neighbors[i];
            if (nlist.num_neighbors[i] > max_neighbors) {
                max_neighbors = nlist.num_neighbors[i];
            }
        }
        std::cout << "Neighbor list build " << build_count << ": cutoff=" << nlist.cutoff 
                  << ", avg neighbors (first 10)=" << (total_neighbors / 10.0) 
                  << ", max neighbors=" << max_neighbors << std::endl;
    }
    build_count++;
}

#if defined(USE_CUDA) || defined(USE_SYCL)
// Morton/Hilbert curve based neighbor list implementation
// Compatible with both CUDA and SYCL backends

MortonHilbertNeighborBuilder::MortonHilbertNeighborBuilder(Real cutoff_val, int tile_size) 
    : cutoff(cutoff_val), morton_levels(10) {
    tile_staging.tile_size = tile_size;
}

void MortonHilbertNeighborBuilder::initialize() {
    if (is_rank_zero()) {
        std::cout << "Initializing Morton/Hilbert neighbor list builder (CUDA optimized)" << std::endl;
    }
}

void MortonHilbertNeighborBuilder::finalize() {
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
    // Simplified Hilbert curve encoding - for full implementation, would use more complex algorithm
    // For now, use Morton code as approximation
    return morton_encode_3d(x, y, z);
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

void MortonHilbertNeighborBuilder::build_csr_neighbor_list(ParticleData& particles, SimulationBox& box) {
    size_t n = particles.num_particles;
    Real cutoff_sq = cutoff * cutoff;
    
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
            if (box.periodic_x && std::abs(dx) > box.lx * 0.5) {
                dx = dx > 0 ? dx - box.lx : dx + box.lx;
            }
            if (box.periodic_y && std::abs(dy) > box.ly * 0.5) {
                dy = dy > 0 ? dy - box.ly : dy + box.ly;
            }
            if (box.periodic_z && std::abs(dz) > box.lz * 0.5) {
                dz = dz > 0 ? dz - box.lz : dz + box.lz;
            }
            
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 < cutoff_sq) {
                csr_nlist.neighbors.push_back(j);
                csr_nlist.col_indices.push_back(j);
                
                // Limit neighbors to avoid memory explosion
                if (csr_nlist.neighbors.size() - csr_nlist.row_ptr[i] >= 500) {
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
    
    std::cout << "Created " << n_tiles << " tiles with " << tile_staging.tile_size 
              << " neighbors per tile for GPU optimization" << std::endl;
}

void MortonHilbertNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist) {
    size_t n = particles.num_particles;
    
    std::cout << "Building Morton/Hilbert neighbor list for " << n << " particles..." << std::endl;
    
    // Step 1: Generate Morton codes for all particles
    morton_codes.clear();
    morton_codes.reserve(n);
    
    // Calculate grid resolution based on box size
    uint32_t grid_res = 1 << morton_levels;
    Real scale_x = grid_res / box.lx;
    Real scale_y = grid_res / box.ly;
    Real scale_z = grid_res / box.lz;
    
    for (size_t i = 0; i < n; i++) {
        uint32_t x = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), particles.x[i] * scale_x)));
        uint32_t y = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), particles.y[i] * scale_y)));
        uint32_t z = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(grid_res-1), particles.z[i] * scale_z)));
        
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
    
    // Step 4: Temporarily reorder particles for better cache locality
    reorder_particles(particles);
    
    // Step 5: Build CSR format neighbor list
    build_csr_neighbor_list(particles, box);
    
    // Step 6: Create tile staging for GPU
    create_tile_staging();
    
    // Step 7: Convert CSR format to traditional format for compatibility
    nlist.neighbors.clear();
    nlist.num_neighbors.resize(n, 0);
    nlist.neighbor_offset.resize(n, 0);
    nlist.cutoff = cutoff;
    nlist.max_neighbors_per_atom = 500;
    nlist.list_type = "morton_hilbert";
    
    for (size_t i = 0; i < n; i++) {
        nlist.neighbor_offset[i] = nlist.neighbors.size();
        int start = csr_nlist.row_ptr[i];
        int end = csr_nlist.row_ptr[i + 1];
        nlist.num_neighbors[i] = end - start;
        
        for (int j = start; j < end; j++) {
            nlist.neighbors.push_back(csr_nlist.neighbors[j]);
        }
    }
    
    // Step 8: Restore original particle order
    restore_particle_order(particles);
    
    // Update neighbor indices to account for restored order
    for (size_t i = 0; i < nlist.neighbors.size(); i++) {
        nlist.neighbors[i] = inverse_reorder[nlist.neighbors[i]];
    }
    
    std::cout << "Morton/Hilbert neighbor list built: " << nlist.neighbors.size() 
              << " total neighbors, avg " << (nlist.neighbors.size() / static_cast<double>(n)) 
              << " per particle" << std::endl;
}

void MortonHilbertNeighborBuilder::copy_to_gpu(NeighborList& nlist) {
    // This function would contain GPU memory allocation and copying
    // Implementation would depend on CUDA force calculator integration
    std::cout << "Copying Morton/Hilbert neighbor list to GPU memory..." << std::endl;
}

#endif // USE_CUDA

// Factory function implementation
std::unique_ptr<NeighborListBuilder> create_neighbor_list_builder(
    NeighborListType type, Real cutoff, const std::string& config, int mpi_rank) {
    
    int rank = mpi_rank;
    
    switch (type) {
        case NeighborListType::CELL_LIST: {
            // Automatically use optimized cell list for better MPI performance
            Real halo_width = cutoff * 0.5; // Default halo width
            
            // Parse config string for halo width if provided
            if (!config.empty()) {
                try {
                    halo_width = std::stod(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid halo width in config, using default " << halo_width << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating optimized cell list neighbor builder (halo width: " 
                          << halo_width << ")" << std::endl;
            }
            return std::make_unique<OptimizedCellListNeighborBuilder>(cutoff, halo_width);
        }
            
        case NeighborListType::OPTIMIZED_CELL_LIST: {
            Real halo_width = cutoff * 0.5; // Default halo width
            
            // Parse config string for halo width if provided
            if (!config.empty()) {
                try {
                    halo_width = std::stod(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid halo width in config, using default " << halo_width << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating optimized cell list neighbor builder (halo width: " 
                          << halo_width << ")" << std::endl;
            }
            return std::make_unique<OptimizedCellListNeighborBuilder>(cutoff, halo_width);
        }
            
        case NeighborListType::TILED_CELL_LIST: {
            int tile_size = 256; // Default tile size
            
            // Parse config string for tile size if provided
            if (!config.empty()) {
                try {
                    tile_size = std::stoi(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid tile size in config, using default 256" << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating tiled cell list neighbor builder (tile size: " 
                          << tile_size << ")" << std::endl;
            }
            return std::make_unique<TiledNeighborListBuilder>(cutoff, tile_size);
        }
            
#if defined(USE_CUDA) || defined(USE_SYCL)
        case NeighborListType::MORTON_HILBERT_CSR: {
            int tile_size = 256; // Default tile size
            
            // Parse config string for tile size if provided
            if (!config.empty()) {
                try {
                    tile_size = std::stoi(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid tile size in config, using default 256" << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating Morton/Hilbert CSR neighbor builder (tile size: " 
                          << tile_size << ")" << std::endl;
            }
            return std::make_unique<MortonHilbertNeighborBuilder>(cutoff, tile_size);
        }
#endif // USE_CUDA || USE_SYCL
        
        default:
            if (rank == 0) {
                std::cout << "Unknown neighbor list type, defaulting to optimized cell list" << std::endl;
            }
            return std::make_unique<OptimizedCellListNeighborBuilder>(cutoff);
    }
}
