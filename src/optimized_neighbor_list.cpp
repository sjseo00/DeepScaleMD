#define OMPI_SKIP_MPICXX 1
#include "optimized_neighbor_list.h"
#include "md_types.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <chrono>

// External debug level variable
extern int g_debug_level;
inline bool debug_level(int level) { return g_debug_level >= level; }

// Helper function to check if this is rank 0
static bool is_rank_zero() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

// =============================================================================
// OptimizedCellListNeighborBuilder Implementation
// =============================================================================

OptimizedCellListNeighborBuilder::OptimizedCellListNeighborBuilder(Real cutoff, Real halo_width)
    : cell_size(cutoff), halo_width(halo_width > 0 ? halo_width : cutoff * 0.5), 
      nx(0), ny(0), nz(0), rank(0), size(1), 
      rank_x(0), rank_y(0), rank_z(0), dims_x(1), dims_y(1), dims_z(1) {
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
}

void OptimizedCellListNeighborBuilder::initialize() {
    if (is_rank_zero() && debug_level(1)) {
        std::cout << "Initializing optimized cell list neighbor builder" << std::endl;
        if (debug_level(2)) {
            std::cout << "Cutoff: " << cell_size << ", Halo width: " << halo_width << std::endl;
        }
    }
}

void OptimizedCellListNeighborBuilder::finalize() {
    cells.clear();
    interior_particles.clear();
    boundary_particles.clear();
    is_boundary.clear();
}

void OptimizedCellListNeighborBuilder::calculate_3d_decomposition() {
    // Find optimal 3D factorization of the number of processes
    // Minimize surface area to reduce communication overhead
    
    dims_x = dims_y = dims_z = 1;
    
    if (size == 1) {
        rank_x = rank_y = rank_z = 0;
        return;
    }
    
    // Simple factorization - can be improved with better algorithms
    int remaining = size;
    
    // Try to make dimensions as equal as possible
    for (int factor = 2; factor <= remaining && remaining > 1; factor++) {
        while (remaining % factor == 0) {
            if (dims_x <= dims_y && dims_x <= dims_z) {
                dims_x *= factor;
            } else if (dims_y <= dims_z) {
                dims_y *= factor;
            } else {
                dims_z *= factor;
            }
            remaining /= factor;
        }
    }
    
    // Handle case where size is not perfectly factorizable
    if (remaining > 1) {
        dims_x *= remaining;
    }
    
    // Calculate rank coordinates from linear rank
    rank_z = rank / (dims_x * dims_y);
    rank_y = (rank % (dims_x * dims_y)) / dims_x;
    rank_x = rank % dims_x;

}

void OptimizedCellListNeighborBuilder::setup_domain_boundaries(SimulationBox& box) {
    // Calculate optimal 3D decomposition for the given number of processes
    calculate_3d_decomposition();
    
    // Calculate domain boundaries for each dimension
    Real domain_width_x = box.lx / dims_x;
    Real domain_width_y = box.ly / dims_y; 
    Real domain_width_z = box.lz / dims_z;
    
    domain_min_x = rank_x * domain_width_x;
    domain_max_x = (rank_x + 1) * domain_width_x;
    
    domain_min_y = rank_y * domain_width_y;
    domain_max_y = (rank_y + 1) * domain_width_y;
    
    domain_min_z = rank_z * domain_width_z;
    domain_max_z = (rank_z + 1) * domain_width_z;
    
}

void OptimizedCellListNeighborBuilder::classify_particles(ParticleData& particles, SimulationBox& box) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    setup_domain_boundaries(box);
    
    interior_particles.clear();
    boundary_particles.clear();
    is_boundary.clear();
    is_boundary.resize(particles.num_particles, false);
    
    for (size_t i = 0; i < particles.num_particles; i++) {
        bool particle_is_boundary_flag = false;
        
        // Check if particle is near domain boundaries
        if (size > 1) {  // Only classify if using MPI
            // Check x-boundary
            if ((particles.x[i] - domain_min_x) < halo_width || 
                (domain_max_x - particles.x[i]) < halo_width) {
                particle_is_boundary_flag = true;
            }
            
            // Check y-boundary
            if ((particles.y[i] - domain_min_y) < halo_width || 
                (domain_max_y - particles.y[i]) < halo_width) {
                particle_is_boundary_flag = true;
            }
            
            // Check z-boundary  
            if ((particles.z[i] - domain_min_z) < halo_width || 
                (domain_max_z - particles.z[i]) < halo_width) {
                particle_is_boundary_flag = true;
            }
            
            // For periodic boundaries, also check global boundaries
            if (box.periodic_x) {
                if (particles.x[i] < halo_width || particles.x[i] > (box.lx - halo_width)) {
                    particle_is_boundary_flag = true;
                }
            }
            if (box.periodic_y) {
                if (particles.y[i] < halo_width || particles.y[i] > (box.ly - halo_width)) {
                    particle_is_boundary_flag = true;
                }
            }
            if (box.periodic_z) {
                if (particles.z[i] < halo_width || particles.z[i] > (box.lz - halo_width)) {
                    particle_is_boundary_flag = true;
                }
            }
        }
        
        is_boundary[i] = particle_is_boundary_flag;
        
        if (particle_is_boundary_flag) {
            boundary_particles.push_back(i);
        } else {
            interior_particles.push_back(i);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
}

void OptimizedCellListNeighborBuilder::build_interior_neighbors(ParticleData& particles, SimulationBox& box, NeighborList& nlist) {
    // Build neighbor list only for interior particles
    // These particles don't need halo exchange data
    
    Real cutoff_sq = cell_size * cell_size;
    
    for (int i : interior_particles) {
        nlist.neighbor_offset[i] = nlist.neighbors.size();
        
        // Find cell for this particle
        int ix = static_cast<int>(particles.x[i] / cell_size);
        int iy = static_cast<int>(particles.y[i] / cell_size);
        int iz = static_cast<int>(particles.z[i] / cell_size);
        
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
                    if (neighbor_cell < 0 || neighbor_cell >= static_cast<int>(cells.size())) continue;
                    
                    // Check all particles in neighboring cell
                    for (int j : cells[neighbor_cell]) {
                        if (i == j) continue;
                        
                        Real dx_ij = particles.x[i] - particles.x[j];
                        Real dy_ij = particles.y[i] - particles.y[j];
                        Real dz_ij = particles.z[i] - particles.z[j];
                        
                        // Apply minimum image convention
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
                            
                            // Limit neighbors to avoid memory issues
                            if (nlist.num_neighbors[i] >= 300) {
                                break;
                            }
                        }
                    }
                    
                    if (nlist.num_neighbors[i] >= 300) break;
                }
                if (nlist.num_neighbors[i] >= 300) break;
            }
            if (nlist.num_neighbors[i] >= 300) break;
        }
    }
}

void OptimizedCellListNeighborBuilder::build_boundary_neighbors(ParticleData& particles, SimulationBox& box, NeighborList& nlist) {
    // Build neighbor list for boundary particles
    // These particles may require halo data from other processes
    
    Real cutoff_sq = cell_size * cell_size;
    
    for (int i : boundary_particles) {
        nlist.neighbor_offset[i] = nlist.neighbors.size();
        
        // Find cell for this particle
        int ix = static_cast<int>(particles.x[i] / cell_size);
        int iy = static_cast<int>(particles.y[i] / cell_size);
        int iz = static_cast<int>(particles.z[i] / cell_size);
        
        // Apply periodic boundary conditions
        if (box.periodic_x) ix = (ix + nx) % nx;
        else ix = std::max(0, std::min(ix, nx - 1));
        if (box.periodic_y) iy = (iy + ny) % ny;
        else iy = std::max(0, std::min(iy, ny - 1));
        if (box.periodic_z) iz = (iz + nz) % nz;
        else iz = std::max(0, std::min(iz, nz - 1));
        
        // Search neighboring cells (including halo cells if available)
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
                    if (neighbor_cell < 0 || neighbor_cell >= static_cast<int>(cells.size())) continue;
                    
                    // Check all particles in neighboring cell
                    for (int j : cells[neighbor_cell]) {
                        if (i == j) continue;
                        
                        Real dx_ij = particles.x[i] - particles.x[j];
                        Real dy_ij = particles.y[i] - particles.y[j];
                        Real dz_ij = particles.z[i] - particles.z[j];
                        
                        // Apply minimum image convention
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
                            
                            // Limit neighbors to avoid memory issues
                            if (nlist.num_neighbors[i] >= 400) {
                                break;
                            }
                        }
                    }
                    
                    if (nlist.num_neighbors[i] >= 400) break;
                }
                if (nlist.num_neighbors[i] >= 400) break;
            }
            if (nlist.num_neighbors[i] >= 400) break;
        }
    }
}

void OptimizedCellListNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    nlist.cutoff = cell_size;
    nlist.max_neighbors_per_atom = 350; // Increased limit for better accuracy
    nlist.list_type = "optimized_cell_list";
    
    // Step 1: Classify particles into interior/boundary
    classify_particles(particles, box);
    
    // Step 2: Set up cell grid
    Real effective_cutoff = cell_size;
    if (particles.num_particles > 10000) {
        effective_cutoff = std::min(cell_size, 10.0); // Slightly larger for better accuracy
    }
    
    nx = static_cast<int>(box.lx / effective_cutoff) + 1;
    ny = static_cast<int>(box.ly / effective_cutoff) + 1;
    nz = static_cast<int>(box.lz / effective_cutoff) + 1;
    
    int total_cells = nx * ny * nz;
    cells.clear();
    cells.resize(total_cells);
    
    // Step 3: Assign particles to cells
    for (size_t i = 0; i < particles.num_particles; i++) {
        int ix = static_cast<int>(particles.x[i] / effective_cutoff);
        int iy = static_cast<int>(particles.y[i] / effective_cutoff);
        int iz = static_cast<int>(particles.z[i] / effective_cutoff);
        
        // Apply periodic boundary conditions
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
    
    // Step 4: Initialize neighbor list arrays
    nlist.neighbors.clear();
    nlist.num_neighbors.resize(particles.num_particles, 0);
    nlist.neighbor_offset.resize(particles.num_particles, 0);
    
    // Step 5: Build neighbor lists separately for interior and boundary particles
    build_interior_neighbors(particles, box, nlist);
    build_boundary_neighbors(particles, box, nlist);
    
    // Print classification statistics
    // print_classification_stats();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
}

void OptimizedCellListNeighborBuilder::print_classification_stats() const {
    if (rank == 0) {
        std::cout << "=== Optimized Neighbor List Statistics ===" << std::endl;
        std::cout << "3D Domain decomposition: " << dims_x << "x" << dims_y << "x" << dims_z << std::endl;
        std::cout << "Halo width: " << halo_width << std::endl;
        std::cout << "Particle classification stats:" << std::endl;
        std::cout << "  Interior particles: " << interior_particles.size() << std::endl;
        std::cout << "  Boundary particles: " << boundary_particles.size() << std::endl;
        std::cout << "  Total particles: " << (interior_particles.size() + boundary_particles.size()) << std::endl;
        std::cout << "  Boundary ratio: " << (static_cast<double>(boundary_particles.size()) / 
                     (interior_particles.size() + boundary_particles.size()) * 100.0) << "%" << std::endl;
        std::cout << "=============================================" << std::endl;
    }
}

// =============================================================================
// TiledNeighborListBuilder Implementation
// =============================================================================

TiledNeighborListBuilder::TiledNeighborListBuilder(Real cutoff_val, int tile_size)
    : cutoff(cutoff_val), tile_size(tile_size) {
}

void TiledNeighborListBuilder::initialize() {
    if (is_rank_zero()) {
        std::cout << "Initializing tiled neighbor list builder with tile size: " << tile_size << std::endl;
    }
}

void TiledNeighborListBuilder::finalize() {
    cell_tiles.clear();
}

void TiledNeighborListBuilder::build_tile(int tile_start, int tile_end, 
                                         ParticleData& particles, SimulationBox& box, 
                                         NeighborList& nlist) {
    Real cutoff_sq = cutoff * cutoff;
    
    for (int i = tile_start; i < tile_end && i < static_cast<int>(particles.num_particles); i++) {
        nlist.neighbor_offset[i] = nlist.neighbors.size();
        
        // Simple brute force within tile for now
        // In practice, you would use cell-based approach within each tile
        for (size_t j = 0; j < particles.num_particles; j++) {
            if (static_cast<int>(j) == i) continue;
            
            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            
            // Apply minimum image convention
            if (box.periodic_x) {
                if (dx > box.lx * 0.5) dx -= box.lx;
                else if (dx < -box.lx * 0.5) dx += box.lx;
            }
            if (box.periodic_y) {
                if (dy > box.ly * 0.5) dy -= box.ly;
                else if (dy < -box.ly * 0.5) dy += box.ly;
            }
            if (box.periodic_z) {
                if (dz > box.lz * 0.5) dz -= box.lz;
                else if (dz < -box.lz * 0.5) dz += box.lz;
            }
            
            Real r2 = dx * dx + dy * dy + dz * dz;
            
            if (r2 < cutoff_sq) {
                nlist.neighbors.push_back(j);
                nlist.num_neighbors[i]++;
                
                // Limit neighbors
                if (nlist.num_neighbors[i] >= 200) {
                    break;
                }
            }
        }
    }
}

void TiledNeighborListBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist) {
    nlist.cutoff = cutoff;
    nlist.max_neighbors_per_atom = 200;
    nlist.list_type = "tiled";
    
    nlist.neighbors.clear();
    nlist.num_neighbors.resize(particles.num_particles, 0);
    nlist.neighbor_offset.resize(particles.num_particles, 0);
    
    // Process particles in tiles
    int num_particles = static_cast<int>(particles.num_particles);
    for (int tile_start = 0; tile_start < num_particles; tile_start += tile_size) {
        int tile_end = std::min(tile_start + tile_size, num_particles);
        build_tile(tile_start, tile_end, particles, box, nlist);
    }
}
