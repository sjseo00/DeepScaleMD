#ifndef OPTIMIZED_NEIGHBOR_LIST_H
#define OPTIMIZED_NEIGHBOR_LIST_H

#include "neighbor_list.h"
#include "md_types.h"
#include <vector>
#include <memory>

/**
 * @class OptimizedCellListNeighborBuilder
 * @brief Optimized cell list neighbor builder with boundary/interior particle separation
 * 
 * This class separates particles into interior and boundary regions to optimize
 * MPI communication and force calculations. Interior particles have no dependencies
 * on neighboring domains, while boundary particles may require halo exchange.
 */
class OptimizedCellListNeighborBuilder : public NeighborListBuilder {
private:
    Real cell_size;
    Real halo_width;  // Width of halo region for boundary classification
    int nx, ny, nz;
    std::vector<std::vector<int>> cells;
    
    // Particle classification
    std::vector<int> interior_particles;
    std::vector<int> boundary_particles;
    std::vector<bool> is_boundary;
    
    // Domain boundaries for MPI
    Real domain_min_x, domain_max_x;
    Real domain_min_y, domain_max_y;
    Real domain_min_z, domain_max_z;
    
    // MPI info and 3D decomposition
    int rank, size;
    int rank_x, rank_y, rank_z;  // 3D rank coordinates
    int dims_x, dims_y, dims_z;  // Process grid dimensions
    
public:
    OptimizedCellListNeighborBuilder(Real cutoff, Real halo_width = 0.0);
    
    void initialize() override;
    void finalize() override;
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist) override;
    NeighborListType get_type() const override { return NeighborListType::OPTIMIZED_CELL_LIST; }
    
    // New methods for boundary/interior separation
    void classify_particles(ParticleData& particles, SimulationBox& box);
    void build_interior_neighbors(ParticleData& particles, SimulationBox& box, NeighborList& nlist);
    void build_boundary_neighbors(ParticleData& particles, SimulationBox& box, NeighborList& nlist);
    void setup_domain_boundaries(SimulationBox& box);
    void calculate_3d_decomposition();  // Calculate optimal 3D process grid
    
    // Getters for particle classification
    const std::vector<int>& get_interior_particles() const { return interior_particles; }
    const std::vector<int>& get_boundary_particles() const { return boundary_particles; }
    bool particle_is_boundary(int particle_id) const { 
        return particle_id < is_boundary.size() ? is_boundary[particle_id] : false; 
    }
    
    // Statistics
    void print_classification_stats() const;
};

/**
 * @class TiledNeighborListBuilder  
 * @brief Memory-efficient tiled neighbor list builder for large systems
 * 
 * Processes particles in tiles to reduce memory usage and improve cache locality
 */
class TiledNeighborListBuilder : public NeighborListBuilder {
private:
    Real cutoff;
    int tile_size;
    std::vector<std::vector<int>> cell_tiles;
    
public:
    TiledNeighborListBuilder(Real cutoff_val, int tile_size = 256);
    
    void initialize() override;
    void finalize() override;
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist) override;
    NeighborListType get_type() const override { return NeighborListType::TILED_CELL_LIST; }
    
private:
    void build_tile(int tile_start, int tile_end, ParticleData& particles, 
                   SimulationBox& box, NeighborList& nlist);
};

#endif // OPTIMIZED_NEIGHBOR_LIST_H
