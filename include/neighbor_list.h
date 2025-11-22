#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include "md_types.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

enum class NeighborListType {
    VERLET_CELL_LIST,
    MORTON_Z_ORDER_CSR
};

class NeighborListBuilder {
public:
    virtual ~NeighborListBuilder() = default;
    virtual void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local = 0) = 0;
    virtual void initialize() = 0;
    virtual void finalize() = 0;
    virtual NeighborListType get_type() const = 0;
};



// Verlet list with cell-based acceleration (hybrid approach)
class VerletCellListNeighborBuilder : public NeighborListBuilder {
private:
    Real cutoff;
    Real cell_size;
    Real verlet_skin;
    int nx, ny, nz;
    int update_frequency;
    int delay_steps;
    bool info_printed;  // Flag to print info only once
    std::vector<std::vector<int>> cells;
    void broadcast_neighbor_list(NeighborList& nlist, size_t n, int mpi_rank, int mpi_size);
    
    // Automatic rebuild check (LAMMPS-style)
    std::vector<Real> last_build_x, last_build_y, last_build_z;
    Real max_displacement_threshold;
    bool check_enabled;
    
public:
    VerletCellListNeighborBuilder(Real cutoff, Real skin = 1.0);
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local = 0) override;
    void initialize() override;
    void finalize() override;
    NeighborListType get_type() const override { return NeighborListType::VERLET_CELL_LIST; }

    // Set update frequency information for logging
    void set_update_info(int freq, int delay = 0) { update_frequency = freq; delay_steps = delay; }
    
    // Check if neighbor list needs rebuild (LAMMPS-style displacement check)
    bool needs_rebuild(const ParticleData& particles, size_t num_local);
    void enable_check(bool enable) { check_enabled = enable; }
};

// Modern Morton Z-order curve based neighbor list builder with CSR format
// Optimized for MPI distribution and CUDA/SYCL GPU acceleration
// Uses space-filling curve for excellent cache locality and parallel efficiency
class MortonZOrderNeighborBuilder : public NeighborListBuilder {
private:
    Real cutoff;
    Real verlet_skin;
    int morton_levels;  // Depth of Morton curve subdivision (typically 10-21 bits per dimension)
    int update_frequency;
    int delay_steps;
    bool info_printed;  // Flag to print info only once
    
    // Morton Z-order encoding structure
    struct MortonCode {
        uint64_t code;
        int particle_id;
        bool operator<(const MortonCode& other) const { return code < other.code; }
    };
    
    // CSR (Compressed Sparse Row) format for neighbor list - GPU friendly
    struct CSRNeighborList {
        std::vector<int> neighbors;     // All neighbor indices (contiguous)
        std::vector<int> row_ptr;       // Start index for each particle (n+1 elements)
        std::vector<int> col_indices;   // Column indices for symmetric matrices
    };
    
    // Tile staging for GPU memory coalescing and warp-level optimization
    struct TileStaging {
        std::vector<std::vector<int>> tiles;
        int tile_size;  // Typically 32 (warp size) or 256 for better occupancy
    };
    
    std::vector<MortonCode> morton_codes;
    std::vector<int> particle_reorder;  // Original -> Morton-sorted index
    std::vector<int> inverse_reorder;   // Morton-sorted -> original index
    CSRNeighborList csr_nlist;
    TileStaging tile_staging;
    
    // GPU memory management
    bool gpu_available = false;
    bool gpu_memory_allocated = false;
    void* d_neighbor_list = nullptr;
    void* d_tile_data = nullptr;
    int gpu_device_id = 0;
    
#ifdef USE_SYCL
    sycl::queue sycl_queue;
#endif
    
    // Morton Z-order encoding functions (optimized with bit interleaving)
    uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z);
    void morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z);
    
    // MPI-aware broadcast with chunked transfer for large data
    void optimized_mpi_broadcast(NeighborList& nlist, size_t n, int mpi_rank, int mpi_size);
    
    // Particle reordering for cache-coherent access patterns
    void reorder_particles(ParticleData& particles);
    void restore_particle_order(ParticleData& particles);
    
    // CSR neighbor list construction with spatial sorting
    void build_csr_neighbor_list(ParticleData& particles, SimulationBox& box, int max_neighbors_per_particle);
    
    // Tile staging for warp-efficient GPU kernels
    void create_tile_staging();
    
public:
    MortonZOrderNeighborBuilder(Real cutoff, Real skin = 2.0, int tile_size = 256);
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local = 0) override;
    void initialize() override;
    void finalize() override;
    NeighborListType get_type() const override { return NeighborListType::MORTON_Z_ORDER_CSR; }
    
    // Set update frequency information for logging
    void set_update_info(int freq, int delay = 0) { update_frequency = freq; delay_steps = delay; }
    
    // GPU-specific functions (CUDA and SYCL compatible)
    void copy_to_gpu(NeighborList& nlist);
    const CSRNeighborList& get_csr_format() const { return csr_nlist; }
    const TileStaging& get_tile_staging() const { return tile_staging; }
};

// Factory function to create appropriate neighbor list builder
std::unique_ptr<NeighborListBuilder> create_neighbor_list_builder(
    NeighborListType type, Real cutoff, const std::string& config = "", int mpi_rank = 0);

#endif // NEIGHBOR_LIST_H
