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
    MORTON_HILBERT_CSR
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
    std::vector<std::vector<int>> cells;
    
public:
    VerletCellListNeighborBuilder(Real cutoff, Real skin = 1.0);
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local = 0) override;
    void initialize() override;
    void finalize() override;
    NeighborListType get_type() const override { return NeighborListType::VERLET_CELL_LIST; }
};

// Modern Morton/Hilbert curve based neighbor list builder with CSR format and tile staging
// Compatible with CUDA, SYCL, and CPU backends
class MortonHilbertNeighborBuilder : public NeighborListBuilder {
private:
    Real cutoff;
    Real verlet_skin;  // Skin distance for stability
    // int max_particles_per_tile;  // Currently unused
    int morton_levels;  // Depth of Morton curve subdivision
    
    // Morton curve encoding/decoding
    struct MortonCode {
        uint64_t code;
        int particle_id;
        bool operator<(const MortonCode& other) const { return code < other.code; }
    };
    
    // CSR (Compressed Sparse Row) format for neighbor list
    struct CSRNeighborList {
        std::vector<int> neighbors;     // All neighbor indices
        std::vector<int> row_ptr;       // Start index for each particle
        std::vector<int> col_indices;   // Column indices (same as neighbors for symmetric)
    };
    
    // Tile staging for GPU memory coalescing
    struct TileStaging {
        std::vector<std::vector<int>> tiles;
        int tile_size;
    };
    
    std::vector<MortonCode> morton_codes;
    std::vector<int> particle_reorder;  // Mapping from original to reordered indices
    std::vector<int> inverse_reorder;   // Mapping from reordered to original indices
    CSRNeighborList csr_nlist;
    TileStaging tile_staging;
    
    // GPU-related variables
    bool gpu_available = false;
    bool gpu_memory_allocated = false;
    void* d_neighbor_list = nullptr;
    void* d_tile_data = nullptr;
    int gpu_device_id = 0;
    
#ifdef USE_SYCL
    sycl::queue sycl_queue;
#endif
    
    // Morton curve functions
    uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z);
    void morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z);
    uint64_t hilbert_encode_3d(uint32_t x, uint32_t y, uint32_t z, int level);
    
    // Particle reordering for better cache locality
    void reorder_particles(ParticleData& particles);
    void restore_particle_order(ParticleData& particles);
    
    // CSR neighbor list construction
    void build_csr_neighbor_list(ParticleData& particles, SimulationBox& box, int max_cap);
    
    // Tile staging for GPU optimization
    void create_tile_staging();
    
public:
    MortonHilbertNeighborBuilder(Real cutoff, Real skin = 2.0, int tile_size = 256);
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local = 0) override;
    void initialize() override;
    void finalize() override;
    NeighborListType get_type() const override { return NeighborListType::MORTON_HILBERT_CSR; }
    
    // GPU-specific functions (CUDA and SYCL compatible)
    void copy_to_gpu(NeighborList& nlist);
    const CSRNeighborList& get_csr_format() const { return csr_nlist; }
    const TileStaging& get_tile_staging() const { return tile_staging; }
};

// Factory function to create appropriate neighbor list builder
std::unique_ptr<NeighborListBuilder> create_neighbor_list_builder(
    NeighborListType type, Real cutoff, const std::string& config = "", int mpi_rank = 0);

#endif // NEIGHBOR_LIST_H
