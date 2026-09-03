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
    std::vector<std::vector<int>> cells;  // kept for backward compat, now unused
    // LAMMPS-style flat cell linked list (replaces cells above)
    std::vector<int> flat_cell_head;  // flat_cell_head[c] = first atom in cell c (-1 = empty)
    std::vector<int> flat_cell_next;  // flat_cell_next[i] = next atom in same cell (-1 = end)
    // Reusable work arrays (avoid reallocation each build)
    std::vector<int> cx_arr, cy_arr, cz_arr;  // per-particle cell index caches
    // Precomputed stencil offsets [(dx,dy,dz)] for 27-cell search
    std::vector<int> stencil_dx, stencil_dy, stencil_dz;
    int stencil_size;
    // Half-list (Newton ON): only store j > i for local-local pairs.
    // Set to false for GPU path (GPU kernel uses Newton OFF full-list).
    bool use_half_list;
    void broadcast_neighbor_list(NeighborList& nlist, size_t n, int mpi_rank, int mpi_size);
    
    // Automatic rebuild check (LAMMPS-style)
    std::vector<Real> last_build_x, last_build_y, last_build_z;
    Real max_displacement_threshold;
    bool check_enabled;
    
public:
    VerletCellListNeighborBuilder(Real cutoff, Real skin = 1.0, bool half_list = true);
    void build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local = 0) override;
    void initialize() override;
    void finalize() override;
    NeighborListType get_type() const override { return NeighborListType::VERLET_CELL_LIST; }

    // Set update frequency information for logging
    void set_update_info(int freq, int delay = 0) { update_frequency = freq; delay_steps = delay; }
    // Switch between half-list (Newton ON, CPU) and full-list (Newton OFF, GPU)
    void set_use_half_list(bool v) { use_half_list = v; }
    
    // Check if neighbor list needs rebuild (LAMMPS-style displacement check)
    bool needs_rebuild(const ParticleData& particles, const SimulationBox& box, size_t num_local);
    void enable_check(bool enable) { check_enabled = enable; }
    Real get_skin_distance() const { return verlet_skin; }
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
    
    // Per-build Morton order (sorted by code)
    std::vector<MortonCode> morton_codes;
    // old -> new (Morton order) and new -> old mappings
    std::vector<int> particle_reorder;
    std::vector<int> inverse_reorder;
    
    // LAMMPS-style displacement check: skip rebuild if max displacement < skin/2
    std::vector<Real> last_build_x, last_build_y, last_build_z;
    Real max_displacement_threshold;  // (skin/2)^2
    bool displacement_check_enabled;
    bool has_last_build;  // whether we have a previous build to compare against
    
    // GPU memory management
    bool gpu_available = false;
    bool gpu_memory_allocated = false;
    
    // Persistent GPU buffers for Morton-reordered build (allocated once & reused)
    size_t gpu_capacity_particles = 0;
    size_t gpu_capacity_local = 0;
    size_t gpu_capacity_neighbor_slots = 0;
    size_t gpu_capacity_cells = 0;
    void* d_x = nullptr;
    void* d_y = nullptr;
    void* d_z = nullptr;
    void* d_perm = nullptr;      // old -> new
    void* d_invperm = nullptr;   // new -> old
    void* d_x_reord = nullptr;
    void* d_y_reord = nullptr;
    void* d_z_reord = nullptr;
    void* d_neighbor_list_tmp = nullptr; // neighbor list in reordered indices
    void* d_num_neighbors_tmp = nullptr;
    void* d_neighbor_offset_tmp = nullptr;
    // Morton-specific GPU buffers (for launch_cuda_morton_neighbor_list_build)
    void* d_morton_codes = nullptr;      // uint64 Morton codes per particle
    void* d_sorted_indices = nullptr;    // sorted permutation indices
    void* d_cell_start_morton = nullptr; // cell start in sorted array
    void* d_cell_end_morton = nullptr;   // cell end in sorted array
    void* d_compact_neighbors = nullptr; // GPU-compacted neighbor list (CSR values)
    
    int gpu_device_id = 0;
    
#ifdef USE_SYCL
    sycl::queue sycl_queue;
#endif
    
    // Morton Z-order encoding functions (optimized with bit interleaving)
    uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z);
    void morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z);
    
    // Compute Morton order permutations for current particle positions.
    void build_morton_permutation(const ParticleData& particles, const SimulationBox& box,
                                  size_t n_local, size_t n_total);
    
    // CPU helper: build a reordered view of particles (positions only) so we can reuse Verlet-cell build.
    void build_reordered_position_arrays(
        const ParticleData& particles,
        std::vector<Real>& x_reord,
        std::vector<Real>& y_reord,
        std::vector<Real>& z_reord,
        size_t n_total) const;
    
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
    
    // Kept for backward build compatibility: Morton builder no longer exposes CSR/tile staging.
};

// Factory function to create appropriate neighbor list builder
std::unique_ptr<NeighborListBuilder> create_neighbor_list_builder(
    NeighborListType type, Real cutoff, const std::string& config = "", int mpi_rank = 0);

#endif // NEIGHBOR_LIST_H
