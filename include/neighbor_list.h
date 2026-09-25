#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include "md_types.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <limits>

enum class NeighborListBackend {
    CPU_SPATIAL_VERLET
};

inline bool safe_to_use_half_shell(size_t num_local, size_t total) {
    return total <= num_local;
}

class NeighborList {
public:
    virtual ~NeighborList() = default;

    virtual void build(const ParticleData& particles, const SimulationBox& box, size_t num_local) = 0;

    virtual void update(const ParticleData& particles, const SimulationBox& box, size_t num_local) = 0;

    virtual bool needs_rebuild(const ParticleData& particles, const SimulationBox& box, size_t num_local) const = 0;

    virtual Real compute_local_max_displacement_sq(
        const ParticleData& particles, const SimulationBox& box, size_t num_local) const {
        return needs_rebuild(particles, box, num_local)
            ? std::numeric_limits<Real>::max() : static_cast<Real>(0.0);
    }

    virtual const int* get_neighbors() const = 0;
    virtual const int* get_offsets() const = 0;   // size num_local + 1
    virtual int num_pairs() const = 0;

    virtual void prune_active(const ParticleData& particles,
                              const SimulationBox& box,
                              size_t num_local) = 0;
    virtual const int* get_active_neighbors() const = 0;
    virtual const int* get_active_offsets() const = 0; // size num_local + 1
    virtual int num_active_edges() const = 0;
    virtual void set_nonbonded_kernel(const std::string& kernel) = 0;
    virtual bool needs_active_prune(const ParticleData& particles,
                                    const SimulationBox& box,
                                    size_t num_local) const = 0;

    virtual void set_mlip_edge_list_needed(bool /*needed*/) {}

    virtual bool needs_tile_reprune(const ParticleData& /*particles*/,
                                    const SimulationBox& /*box*/,
                                    size_t /*num_local*/) const {
        return false;
    }
    virtual void reprune_tile_verlet_mask(const ParticleData& /*particles*/,
                                          const SimulationBox& /*box*/,
                                          size_t /*num_local*/) {}

    virtual Real search_cutoff() const = 0;

    virtual NeighborListBackend kind() const = 0;
};

constexpr int kCpuTileSize = 8;

class CpuSpatialVerletList : public NeighborList {
public:
    CpuSpatialVerletList(Real cutoff, Real skin);

    void build(const ParticleData& particles, const SimulationBox& box, size_t num_local) override;
    void update(const ParticleData& particles, const SimulationBox& box, size_t num_local) override;
    bool needs_rebuild(const ParticleData& particles, const SimulationBox& box, size_t num_local) const override;
    Real compute_local_max_displacement_sq(
        const ParticleData& particles, const SimulationBox& box, size_t num_local) const override;

    const int* get_neighbors() const override { return neighbors_.data(); }
    const int* get_offsets() const override { return offsets_.data(); }
    int num_pairs() const override { return static_cast<int>(neighbors_.size()); }
    void prune_active(const ParticleData& particles, const SimulationBox& box,
                      size_t num_local) override;
    const int* get_active_neighbors() const override { return active_neighbors_.data(); }
    const int* get_active_offsets() const override { return active_offsets_.data(); }
    int num_active_edges() const override { return static_cast<int>(active_neighbors_.size()); }
    void set_nonbonded_kernel(const std::string& kernel) override {
        build_tile_graph_ = (kernel == "tiled");
        build_active_graph_ = (kernel == "graph");
    }
    void set_mlip_edge_list_needed(bool needed) override { mlip_edge_list_needed_ = needed; }
    bool needs_active_prune(const ParticleData& particles,
                            const SimulationBox& box,
                            size_t num_local) const override;

    Real active_search_cutoff() const { return cutoff_ + active_skin_; }
    Real search_cutoff() const override { return cutoff_ + skin_; }
    NeighborListBackend kind() const override { return NeighborListBackend::CPU_SPATIAL_VERLET; }

    const int* get_tile_atoms() const { return tile_atoms_.data(); }
    const int* get_tile_pair_offsets() const { return tile_pair_offsets_.data(); }
    const int* get_tile_pair_other() const { return tile_pair_other_.data(); }
    const uint8_t* get_tile_has_local() const { return tile_has_local_.data(); }
    int num_tiles() const { return num_tiles_; }
    int num_tile_pairs() const { return static_cast<int>(tile_pair_other_.size()); }
    const std::uint64_t* get_tile_pair_verlet_mask() const {
        return tile_pair_verlet_mask_.data();
    }
    const Real* get_tile_pair_shift_x() const { return tile_pair_shift_x_.data(); }
    const Real* get_tile_pair_shift_y() const { return tile_pair_shift_y_.data(); }
    const Real* get_tile_pair_shift_z() const { return tile_pair_shift_z_.data(); }
    std::uint64_t pairs_inside_rc() const { return pairs_inside_rc_; }
    std::uint64_t pairs_inside_rc_plus_skin() const { return pairs_inside_rc_plus_skin_; }

    uint64_t build_generation() const { return build_generation_; }

private:
    Real cutoff_;
    Real skin_;
    Real max_displacement_threshold_;  // (0.5 * skin)^2

    std::vector<int> neighbors_;
    std::vector<int> offsets_;  // size num_local + 1

    std::vector<int> active_neighbors_;
    std::vector<int> active_offsets_;
    std::vector<int> active_degree_;
    std::vector<Real> last_prune_x_, last_prune_y_, last_prune_z_;
    Real active_skin_ = 0.0;


    std::vector<Real> last_build_x_, last_build_y_, last_build_z_;

    std::vector<int> cell_of_;            // per (unsorted) particle index -> flat cell id
    std::vector<int> cell_start_;         // CSR over cell_of_, size num_cells + 1
    std::vector<int> per_atom_count_;     // scratch: pairs found per owned atom

    std::vector<int> padded_neighbors_;
    int neighbor_capacity_per_atom_ = 0;

    std::vector<int> tile_atoms_;         // size num_tiles_*TILE_SIZE, -1 padding
    std::vector<int> tile_pair_offsets_;  // size num_tiles_ + 1
    std::vector<int> tile_pair_other_;    // flat neighbor-tile ids, directed (both directions + self)
    std::vector<std::uint64_t> tile_pair_verlet_mask_;
    std::vector<Real> tile_pair_shift_x_, tile_pair_shift_y_, tile_pair_shift_z_;  // parallel to tile_pair_other_
    std::vector<uint8_t> tile_has_local_; // size num_tiles_, 1 when any slot is local atom
    int num_tiles_ = 0;
    uint64_t build_generation_ = 0;       // see build_generation()
    std::uint64_t pairs_inside_rc_ = 0;
    std::uint64_t pairs_inside_rc_plus_skin_ = 0;
    bool build_tile_graph_ = true;
    bool build_active_graph_ = false;
    bool mlip_edge_list_needed_ = false;
};

std::unique_ptr<NeighborList> create_neighbor_list(NeighborListBackend kind, Real cutoff, Real skin);

#endif // NEIGHBOR_LIST_H
