#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include "md_types.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <limits>

// Kind of a concrete NeighborList implementation. Used by the factory and by
// callers that need to know which backend they got.
//
// CUDA builds do NOT use this interface for their nonbonded force path: the
// GPU-resident spatial tile graph (8-atom tiles, AABB-pruned tile pairs) is
// built and consumed directly in ForceCalculatorCUDA/neighbor_list_kernels.cu
// without ever going through a NeighborList subclass or producing an
// atom-CSR list. The only CUDA-build consumer of this interface is the MLIP
// GNN edge-list builder (ForceCalculatorCUDA::build_coo_edge_list_on_device),
// which reads the CPU_SPATIAL_VERLET instance every rank already builds.
enum class NeighborListBackend {
    CPU_SPATIAL_VERLET
};

inline bool safe_to_use_half_shell(size_t num_local, size_t total) {
    return total <= num_local;
}

// Abstract neighbor list: builds and stores, per owned atom, the set of
// candidate atoms (owned or ghost) within (physical cutoff + skin). Consumers
// (force calculators) read it back as a CSR structure: get_offsets() has
// num_local+1 entries, and atom i's neighbors are
// get_neighbors()[get_offsets()[i] .. get_offsets()[i+1]).
//
// Implementations choose their own internal pair-list convention (half-list
// vs full-list) as an implementation detail; both expose the same CSR shape.
class NeighborList {
public:
    virtual ~NeighborList() = default;

    // particles: full SoA, owned atoms in [0, num_local) followed by ghost
    // atoms in [num_local, num_local+num_ghost) (dsdecompx::ParticleInfo
    // layout). box: current per-rank subdomain box. num_local: boundary
    // between owned and ghost slots (system->get_local_particles()).
    virtual void build(const ParticleData& particles, const SimulationBox& box, size_t num_local) = 0;

    // Rebuild if needed; implementations with no cheaper incremental path may
    // simply delegate to build().
    virtual void update(const ParticleData& particles, const SimulationBox& box, size_t num_local) = 0;

    // LAMMPS-style half-skin displacement criterion: true once any atom
    // (owned or ghost) has moved more than skin/2 since the last build().
    // Must not mutate state and must be cheap to call every step.
    virtual bool needs_rebuild(const ParticleData& particles, const SimulationBox& box, size_t num_local) const = 0;

    // Local (this rank only, no MPI) squared max displacement since the last
    // build() -- the raw value behind needs_rebuild(), exposed so a caller
    // can apply its own hi/lo hysteresis band (see
    // ForceCalculatorCUDA::compute_max_displacement_on_device, which the CPU
    // path now mirrors in MDSimulation::handle_neighbor_phase instead of
    // using a single hardcoded threshold with no hysteresis buffer). Default
    // falls back to needs_rebuild()'s own single-threshold verdict for any
    // implementation that hasn't opted into the two-threshold scheme.
    virtual Real compute_local_max_displacement_sq(
        const ParticleData& particles, const SimulationBox& box, size_t num_local) const {
        return needs_rebuild(particles, box, num_local)
            ? std::numeric_limits<Real>::max() : static_cast<Real>(0.0);
    }

    virtual const int* get_neighbors() const = 0;
    virtual const int* get_offsets() const = 0;   // size num_local + 1
    virtual int num_pairs() const = 0;

    // Rebuild the compact, directed force/MLIP graph from the Verlet outer
    // list using current coordinates and the active cutoff (rc plus a small
    // nested skin). Force kernels still enforce the exact physical rc. Rows
    // belong to owned atoms; columns may address owned or ghost atoms.
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

    // Tells the implementation whether some consumer OTHER than the tiled/
    // graph nonbonded kernels (currently: ForceCalculatorCUDA::
    // build_coo_edge_list_on_device for the LibTorch MLIP GNN edge list --
    // see get_neighbors()/get_offsets() above) will read the atom-pair CSR.
    // Default false ("nobody needs it"); CpuSpatialVerletList uses this to
    // skip building that CSR entirely in tiled-kernel mode, where nothing
    // else reads it (see its build()'s comment). Harmless no-op for any
    // implementation that doesn't distinguish.
    virtual void set_mlip_edge_list_needed(bool /*needed*/) {}

    // Nested outer/active reprune of the TILED kernel's atom-pair mask (see
    // CpuSpatialVerletList's override doc comment). Default false/no-op for
    // any implementation that doesn't have a tiled-kernel mask of its own.
    virtual bool needs_tile_reprune(const ParticleData& /*particles*/,
                                    const SimulationBox& /*box*/,
                                    size_t /*num_local*/) const {
        return false;
    }
    virtual void reprune_tile_verlet_mask(const ParticleData& /*particles*/,
                                          const SimulationBox& /*box*/,
                                          size_t /*num_local*/) {}

    // Physical cutoff + skin (the radius pairs were actually searched to).
    virtual Real search_cutoff() const = 0;

    virtual NeighborListBackend kind() const = 0;
};

// Tile width for CpuSpatialVerletList's spatial tile graph (see that
// class's public accessors below). File-scope (not a class static member)
// so it's usable from force_calculator_cpu.cpp without depending on the
// build's C++ standard version for in-class constexpr ODR rules.
//
// CPU tile width used by ForceCalculatorCPU's rank-local tile graph.
// Keep this aligned with a reasonably small tile so bbox pruning stays tight
// and tile-pair false positives do not dominate pair checks.
constexpr int kCpuTileSize = 8;

// CPU backend: counting-sorted cell list + 27-cell-stencil Verlet pair
// generation, OpenMP-parallel build, half-list (j > i) for local-local pairs.
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

    // NOTE: a nested outer/active reprune scheme for tile_pair_verlet_mask_
    // (tighter mask radius + periodic reprune between outer rebuilds,
    // mirroring prune_active()'s graph-mode scheme) was tried here and
    // reverted -- see the comment above tile_pair_verlet_mask_'s build in
    // src/neighbor_list_cpu.cpp for why it measured slower, not faster, and
    // needs_tile_reprune()/reprune_tile_verlet_mask()'s default no-op base-
    // class declarations above for the (currently unused) hook that scheme
    // used, kept in case a per-SLOT-granularity-aware version is worth
    // revisiting later.
    Real active_search_cutoff() const { return cutoff_ + active_skin_; }
    Real search_cutoff() const override { return cutoff_ + skin_; }
    NeighborListBackend kind() const override { return NeighborListBackend::CPU_SPATIAL_VERLET; }

    // ---- Rank-local spatial tile graph (half-list, canonical direction) ----
    // Built alongside the atom-CSR above in build() (see src/neighbor_
    // list_cpu.cpp). Deliberately mirrors the ACTUAL CUDA tile-pair scheme
    // (neighbor_list_kernels.cu's build_lj_tile_pairs_fine_kernel +
    // force_calculator_cuda.cu's compute_lj_forces_tiled_kernel) pair-for-
    // pair, not the alternative "directed full list" design that was tried
        // here first and reverted: every unordered tile pair is kept once, and
        // in MPI mode any local-vs-ghost-only pair is oriented so the emitted
        // home tile contains at least one local atom (avoids dropping pairs when
        // force kernels skip ghost-only home lanes). Same-tile pairs use CUDA's
    // exact j>i tiebreak (applied only when j is local -- a local-ghost
    // same-tile pair has no "other side" to find it, exactly like the
    // atom-CSR's own half_shell rule), and ForceCalculatorCPU's kernels
    // apply an explicit reverse force to j. CPU LJ kernels use thread-private
    // reverse buffers with a sparse touched-entry merge by default; two home
    // tiles can concurrently reference the same neighbor tile. This lives
    // directly on this class rather than
    // outside the NeighborList interface, since there's no separate device-
    // resident build step to route around here. ForceCalculatorCPU's
    // nonbonded kernels use this instead of the atom-CSR; the atom-CSR is
    // kept only for the MLIP GNN edge-list builder (ForceCalculatorCUDA::
    // build_coo_edge_list_on_device) and any other CSR consumer.
    //
    // Every tile is TILE_SIZE atoms (owned and/or ghost, chunked from the
    // same counting-sorted per-cell atom runs the CSR build uses), padded
    // with -1. tile_pair_offsets()[t]..[t+1) (into get_tile_pair_other())
        // lists every OTHER tile (including itself) in that directed form.
        // CPU build additionally applies a tile-AABB prune at (cutoff+skin)
        // before emitting a pair, reducing false-positive tile work while
        // preserving force-kernel-side per-atom cutoff exactness.
    // described above. Pair admission includes a tile-AABB distance prune in
    // addition to the cell-stencil candidate generation.
    //
    // A home tile's force pass only computes the home (i) side directly for
    // its OWN atoms (a lane only does work when its atom is owned, i.e.
    // index < num_local -- ghost atoms occupy tile slots but are never a
    // "home" lane). Every owned atom therefore appears in exactly one home
    // lane. Home-side writes are direct and race-free because concurrent
    // reaction-side writes go only to thread-private buffers.
    const int* get_tile_atoms() const { return tile_atoms_.data(); }
    const int* get_tile_pair_offsets() const { return tile_pair_offsets_.data(); }
    const int* get_tile_pair_other() const { return tile_pair_other_.data(); }
    const uint8_t* get_tile_has_local() const { return tile_has_local_.data(); }
    int num_tiles() const { return num_tiles_; }
    int num_tile_pairs() const { return static_cast<int>(tile_pair_other_.size()); }
    // Rebuild-time atom-pair Verlet filter for each tile edge. Bit
    // (slot*8+lane) represents home lane `lane` against neighbor slot `slot`.
    const std::uint64_t* get_tile_pair_verlet_mask() const {
        return tile_pair_verlet_mask_.data();
    }
    // Per-edge periodic-image shift: the neighbor tile's atoms, in the home
    // tile's frame, sit at (jx+shift_x[e], jy+shift_y[e], jz+shift_z[e]).
    // Computed once per rebuild from which wrapped cell-stencil offset
    // produced this tile pair (see build()), so a force kernel can fold this
    // straight into dx/dy/dz instead of re-deriving it per pair with the
    // usual half-box compare/branch minimum-image convention. Valid for the
    // life of this tile graph -- like the rest of it, it is rebuilt whenever
    // the outer neighbor list is (skin margin guarantees no atom crosses to
    // a different periodic image of its neighbor in between). Always exactly
    // 0 in MPI subdomain mode (ghosts already carry their wrapped position).
    const Real* get_tile_pair_shift_x() const { return tile_pair_shift_x_.data(); }
    const Real* get_tile_pair_shift_y() const { return tile_pair_shift_y_.data(); }
    const Real* get_tile_pair_shift_z() const { return tile_pair_shift_z_.data(); }
    std::uint64_t pairs_inside_rc() const { return pairs_inside_rc_; }
    std::uint64_t pairs_inside_rc_plus_skin() const { return pairs_inside_rc_plus_skin_; }

    // Incremented on every build(). Lets a consumer cache anything derived
    // from the tile graph -- ForceCalculatorCPU bakes per-tile-pair exclusion
    // masks off it (see tile_pair_excl_ in force_calculator_cpu.h) -- and
    // rebuild that cache only when the graph itself actually changed, rather
    // than every step. The tile graph is reused for ~10-20 steps between
    // rebuilds, so anything keyed on this is amortised over that window.
    uint64_t build_generation() const { return build_generation_; }

private:
    Real cutoff_;
    Real skin_;
    Real max_displacement_threshold_;  // (0.5 * skin)^2

    // CSR output (owned atoms only; values may reference ghost indices).
    std::vector<int> neighbors_;
    std::vector<int> offsets_;  // size num_local + 1

    // Compact directed graph at the physical cutoff.  The outer CSR above is
    // a Newton half list; pruning expands local-local pairs in both
    // directions, while local-ghost pairs remain in the owned source row.
    std::vector<int> active_neighbors_;
    std::vector<int> active_offsets_;
    std::vector<int> active_degree_;
    std::vector<Real> last_prune_x_, last_prune_y_, last_prune_z_;
    Real active_skin_ = 0.0;


    // Rebuild-decision snapshot (owned + ghost, taken at last build()).
    std::vector<Real> last_build_x_, last_build_y_, last_build_z_;

    // Reused scratch across builds to avoid reallocation.
    std::vector<int> cell_of_;            // per (unsorted) particle index -> flat cell id
    std::vector<int> cell_start_;         // CSR over cell_of_, size num_cells + 1
    std::vector<int> per_atom_count_;     // scratch: pairs found per owned atom

    // Padded per-atom scratch for the single-pass build (see build()'s
    // comment on why the old count-then-fill two-pass was replaced): sized
    // local_N * neighbor_capacity_per_atom_. Each row's tail past the
    // atom's actual count is uninitialized garbage, not a sentinel value --
    // nothing ever reads it (see build()'s comment on why no per-rebuild
    // fill is needed). neighbor_capacity_per_atom_ starts from a density
    // estimate and only ever grows (persisted across builds), so repeat
    // rebuilds at a stable density hit capacity on the first try.
    std::vector<int> padded_neighbors_;
    int neighbor_capacity_per_atom_ = 0;

    // Rank-local spatial tile graph (see the public accessors' doc comment
    // above); built alongside the atom-CSR in build().
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
    // See set_mlip_edge_list_needed() above.
    bool mlip_edge_list_needed_ = false;
};

// Factory: constructs the concrete backend for `kind`. cutoff is the
// physical cutoff (skin is added internally by the backend, matching the
// existing nlist.cutoff = cutoff + skin convention).
std::unique_ptr<NeighborList> create_neighbor_list(NeighborListBackend kind, Real cutoff, Real skin);

#endif // NEIGHBOR_LIST_H
