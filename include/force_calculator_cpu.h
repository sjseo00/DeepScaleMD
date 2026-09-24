#ifndef FORCE_CALCULATOR_CPU_H
#define FORCE_CALCULATOR_CPU_H

#include "force_calculator.h"
#include "md_system.h"
#include <vector>
#include <cstdint>
#include <chrono>
#include <fftw3.h>
#include <mpi.h>

class Topology;
class CpuSpatialVerletList;

class ForceCalculatorCPU : public ForceCalculator {
public:
    ForceCalculatorCPU() = default;
    virtual ~ForceCalculatorCPU();

    virtual void initialize() override;
    virtual void finalize() override;
    virtual void compute_forces(MDSystem& system) override;
    virtual void calculate_kinetic_energy(MDSystem& system) override;

    // Diagnostic per-force-component timing (MD_TIMING_TRACE_FORCE=1). Wraps
    // each force contribution in compute_forces() with std::chrono, mirroring
    // ForceCalculatorCUDA's cudaEvent-based breakdown. Off by default (skips
    // the chrono calls entirely).
    void set_timing_breakdown_enabled(bool enabled) override { timing_breakdown_enabled_ = enabled; }
    bool is_timing_breakdown_enabled() const override { return timing_breakdown_enabled_; }
    const std::vector<ForceComponentTiming>& get_last_force_component_timings() const override {
        return last_force_component_timings_;
    }

    // Diagnostic: see ForceCalculator::get_lj_pair_counts(). Populated by
    // compute_tiled_nonbonded()/compute_graph_nonbonded() whenever
    // do_lj is true, index-aligned with local particles 0..local_particles-1.
    const std::vector<int>& get_lj_pair_counts() const override {
        return lj_pair_counts_;
    }

    // force_calculation.precision = mixed. Wires MDSimulation's config knob
    // straight to mixed_precision_nonbonded_ (marking it as explicitly
    // decided, so the MD_NONBONDED_PRECISION env-var lazy probe in
    // ensure_mixed_precision_tables() never overrides it) instead of the
    // previous state where this config value was silently ignored on the
    // CPU backend and only the env var had any effect. See
    // ensure_mixed_precision_tables()'s comment for the AVX2 requirement.
    void set_mixed_precision(bool enabled) override;

private:
    void compute_bond_forces(MDSystem& system);
    void compute_angle_forces(MDSystem& system);
    void compute_dihedral_forces(MDSystem& system);
    void compute_improper_forces(MDSystem& system);
    // Shared by compute_dihedral_forces/compute_improper_forces: propers and
    // impropers use the identical periodic-torsion functional form and
    // analytical gradient, just over different atom-quadruple lists.
    void compute_torsion_forces(MDSystem& system, const std::vector<MDSystem::LocalDihedral>& torsions);
    // Scaled 1-4 (or FF-specific) nonbonded pairs re-added on top of
    // [exclusions]; see Topology::SpecialPair for the exact semantics.
    void compute_special_pair_forces(MDSystem& system);

    // Reciprocal-space (FFTW PME grid) part + self-energy correction, as a
    // standalone begin+end pair for callers with no real-space kernel to
    // overlap it against (see pme_reciprocal_begin/end below).
    void compute_electrostatic_forces_reciprocal(MDSystem& system);

    // PERF FIX: the reciprocal-space pass split in two so the inter-rank
    // charge-grid reduction can be OVERLAPPED with the real-space kernel.
    //
    //   pme_reciprocal_begin()  spread rho(r), then start a non-blocking
    //                           MPI_Ireduce of the grid onto rank 0
    //   <the caller runs the real-space pair kernel here>
    //   pme_reciprocal_end()    wait, FFT/Green/FFT on rank 0, broadcast
    //                           Phi(r), interpolate forces, self-energy
    //
    // The grid reduce is the one part of PME that is pure communication and
    // depends on nothing the real-space kernel touches (it reads positions
    // and writes the PME grid; the pair kernel reads positions and writes
    // forces), so it is free to run underneath the most expensive phase of
    // the step instead of serialising in front of it. On one rank there is
    // no reduce and the split is a pure no-op.
    //
    // compute_electrostatic_forces_reciprocal() remains as begin+end
    // back-to-back for callers that have nothing to overlap.
    void pme_reciprocal_begin(MDSystem& system);
    void pme_reciprocal_end(MDSystem& system);
    MPI_Request pme_reduce_req_ = MPI_REQUEST_NULL;
    bool pme_reduce_pending_ = false;
    // Real time spent inside begin()+end() themselves, i.e. excluding the
    // overlapped real-space kernel, reported as the "coulomb_recip" component.
    double pme_time_ms_ = 0.0;

    // Shared by compute_graph_nonbonded()/compute_tiled_nonbonded(): the two
    // kernel backends run genuinely different pair-loop bodies (directed CSR
    // vs AVX-512 tile graph) and pick their own timing label (graph only
    // ever needs two, tiled needs three -- see each caller), but the PME
    // reciprocal-space overlap and per-step timing WRAPPED AROUND that body
    // is identical, so it lives here once instead of being duplicated in
    // both functions. `body` is the kernel's real-space pass.
    template <typename Fn>
    void run_nonbonded_with_pme_overlap(MDSystem& system, bool do_coulomb, const char* label, Fn&& body) {
        if (do_coulomb) {
            ensure_ewald_initialized(system);
            pme_reciprocal_begin(system);
        }
        using clock = std::chrono::high_resolution_clock;
        clock::time_point t0;
        if (timing_breakdown_enabled_) t0 = clock::now();

        body();

        if (timing_breakdown_enabled_) {
            last_force_component_timings_.push_back(
                {label, std::chrono::duration<double, std::milli>(clock::now() - t0).count()});
        }
        if (do_coulomb) {
            pme_reciprocal_end(system);
        }
    }
    // Unified tiled nonbonded kernel: LJ and/or real-space (erfc-screened)
    // Ewald Coulomb over the same neighbor-list tile-pair graph, gated by
    // runtime do_lj/do_coulomb flags -- one distance calc, one minimum-image
    // shift, one exclusion lookup, one neighbor traversal per pair, shared
    // between whichever interaction(s) are enabled, instead of the three
    // near-duplicate hand-written passes this replaced (LJ-only, Coulomb-
    // only, LJ+Coulomb-fused). The scalar LJ/Coulomb pair math itself is
    // further factored out into compute_lj_pair()/compute_coulomb_pair_
    // real_space() (top of this file), shared with compute_graph_nonbonded()
    // -- only the neighbor-list traversal and distance calc differ between
    // the two kernel backends, not the physics. The PME reciprocal-space
    // overlap and component timing around this pass are shared with
    // compute_graph_nonbonded() via run_nonbonded_with_pme_overlap() above.
    // compute_forces() calls this or compute_graph_nonbonded() -- the two
    // kernel backends -- and nothing else about LJ/Coulomb dispatch lives in
    // compute_forces() itself.
    void compute_tiled_nonbonded(MDSystem& system, bool do_lj, bool do_coulomb);
    void compute_graph_nonbonded(MDSystem& system, bool do_lj, bool do_coulomb);
    void ensure_ewald_initialized(MDSystem& system);
    void free_ewald_resources();

    // PERF FIX: Topology::is_nonbonded_excluded() binary-searches ONE global
    // sorted array of all ~200k exclusions in the system (~18 branches/call)
    // for every single candidate pair, every force step -- profiling (perf
    // stat) showed this codebase executing ~28x more branches per instruction
    // than NAMD on the identical apoa1 benchmark, and this per-pair global
    // search is the leading suspect (bonded exclusion topology never changes
    // during a run, so re-searching the full global array from scratch every
    // step for every pair is pure repeated work). Decodes topology's
    // exclusion_keys once into a per-GLOBAL-ATOM-id CSR (offsets_[num_atoms+1]
    // + a small sorted partners_ list per atom, typically ~2-5 entries),
    // so each pair's exclusion check becomes a binary search over a handful
    // of elements instead of the full ~200k-entry global array. Correctness
    // is identical to is_nonbonded_excluded() -- same underlying exclusion
    // set, just re-indexed by atom for a much smaller per-query search space.
    void ensure_exclusion_cache_built(const Topology& topology);
    std::vector<int> excl_cache_offsets_;   // size num_atoms + 1, indexed by global_id
    std::vector<int> excl_cache_partners_;  // flat, sorted per atom's [offset, offset+1) range
    bool excl_cache_built_ = false;

    // STAGE 3 PLACEHOLDER (PME reciprocal-space parallelization roadmap):
    // Fast Multipole Method long-range electrostatics solver, for the regime
    // where FFT-based PME's communication cost (all-to-all transposes in a
    // slab/pencil-distributed FFT) becomes the hard scaling limit at very
    // large atom/rank counts. Not implemented -- MDSimulation rejects
    // force_calculation.electrostatics_solver = fmm at config-parse time
    // before this would ever be called; this stub exists purely so the call
    // site/API shape is already in place for that future work.
    void compute_electrostatic_forces_fmm(MDSystem& system);

    bool timing_breakdown_enabled_ = false;
    std::vector<ForceComponentTiming> last_force_component_timings_;
    std::vector<int> lj_pair_counts_;
    bool graph_kernel_logged_ = false;
    // ===== EWALD / PME RECIPROCAL-SPACE ELECTROSTATICS (CPU, FFTW) =====
    bool ewald_initialized_ = false;
    int ewald_grid_x_ = 0, ewald_grid_y_ = 0, ewald_grid_z_ = 0;
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;
    Real ewald_prefactor_ = 0.0;  // 4*pi*k_coulomb / box_volume

    // PERF FIX: real-to-complex (R2C/C2R) in-place grid, replacing the old
    // complex-to-complex pair. The charge density is purely real (the
    // imaginary half of the C2C grid was always exactly zero), so a C2C
    // transform did twice the FFT butterflies, held twice the memory, and
    // -- since the whole grid is MPI_Reduce'd to rank 0 and MPI_Bcast back
    // -- moved twice the bytes over the network, for nothing.
    //
    // ONE allocation, two views of it (the standard FFTW in-place R2C
    // layout):
    //   pme_real_grid_    : rho(r), then Phi(r) after the backward transform.
    //                       Row-major [gx][gy][gz] with the LAST dimension
    //                       PADDED to pme_grid_z_pad_ = 2*(nz/2+1) reals --
    //                       the padding slots are never read or written by
    //                       our own code, they only exist so the complex
    //                       half-spectrum fits in the same buffer.
    //   pme_complex_grid_ : rho(k), then Phi(k) after the Poisson solve.
    //                       Dimensions [nx][ny][nz/2+1] complex; the kz >
    //                       nz/2 half is implied by Hermitian symmetry and
    //                       is never stored (see ewald_solve_poisson_cpu's
    //                       kz weighting for how the energy/virial sums
    //                       account for the unstored conjugate half).
    // c2r destroys its input, which is fine: the Poisson solve and the
    // virial sum both run before the backward transform.
    Real* pme_real_grid_ = nullptr;
    fftw_complex* pme_complex_grid_ = nullptr;  // aliases pme_real_grid_
    int pme_grid_z_pad_ = 0;       // 2*(nz/2+1), the real grid's z stride
    int pme_grid_z_half_ = 0;      // nz/2+1, the complex grid's z extent
    std::vector<Real> pme_b_spline_mod_x_, pme_b_spline_mod_y_, pme_b_spline_mod_z_;
    fftw_plan pme_fft_plan_fwd_ = nullptr;  // r2c, rho(r) -> rho(k)
    fftw_plan pme_fft_plan_bwd_ = nullptr;  // c2r, Phi(k) -> Phi(r)

    // PERF FIX: precomputed Green's function, replacing two full-grid passes
    // that each recomputed std::exp(-k^2/4alpha^2) and a division at every
    // single k point, every step (~2.3M transcendentals/step on a 120^3-ish
    // grid). For fixed box dimensions -- i.e. everything except NPT -- these
    // are compile-time-constant-in-time arrays; ensure_pme_green_cached()
    // rebuilds them only when the box actually changes.
    //   pme_green_       : the in-place k-space multiplier, Phi(k) = g*rho(k)
    //   pme_green_energy_: 0.5*w*g, w = the Hermitian multiplicity (1 for the
    //                      kz planes that are their own conjugate, else 2)
    //   pme_green_virial_: w*g*(1 + k^2/(2 alpha^2))
    // All three are zero at k=0, which reproduces the old code's explicit
    // "zero the DC bin / skip it in the sums" special case with no branch.
    std::vector<Real> pme_green_, pme_green_energy_, pme_green_virial_;
    Real pme_green_cached_lx_ = 0.0, pme_green_cached_ly_ = 0.0, pme_green_cached_lz_ = 0.0;
    void ensure_pme_green_cached(const SimulationBox& box);

    // Thread-private charge-spread scratch: ewald_spread_charges_cpu scatters
    // each atom's charge into up to spline_order^3 grid cells, and different
    // atoms/threads can land on the SAME cell, so a direct parallel-for over
    // atoms would race on charge_grid writes. Each thread accumulates into
    // its own full-size slice here; a separate parallel-for-over-grid-points
    // pass then sums the nthreads slices into charge_grid. Sized
    // nthreads * grid_size (real-valued, no imaginary component needed);
    // reused across calls to avoid reallocation.
    std::vector<Real> pme_charge_spread_priv_;

    // ===== REAL-SPACE EWALD LOOKUP TABLES (PERF) =====
    // compute_nonbonded_forces_fused()/compute_electrostatic_forces_real_
    // space() spend the overwhelming majority of a single-core step here
    // (measured: ~95% of per-step time on the apoa1 benchmark), and per-pair
    // cost was dominated by std::exp() + a rational erfc approximation +
    // sqrt(). Since both the energy factor (erfc(alpha*r)/r, or erf(...)/r
    // for excluded pairs) and the force factor depend only on r^2, they're
    // precomputed once (using exact std::erf/erfc -- table build cost is
    // negligible, it happens once per run) on a uniform r^2 grid and looked
    // up + linearly interpolated per pair instead, matching the classic
    // "tabulated nonbonded" approach NAMD itself uses (see its "NONBONDED
    // TABLE" startup log line). Two separate tables are needed since
    // excluded pairs (present in the neighbor list but screened by erf
    // instead of erfc, so their real-space contribution exactly cancels the
    // reciprocal-space term for that pair) use a different function than
    // ordinary pairs. Built once in ensure_ewald_initialized() (same guard
    // as the rest of the one-time PME setup) and never resized after.
    //
    // PERF FIX (layout): energy and force are INTERLEAVED into a single
    // array -- [e0,f0, e1,f1, ...] -- instead of living in two separate
    // ones. Every lookup needs both factors at the same table index, and
    // linear interpolation needs index+1 too, so the interleaved form puts
    // all four doubles a lookup touches inside 32 consecutive bytes (one
    // cache line) instead of spreading them across two cache lines in two
    // distant arrays. Same tabulated values, half the cache pressure; the
    // AVX2 path's four gathers now all hit the same line.
    std::vector<Real> ewald_rs_table_;       // erfc-screened (normal pairs), interleaved e/f
    std::vector<Real> ewald_rs_table_excl_;  // erf-based (excluded pairs), interleaved e/f
    Real ewald_rs_table_r2_min_ = 0.0;
    Real ewald_rs_table_inv_spacing_ = 0.0;
    int ewald_rs_table_size_ = 0;            // number of NODES (not doubles)

    // PERF FIX: flat, interleaved [c6, c12] LJ coefficient table replacing
    // the per-pair reconstruction of sigma^6/sigma^12 out of
    // ForceFieldParams::epsilon_mixed/sigma_mixed. Those are
    // vector<vector<Real>>, i.e. two pointer chases to two rows that sit in
    // unrelated heap allocations, and every single pair then paid
    // sig*sig, sig2*sig2*sig2 and a 4.0*eps multiply to rebuild constants
    // that depend only on the type pair. Here c6 = 4*eps*sigma^6 and
    // c12 = 4*eps*sigma^12 are precomputed once at
    // [2*(num_types*type_i + type_j)] and [.. + 1], so a pair's LJ
    // coefficients are two adjacent doubles in one flat array -- one cache
    // line, no pointer chase, no per-pair reconstruction.
    void ensure_lj_coeff_table_built(const ForceFieldParams& params);
    std::vector<Real> lj_c6c12_;
    int lj_coeff_num_types_ = 0;

    // Per-tile bounding spheres used by the tile-pair cull in
    // compute_nonbonded_forces_fused(). PERF FIX: kept as members (reused
    // allocations) instead of four std::vectors heap-allocated and freed on
    // every single step.
    std::vector<Real> tile_cx_, tile_cy_, tile_cz_, tile_rad_;

    // ===== CLUSTER-PACKED TILE SoA =====
    // PERF FIX: per-atom data permuted into tile-slot order, so a pair kernel
    // reads a neighbour tile's 8 atoms as ONE contiguous run instead of
    // gathering them through tile_atoms[]. Indexed exactly like
    // CpuSpatialVerletList::get_tile_atoms(): slot s of tile T is at
    // T*kCpuTileSize + s, with padding slots carrying atom 0's (harmless,
    // always masked) values.
    //
    // This is the packed-SoA half of what GROMACS' nbnxn gets by permuting
    // the particle arrays themselves. Doing it as a SIDE array instead means
    // no global atom reordering: the canonical ParticleData order -- which
    // the halo exchange, the topology's local indices, constraints and
    // trajectory output all depend on -- is untouched.
    //
    // Rebuilt every step (positions move), which costs one O(N) gather with
    // sequential writes. The kernels previously re-gathered each tile once
    // per tile PAIR it appeared in, i.e. ~50x more often at typical pair
    // counts.
    void repack_tile_soa(const ParticleData& particles, const CpuSpatialVerletList& nlist);
    std::vector<Real> tile_px_, tile_py_, tile_pz_, tile_q_;
    std::vector<int>  tile_type_, tile_gid_;

    // ===== MIXED PRECISION (opt-in: MD_NONBONDED_PRECISION=mixed) =====
    // Single-precision mirrors of the packed SoA and of the pair-parameter
    // tables, feeding an 8-wide AVX2 float kernel in
    // compute_nonbonded_forces_fused(). TILE_SIZE is 8 and an AVX2 float
    // vector is 8 wide, so one tile pair's j side becomes exactly one vector
    // -- half the instructions and half the load/gather bytes of the 4-wide
    // double path, with the divide replaced by rcp+Newton.
    //
    // OFF BY DEFAULT, deliberately. It changes the numerics of every
    // simulation (float pair arithmetic, ~1e-6 relative on forces -- the same
    // trade as GROMACS' default mixed-precision build), so it is a decision
    // for whoever owns the run, not a silent default. Accumulation stays in
    // double: per-(i, tile pair) sums fold back after at most 8 pairs.
    //
    // Validate with an energy-conservation run before using it in anger.
    void ensure_mixed_precision_tables();
    bool mixed_precision_nonbonded_ = false;
    bool mixed_precision_probed_ = false;
    std::vector<float> tile_fx_, tile_fy_, tile_fz_, tile_fq_;
    std::vector<float> lj_c6c12_f_;
    std::vector<float> ewald_rs_table_f_, ewald_rs_table_excl_f_;

    // ===== BAKED PER-TILE-PAIR EXCLUSION MASKS =====
    // PERF FIX: one 64-bit mask per tile pair, bit (lane*kCpuTileSize + slot)
    // set when that (i,j) atom pair is a bonded exclusion. Replaces a
    // per-pair search -- two dependent random loads into the exclusion CSR
    // plus an in-row scan, inside the innermost loop, defeating vectorization
    // -- with a shift and a mask.
    //
    // Baked once per NEIGHBOUR-LIST REBUILD (keyed on
    // CpuSpatialVerletList::build_generation()), not per step: the tile graph
    // is reused for ~10-20 steps, so the 64 lookups per tile pair are
    // amortised over that whole window. Exclusion topology never changes
    // during a run, and atom-to-slot assignment only changes on a rebuild,
    // so the cache is exact.
    //
    // Requires repack_tile_soa() to have run first (it reads tile_gid_).
    void ensure_tile_exclusion_masks(const CpuSpatialVerletList& nlist);
    std::vector<std::uint64_t> tile_pair_excl_;
    std::uint64_t tile_excl_generation_ = ~static_cast<std::uint64_t>(0);

    // ===== LJ KERNEL: PER-THREAD PRIVATE FORCE ACCUMULATION =====
    // PERF FIX (false sharing): the previous design assigned every tile to
    // exactly one owner thread (contiguous, edge-count-balanced tile-id
    // ranges) so that direct writes to particles.f{x,y,z} could never race,
    // diverting cross-owner j-reactions into a small preassigned "remote
    // reaction" slot merged after a barrier. That protected against races,
    // but NOT against false sharing: tile membership is grouped by spatial
    // proximity, not by original atom-array index (see repack_tile_soa()'s
    // comment -- atoms are never physically reordered), so two atoms that
    // share a cache line in particles.fx/fy/fz can easily land in tiles
    // owned by two DIFFERENT threads. Every step, that pair of threads then
    // ping-pongs that cache line back and forth -- measured via `perf stat`
    // on this benchmark: cache-references barely changed 1->8 threads
    // (+9.7%) while cache-misses more than doubled (+136%, 10.0%->21.6%
    // miss rate) and per-thread user time INCREASED 24.7% despite 8x the
    // threads, both signatures of cross-core coherence misses rather than
    // more useful work.
    //
    // Fix: every thread accumulates into its OWN full-size private force
    // buffer (lj_thread_fx_/fy_/fz_, size nthreads*local_particles),
    // indexed by the same original atom id as particles.fx/fy/fz. Since each
    // thread's slice is written by that thread ONLY, there is no cross-
    // thread write to the same memory ever, so tiles need no ownership
    // assignment at all -- compute_tiled_nonbonded() dispatches them with a
    // plain dynamic "omp for" (which also fixes the old static edge-count
    // partition's load-imbalance exposure). A single chunked parallel
    // reduction pass (same pattern as ewald_spread_charges_cpu's merge
    // below) then sums the nthreads private slices into particles.fx/fy/fz
    // once per step.
    void ensure_lj_thread_buffers(int nthreads, size_t local_particles);
    std::vector<Real> lj_thread_fx_, lj_thread_fy_, lj_thread_fz_;  // size nthreads*local_particles

    // ===== LJ KERNEL: PER-TILE VALIDITY/LOCALITY + SLOT-MAJOR MASKS =====
    // The redesigned LJ kernel vectorizes over the home tile's 8 i-atoms
    // (lanes) for one j-slot at a time, the transpose of tile_pair_excl_'s
    // lane-major bit layout, so it needs its own slot-major masks:
    //   lj_tile_valid_mask_[T]  bit s = tile T's slot s is a real atom (not
    //                           -1 padding).
    //   lj_tile_local_mask_[T]  bit s = that atom is LOCAL (owned by this
    //                           rank), used both as the home-tile "i must be
    //                           local" mask and as the neighbour-tile "j is
    //                           local" (reaction-eligible) mask.
    //   lj_same_tile_mask_[T]   bit (slot*8+lane): 1 if pair (i=lane,
    //                           j=slot) should be evaluated when a tile
    //                           pairs with ITSELF (half-list intra-tile
    //                           tie-break, replacing the old scalar
    //                           "j_atom > i_atom" branch with one AND).
    void ensure_lj_tile_masks(const CpuSpatialVerletList& nlist, size_t local_particles);
    std::vector<uint8_t> lj_tile_valid_mask_, lj_tile_local_mask_;
    std::vector<std::uint64_t> lj_same_tile_mask_;
    std::uint64_t lj_tile_mask_generation_ = ~static_cast<std::uint64_t>(0);
    size_t lj_tile_mask_local_particles_ = ~static_cast<size_t>(0);

    // Transpose of tile_pair_excl_ (bit (slot*8+lane) instead of
    // (lane*8+slot)), needed by the same i-vector/j-scalar orientation as
    // above. Derived from tile_pair_excl_, so it is only ever rebuilt when
    // that cache itself changes (tile_excl_generation_).
    void ensure_lj_tile_excl_col(const CpuSpatialVerletList& nlist);
    std::vector<std::uint64_t> lj_tile_excl_col_;
    std::uint64_t lj_tile_excl_col_generation_ = ~static_cast<std::uint64_t>(0);
};

#endif // FORCE_CALCULATOR_CPU_H
