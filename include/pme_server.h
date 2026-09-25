#ifndef PME_SERVER_H
#define PME_SERVER_H

#include "pme_reciprocal_engine.h"
#include "pme_rank_split.h"

// Drives the dedicated PME-group side of the PP/PME rank split
// (force_calculation.pme_ranks > 0): a rank in this role owns no atoms, runs
// no dsDecompX domain decomposition, and does nothing but repeatedly receive
// a reduced charge grid from the PP group over the intercommunicator, solve
// it, and broadcast the potential grid (+ reciprocal energy/virial) back.
//
// Constructed and driven only from MDSimulation::initialize_impl on ranks
// where PmePpSplit::instance().is_pme_rank() is true; PP ranks never touch
// this class.
class PmeServer {
public:
    explicit PmeServer(PmePpSplit& split);

    // Blocks in a receive loop until the PP group signals it is done, then
    // returns. total_steps is NOT used to bound the loop: the PP side calls
    // pme_reciprocal_begin()/end() (and therefore needs one PME step served)
    // once per compute_forces() call, and that includes energy-minimization
    // iterations before the MD loop even starts -- those aren't counted in
    // total_steps, so a fixed total_steps-iteration loop here silently runs
    // out early and deadlocks the PP side partway through the MD loop (see
    // MDSimulation::finalize()'s matching stop broadcast and
    // pme_reciprocal_begin()'s per-call continue broadcast, both on
    // split.inter_comm()). Instead, every iteration first receives a small
    // int flag broadcast from the PP group's leader: nonzero means "one more
    // step follows", zero means "PP is finished, stop". The real-space
    // cutoff is still read once from cutoff here, parsed identically from
    // the same input file on every rank. Box dimensions are not known on
    // this side (PME ranks own no atoms/box) -- each step() receives them
    // fresh from the PP group before touching the engine, so NPT box
    // changes are picked up automatically every step.
    void run(int total_steps, Real cutoff);

private:
    // One reciprocal-space step, group size == 1 (default, always available):
    // receive box dims, receive reduced charge grid, solve, broadcast
    // potential grid + energy/virial.
    void step();

#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    // One reciprocal-space step, group size > 1 (force_calculation.pme_ranks
    // > 1). Deliberately funnels through the PME leader rather than using
    // intercommunicator MPI_Reduce_scatter/MPI_Allgatherv directly: those
    // are rarely-exercised MPI operations and crashed (segfault inside
    // OpenMPI's coll/basic reduce_scatter_inter) in real testing. Instead:
    // intercomm MPI_Reduce (full grid -> this group's leader, identical
    // mechanism to the group-size==1 path) -> intracommunicator
    // MPI_Scatterv (leader -> each rank's slab, ordinary/well-tested) ->
    // collective solve() across pme_comm -> intracommunicator MPI_Gatherv
    // (slabs -> leader) -> intercomm MPI_Bcast (full grid + energy/virial
    // back to PP, identical mechanism to the group-size==1 path). This
    // funnels more data through one rank than a true reduce-scatter would,
    // but only the leader-mediated hop is exotic-collective-free; the FFT
    // itself is still genuinely distributed across the whole PME group.
    // See PmeReciprocalEngine's class-level risk note.
    void step_distributed();
    // One-time (cached) exchange of each PME rank's slab size with the
    // leader, needed to build MPI_Scatterv/MPI_Gatherv count/displ arrays.
    void ensure_slab_layout_exchanged();
    std::vector<int> slab_counts_;   // real-element count per PME rank, PME-rank order (leader only, meaningful)
    std::vector<int> slab_displs_;   // prefix-sum offsets into full_grid_buffer_ (leader only, meaningful)
    std::vector<Real> full_grid_buffer_;  // leader-only: full global grid, staging for Reduce/Scatterv/Gatherv/Bcast
#endif

    PmePpSplit& split_;
    PmeReciprocalEngine engine_;
    Real cutoff_ = 0.0;
    SimulationBox box_{};
};

#endif // PME_SERVER_H
