#ifndef PME_RANK_SPLIT_H
#define PME_RANK_SPLIT_H

#include <mpi.h>

// Splits MPI_COMM_WORLD into a "PP" (particle/real-space) group and a "PME"
// (reciprocal-space) group, GROMACS-style, so PME's global FFT is confined to
// a small dedicated group of ranks instead of every rank in the simulation.
//
// Bypass semantics: when pme_rank_count == 0 (the default), initialize()
// leaves this singleton inactive and pp_comm()/pme_comm()/inter_comm() report
// MPI_COMM_WORLD / MPI_COMM_NULL / MPI_COMM_NULL respectively, so every call
// site written as `active() ? pp_comm() : MPI_COMM_WORLD` reproduces today's
// exact single-group behavior with zero change.
class PmePpSplit {
public:
    static PmePpSplit& instance();

    // Must be called exactly once, on every rank, after the simulation config
    // (which carries pme_rank_count) has been parsed but before any other
    // rank-consequential MPI collective (domain decomposition, I/O, etc.).
    void initialize(int pme_rank_count);

    bool active() const { return active_; }
    bool is_pme_rank() const { return active_ && is_pme_rank_; }
    // True on the rank that is local rank 0 of pme_comm(); meaningless if !is_pme_rank().
    bool is_pme_leader() const { return active_ && is_pme_rank_ && pme_local_rank_ == 0; }

    MPI_Comm pp_comm() const { return active_ ? pp_comm_ : MPI_COMM_WORLD; }
    MPI_Comm pme_comm() const { return active_ ? pme_comm_ : MPI_COMM_NULL; }
    // Intercommunicator spanning both groups: local group is whichever of
    // pp_comm()/pme_comm() this rank belongs to, remote group is the other.
    MPI_Comm inter_comm() const { return active_ ? inter_comm_ : MPI_COMM_NULL; }

    int pp_size() const { return active_ ? pp_size_ : world_size_at_init_; }
    int pme_size() const { return active_ ? pme_size_ : 0; }

    // Releases the communicators created by initialize(). Safe to call even
    // if initialize() was never called or left this inactive.
    void finalize();

private:
    PmePpSplit() = default;

    bool active_ = false;
    bool is_pme_rank_ = false;
    int pme_local_rank_ = -1;
    int pp_size_ = 1;
    int pme_size_ = 0;
    int world_size_at_init_ = 1;

    MPI_Comm pp_comm_ = MPI_COMM_NULL;
    MPI_Comm pme_comm_ = MPI_COMM_NULL;
    MPI_Comm inter_comm_ = MPI_COMM_NULL;
};

#endif // PME_RANK_SPLIT_H
