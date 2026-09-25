#include "pme_rank_split.h"

#include <stdexcept>

namespace {
constexpr int kPpColor = 0;
constexpr int kPmeColor = 1;
constexpr int kInterCommTag = 4242;
}  // namespace

PmePpSplit& PmePpSplit::instance() {
    static PmePpSplit s_instance;
    return s_instance;
}

void PmePpSplit::initialize(int pme_rank_count) {
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    world_size_at_init_ = world_size;

    if (pme_rank_count <= 0) {
        active_ = false;
        is_pme_rank_ = false;
        pp_size_ = world_size;
        pme_size_ = 0;
        return;
    }

    if (pme_rank_count >= world_size) {
        throw std::runtime_error(
            "PmePpSplit::initialize: pme_ranks must leave at least one PP rank");
    }

    // PME ranks are the LAST pme_rank_count ranks of MPI_COMM_WORLD, so rank 0
    // (used throughout the codebase as the "print/log from here" rank) always
    // stays a PP rank and every existing rank==0 gate keeps working unchanged.
    const int pp_size = world_size - pme_rank_count;
    const bool is_pme = world_rank >= pp_size;
    const int color = is_pme ? kPmeColor : kPpColor;

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local_comm);

    MPI_Comm inter_comm = MPI_COMM_NULL;
    const int remote_leader_world_rank = is_pme ? 0 : pp_size;
    MPI_Intercomm_create(local_comm, /*local_leader=*/0, MPI_COMM_WORLD,
                         remote_leader_world_rank, kInterCommTag, &inter_comm);

    active_ = true;
    is_pme_rank_ = is_pme;
    inter_comm_ = inter_comm;

    if (is_pme) {
        pme_comm_ = local_comm;
        pp_comm_ = MPI_COMM_NULL;
        MPI_Comm_rank(pme_comm_, &pme_local_rank_);
        MPI_Comm_size(pme_comm_, &pme_size_);
        pp_size_ = pp_size;
    } else {
        pp_comm_ = local_comm;
        pme_comm_ = MPI_COMM_NULL;
        pme_local_rank_ = -1;
        MPI_Comm_size(pp_comm_, &pp_size_);
        pme_size_ = pme_rank_count;
    }
}

void PmePpSplit::finalize() {
    if (pp_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&pp_comm_);
    }
    if (pme_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&pme_comm_);
    }
    if (inter_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&inter_comm_);
    }
    active_ = false;
    is_pme_rank_ = false;
    pme_local_rank_ = -1;
}
