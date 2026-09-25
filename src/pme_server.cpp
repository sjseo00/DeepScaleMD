#include "pme_server.h"

#include <mpi.h>

PmeServer::PmeServer(PmePpSplit& split) : split_(split) {
    box_.lx = box_.ly = box_.lz = 1.0;
    box_.periodic_x = box_.periodic_y = box_.periodic_z = true;
    box_.update_box_length();
}

void PmeServer::run(int /*total_steps*/, Real cutoff) {
    cutoff_ = cutoff;
    MPI_Comm inter = split_.inter_comm();
    while (true) {
        int continue_flag = 0;
        // Root is rank 0 of the remote (PP) group -- matches
        // pme_reciprocal_begin()'s per-call continue broadcast and
        // MDSimulation::finalize()'s final stop broadcast, both issued from
        // the PP leader on this same intercommunicator.
        MPI_Bcast(&continue_flag, 1, MPI_INT, /*root in remote (PP) group*/ 0, inter);
        if (!continue_flag) break;
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
        if (split_.pme_size() > 1) {
            step_distributed();
        } else {
            step();
        }
#else
        step();
#endif
    }
}

void PmeServer::step() {
    MPI_Comm inter = split_.inter_comm();
    const bool leader = split_.is_pme_leader();
    const int leader_root = leader ? MPI_ROOT : MPI_PROC_NULL;

    // Box dims come from the PP group's leader (its local rank 0) every step
    // so NPT box changes propagate without a separate handshake. On the PP
    // side this is a matching MPI_Bcast issued from pme_reciprocal_begin()
    // before the charge-grid reduce below.
    Real box_dims[3] = {box_.lx, box_.ly, box_.lz};
    MPI_Bcast(box_dims, 3, MPI_DOUBLE, /*root in remote (PP) group*/ 0, inter);
    box_.lx = box_dims[0];
    box_.ly = box_dims[1];
    box_.lz = box_dims[2];
    box_.update_box_length();

    engine_.ensure_initialized(cutoff_, box_);
    engine_.ensure_green_cached(box_);

    MPI_Request req;
    MPI_Ireduce(nullptr, engine_.grid_data(), static_cast<int>(engine_.real_grid_size()),
                MPI_DOUBLE, MPI_SUM, leader_root, inter, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    Real recip_energy = 0.0, recip_virial = 0.0;
    if (leader) {
        recip_energy = engine_.solve(recip_virial);
    }

    MPI_Bcast(engine_.grid_data(), static_cast<int>(engine_.real_grid_size()), MPI_DOUBLE,
              leader_root, inter);

    Real energy_virial[2] = {recip_energy, recip_virial};
    MPI_Bcast(energy_virial, 2, MPI_DOUBLE, leader_root, inter);
}

#ifdef DEEPSCALEMD_HAVE_FFTW_MPI

void PmeServer::ensure_slab_layout_exchanged() {
    if (!slab_counts_.empty()) return;  // one-time; grid dims are fixed after first ensure_initialized

    MPI_Comm pme_comm = split_.pme_comm();
    const int pme_size = split_.pme_size();
    const int my_count = static_cast<int>(engine_.real_grid_size());

    // Only the leader needs every rank's slab size (it drives the
    // Scatterv/Gatherv below) -- an ordinary intracommunicator Gather.
    slab_counts_.assign(static_cast<size_t>(pme_size), 0);
    MPI_Gather(&my_count, 1, MPI_INT, slab_counts_.data(), 1, MPI_INT, 0, pme_comm);

    slab_displs_.assign(static_cast<size_t>(pme_size), 0);
    if (split_.is_pme_leader()) {
        for (int i = 1; i < pme_size; ++i) {
            slab_displs_[i] = slab_displs_[i - 1] + slab_counts_[i - 1];
        }
    }
}

void PmeServer::step_distributed() {
    MPI_Comm inter = split_.inter_comm();
    MPI_Comm pme_comm = split_.pme_comm();
    const bool leader = split_.is_pme_leader();
    const int leader_root = leader ? MPI_ROOT : MPI_PROC_NULL;

    Real box_dims[3] = {box_.lx, box_.ly, box_.lz};
    MPI_Bcast(box_dims, 3, MPI_DOUBLE, /*root in remote (PP) group*/ 0, inter);
    box_.lx = box_dims[0];
    box_.ly = box_dims[1];
    box_.lz = box_dims[2];
    box_.update_box_length();

    engine_.ensure_initialized_distributed(cutoff_, box_, pme_comm);
    engine_.ensure_green_cached(box_);
    ensure_slab_layout_exchanged();

    const int global_count = engine_.grid_x() * engine_.grid_y() * engine_.grid_z_pad();
    if (leader && full_grid_buffer_.size() != static_cast<size_t>(global_count)) {
        full_grid_buffer_.assign(static_cast<size_t>(global_count), 0.0);
    }

    // Step 1: reduce the full grid from the PP group onto this group's
    // leader -- same intercomm MPI_Reduce mechanism as the group-size==1
    // path (root = MPI_ROOT on the leader, MPI_PROC_NULL on other PME
    // ranks; PP ranks all pass root=0, this group's leader rank).
    MPI_Reduce(nullptr, leader ? full_grid_buffer_.data() : nullptr, global_count,
               MPI_DOUBLE, MPI_SUM, leader_root, inter);

    // Step 2: leader hands each rank its slab -- ordinary intracommunicator
    // Scatterv.
    MPI_Scatterv(leader ? full_grid_buffer_.data() : nullptr,
                slab_counts_.data(), slab_displs_.data(), MPI_DOUBLE,
                engine_.grid_data(), static_cast<int>(engine_.real_grid_size()), MPI_DOUBLE,
                /*root in pme_comm*/ 0, pme_comm);

    // Collective across pme_comm -- every PME rank participates, each gets
    // back its own local partial energy/virial sum.
    Real local_energy = 0.0, local_virial = 0.0;
    local_energy = engine_.solve(local_virial);

    Real local_sums[2] = {local_energy, local_virial};
    Real total_sums[2] = {0.0, 0.0};
    MPI_Allreduce(local_sums, total_sums, 2, MPI_DOUBLE, MPI_SUM, pme_comm);

    // Step 3: gather solved slabs back onto the leader -- intracommunicator
    // Gatherv.
    MPI_Gatherv(engine_.grid_data(), static_cast<int>(engine_.real_grid_size()), MPI_DOUBLE,
               leader ? full_grid_buffer_.data() : nullptr, slab_counts_.data(), slab_displs_.data(),
               MPI_DOUBLE, 0, pme_comm);

    // Step 4: leader broadcasts the reassembled full grid + energy/virial
    // back to the PP group -- same intercomm MPI_Bcast mechanism as the
    // group-size==1 path.
    MPI_Bcast(leader ? full_grid_buffer_.data() : nullptr, global_count, MPI_DOUBLE, leader_root, inter);
    MPI_Bcast(total_sums, 2, MPI_DOUBLE, leader_root, inter);
}

#endif  // DEEPSCALEMD_HAVE_FFTW_MPI
