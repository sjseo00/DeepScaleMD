#include "comm_backend.h"

#include "md_system.h"
#include "force_calculator.h"
#include "force_calculator_cuda.h"

#include <mpi.h>

namespace {

class SerialCommBackend final : public CommBackend {
public:
    bool is_distributed() const override {
        return false;
    }

    void unify_rebuild_decision(bool&) override {
    }

    bool migrate_particles_if_needed(MDSystem& system,
                                     bool should_rebuild,
                                     ForceCalculator* force_calc) override {
        if (!should_rebuild) return false;

#ifdef USE_CUDA
        auto* fc_cuda = dynamic_cast<ForceCalculatorCUDA*>(force_calc);
        if (fc_cuda) {
            return false;
        }
#endif

        system.migrate_particles_check();
        return false;
    }

    bool is_gpu_single_rank(bool using_cuda) const override {
        return using_cuda;
    }

    Real max_reduce(Real local_value) const override {
        return local_value;
    }

    Real sum_reduce(Real local_value) const override {
        return local_value;
    }

    int sum_int(int local_value) const override {
        return local_value;
    }

    bool all_true(bool local_value) const override {
        return local_value;
    }

    void reduce_max_force_particle(Real local_force,
                                   size_t local_particle_id,
                                   int,
                                   Real& global_force,
                                   size_t& global_particle_id) const override {
        global_force = local_force;
        global_particle_id = local_particle_id;
    }
};

class MpiCommBackend final : public CommBackend {
public:
    bool is_distributed() const override {
        return true;
    }

    void unify_rebuild_decision(bool& should_rebuild) override {
        int local_rebuild = should_rebuild ? 1 : 0;
        int global_rebuild = 0;
        MPI_Allreduce(&local_rebuild, &global_rebuild, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        should_rebuild = global_rebuild != 0;
    }

    bool migrate_particles_if_needed(MDSystem& system,
                                     bool should_rebuild,
                                     ForceCalculator* force_calc) override {
        if (!should_rebuild) return false;

#ifdef USE_CUDA
        auto* fc_cuda = dynamic_cast<ForceCalculatorCUDA*>(force_calc);
        if (fc_cuda) {
            return system.migrate_particles_on_device(*fc_cuda);
        }
#endif

        system.migrate_particles_check();
        return false;
    }

    bool is_gpu_single_rank(bool using_cuda) const override {
        if (!using_cuda) return false;
        int mpi_world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
        return mpi_world_size == 1;
    }

    Real max_reduce(Real local_value) const override {
        Real global_value = local_value;
        MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        return global_value;
    }

    Real sum_reduce(Real local_value) const override {
        Real global_value = local_value;
        MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        return global_value;
    }

    int sum_int(int local_value) const override {
        int global_value = local_value;
        MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        return global_value;
    }

    bool all_true(bool local_value) const override {
        int local_ok = local_value ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        return global_ok != 0;
    }

    void reduce_max_force_particle(Real local_force,
                                   size_t local_particle_id,
                                   int local_rank,
                                   Real& global_force,
                                   size_t& global_particle_id) const override {
        struct {
            Real value;
            int rank;
        } local_max, global_max;

        local_max.value = local_force;
        local_max.rank = local_rank;

        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);
        global_force = global_max.value;

        unsigned long long particle_id = static_cast<unsigned long long>(local_particle_id);
        MPI_Bcast(&particle_id, 1, MPI_UNSIGNED_LONG_LONG, global_max.rank,
                  MPI_COMM_WORLD);
        global_particle_id = static_cast<size_t>(particle_id);
    }
};

} // namespace

std::unique_ptr<CommBackend> create_comm_backend() {
    return std::make_unique<MpiCommBackend>();
}
