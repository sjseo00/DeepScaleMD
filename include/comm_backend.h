#ifndef COMM_BACKEND_H
#define COMM_BACKEND_H

#include <memory>
#include <cstddef>

#include "md_types.h"

class MDSystem;
class ForceCalculator;

class CommBackend {
public:
    virtual ~CommBackend() = default;

    virtual bool is_distributed() const = 0;
    virtual void unify_rebuild_decision(bool& should_rebuild) = 0;
    virtual bool migrate_particles_if_needed(MDSystem& system,
                                             bool should_rebuild,
                                             ForceCalculator* force_calc) = 0;
    virtual bool is_gpu_single_rank(bool using_cuda) const = 0;

    virtual Real max_reduce(Real local_value) const = 0;
    virtual Real sum_reduce(Real local_value) const = 0;
    virtual int sum_int(int local_value) const = 0;
    virtual bool all_true(bool local_value) const = 0;
    virtual void reduce_max_force_particle(Real local_force,
                                           size_t local_particle_id,
                                           int local_rank,
                                           Real& global_force,
                                           size_t& global_particle_id) const = 0;
};

std::unique_ptr<CommBackend> create_comm_backend();

#endif
