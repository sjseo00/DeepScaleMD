#include "halo_backend.h"
#include "force_calculator_cuda.h"
#include "md_system.h"

namespace {

class HostHaloBackend final : public HaloBackend {
public:
    bool try_gpu_direct_halo(MDSystem&, ForceCalculator*, bool&) override {
        return false;
    }

    void on_post_halo(MDSystem&, ForceCalculator*, bool, bool) override {
    }

    const char* name() const override {
        return "host_mpi_halo";
    }
};

#ifdef USE_CUDA
class CudaResidentHaloBackend final : public HaloBackend {
public:
    bool try_gpu_direct_halo(MDSystem& system,
                             ForceCalculator* force_calc,
                             bool& should_rebuild) override {
        auto* cuda = dynamic_cast<ForceCalculatorCUDA*>(force_calc);
        if (!cuda) {
            throw std::runtime_error(
                "CUDA build requires ForceCalculatorCUDA for GPU-resident halo exchange");
        }
        // Cross-rank ownership migration must run every step, not just on
        // rebuild steps: a particle can drift past its subdomain boundary on
        // any step, and until it is reassigned to its new owning rank it is
        // computed with stale local-frame coordinates on the old rank while
        // the new rank only ever sees it as a ghost -- missing/duplicated
        // near-boundary interactions that get worse the longer migration is
        // skipped. This mirrors the host path's migrate_particles_check(),
        // which runs unconditionally in MDSystem::sync_distributed_particles()
        // for exactly this reason. Ownership changes invalidate the halo, so
        // this must happen before exchange_halo_on_device() below.
        if (system.migrate_particles_on_device(*cuda)) {
            should_rebuild = true;
        }
        // A capacity change invalidates all cached device neighbor indices.
        if (system.exchange_halo_on_device(*cuda, should_rebuild)) {
            should_rebuild = true;
        }
        return true;
    }

    void on_post_halo(MDSystem&, ForceCalculator*, bool, bool) override {}
    const char* name() const override { return "dsdecompx_cuda_resident"; }
};
#endif

} // namespace

std::unique_ptr<HaloBackend> create_halo_backend(const std::string& force_calculation_mode,
                                                 ForceCalculator* force_calc) {
    (void)force_calculation_mode;
#ifdef USE_CUDA
    if (dynamic_cast<ForceCalculatorCUDA*>(force_calc)) {
        return std::make_unique<CudaResidentHaloBackend>();
    }
#else
    (void)force_calc;
#endif
    return std::make_unique<HostHaloBackend>();
}
