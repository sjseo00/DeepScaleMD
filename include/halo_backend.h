#ifndef HALO_BACKEND_H
#define HALO_BACKEND_H

#include <memory>
#include <string>

class MDSystem;
class ForceCalculator;

class HaloBackend {
public:
    virtual ~HaloBackend() = default;

    virtual bool try_gpu_direct_halo(MDSystem& system,
                                     ForceCalculator* force_calc,
                                     bool& should_rebuild) = 0;

    virtual void on_post_halo(MDSystem& system,
                              ForceCalculator* force_calc,
                              bool should_rebuild,
                              bool used_gpu_direct_halo) = 0;

    virtual const char* name() const = 0;
};

std::unique_ptr<HaloBackend> create_halo_backend(const std::string& force_calculation_mode,
                                                 ForceCalculator* force_calc);

#endif
