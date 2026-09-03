#include "simulation_backend_factory.h"
#include "force_calculator_cuda.h"
#include "force_calculator_cpu.h"
#include <iostream>

const char* backend_label() {
#ifdef USE_SYCL
    return "SYCL";
#elif defined(USE_CUDA)
    return "CUDA";
#else
    return "CPU";
#endif
}

std::unique_ptr<ForceCalculator> create_force_calculator_for_build(int rank) {
#ifdef USE_SYCL
    auto fc = std::make_unique<ForceCalculatorSYCL>();
    if (rank == 0) {
        std::cout << "Using SYCL for force calculations" << std::endl;
    }
    fc->initialize();
    return fc;
#else
    try {
        auto fc = std::make_unique<ForceCalculatorCUDA>();
        fc->initialize();
        if (rank == 0) {
            std::cout << "Using CUDA for force calculations (GPU-resident mode)" << std::endl;
        }
        return fc;
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cout << "CUDA initialization failed: " << e.what() << std::endl;
            std::cout << "Falling back to CPU force calculations" << std::endl;
        }
        auto fc = std::make_unique<ForceCalculatorCPU>();
        fc->initialize();
        return fc;
    }
#endif
}

std::unique_ptr<Integrator> create_velocity_verlet_integrator_for_build(Real dt) {
#ifdef USE_SYCL
    return std::make_unique<VelocityVerletIntegratorSYCL>(dt);
#elif defined(USE_CUDA)
    return std::make_unique<VelocityVerletIntegratorCUDA>(dt);
#else
    return std::make_unique<VelocityVerletIntegrator>(dt);
#endif
}

std::unique_ptr<Integrator> create_leap_frog_integrator_for_build(Real dt) {
#ifdef USE_SYCL
    return std::make_unique<LeapFrogIntegratorSYCL>(dt);
#elif defined(USE_CUDA)
    return std::make_unique<LeapFrogIntegratorCUDA>(dt);
#else
    return std::make_unique<LeapFrogIntegrator>(dt);
#endif
}

std::unique_ptr<Thermostat> create_nose_hoover_thermostat_for_build(Real target_temperature, Real q) {
#ifdef USE_SYCL
    return std::make_unique<NoseHooverThermostatSYCL>(target_temperature, q);
#elif defined(USE_CUDA)
    return std::make_unique<NoseHooverThermostatCUDA>(target_temperature, q);
#else
    return std::make_unique<NoseHooverThermostat>(target_temperature, q);
#endif
}

std::unique_ptr<Thermostat> create_langevin_thermostat_for_build(Real target_temperature, Real gamma) {
#ifdef USE_SYCL
    return std::make_unique<LangevinThermostatSYCL>(target_temperature, gamma);
#elif defined(USE_CUDA)
    return std::make_unique<LangevinThermostatCUDA>(target_temperature, gamma);
#else
    return std::make_unique<LangevinThermostat>(target_temperature, gamma);
#endif
}

std::unique_ptr<Thermostat> create_andersen_thermostat_for_build(Real target_temperature, Real collision_freq) {
#ifdef USE_SYCL
    return std::make_unique<AndersenThermostatSYCL>(target_temperature, collision_freq);
#elif defined(USE_CUDA)
    return std::make_unique<AndersenThermostatCUDA>(target_temperature, collision_freq);
#else
    return std::make_unique<AndersenThermostat>(target_temperature, collision_freq);
#endif
}

std::unique_ptr<Barostat> create_parrinello_rahman_barostat_for_build(Real target_pressure, Real w) {
#ifdef USE_SYCL
    return std::make_unique<ParrinelloRahmanBarostatSYCL>(target_pressure, w);
#elif defined(USE_CUDA)
    return std::make_unique<ParrinelloRahmanBarostatCUDA>(target_pressure, w);
#else
    return std::make_unique<ParrinelloRahmanBarostat>(target_pressure, w);
#endif
}

std::unique_ptr<Barostat> create_andersen_barostat_for_build(Real target_pressure, Real compressibility) {
#ifdef USE_SYCL
    return std::make_unique<AndersenBarostatSYCL>(target_pressure, compressibility);
#elif defined(USE_CUDA)
    return std::make_unique<AndersenBarostatCUDA>(target_pressure, compressibility);
#else
    return std::make_unique<AndersenBarostat>(target_pressure, compressibility);
#endif
}
