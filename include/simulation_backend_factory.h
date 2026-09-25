#ifndef SIMULATION_BACKEND_FACTORY_H
#define SIMULATION_BACKEND_FACTORY_H

#include "md_types.h"
#include "force_calculator.h"
#include "integrator.h"
#include "thermostat.h"
#include "barostat.h"
#include <memory>

const char* backend_label();

std::unique_ptr<ForceCalculator> create_force_calculator_for_build(int rank);

std::unique_ptr<Integrator> create_velocity_verlet_integrator_for_build(Real dt);
std::unique_ptr<Integrator> create_leap_frog_integrator_for_build(Real dt);

std::unique_ptr<Thermostat> create_nose_hoover_thermostat_for_build(Real target_temperature, Real q);
std::unique_ptr<Thermostat> create_langevin_thermostat_for_build(Real target_temperature, Real gamma);
std::unique_ptr<Thermostat> create_andersen_thermostat_for_build(Real target_temperature, Real collision_freq);

std::unique_ptr<Barostat> create_parrinello_rahman_barostat_for_build(Real target_pressure, Real w,
                                                                       Real gamma = 0.0, Real piston_temp = 0.0);
std::unique_ptr<Barostat> create_andersen_barostat_for_build(Real target_pressure, Real compressibility);
std::unique_ptr<Barostat> create_berendsen_barostat_for_build(Real target_pressure, Real tau_p);

#endif
