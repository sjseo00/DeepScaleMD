#ifndef FORCE_CALCULATOR_H
#define FORCE_CALCULATOR_H

#include "md_types.h"
#include <map>
#include <string>
#include <vector>

class MDSystem;
class NeighborList;

class ForceCalculator {
protected:
    Real last_potential_energy = 0.0;   // Last computed PE
    Real last_kinetic_energy = 0.0;     // Last computed KE
    Real last_virial = 0.0;             // Last computed virial (for pressure calculation)
    // Diagonal of the virial tensor of this rank, W_aa = sum r_a F_a (kcal/mol), and of the kinetic energy tensor,
    // KE_a = sum 0.5 m v_a^2 (the same raw unit as last_kinetic_energy). Only filled while the tensor is enabled
    // (an anisotropic barostat needs P_xx, P_yy, P_zz); their sums are last_virial and last_kinetic_energy.
    Real last_virial_diag[3] = {0.0, 0.0, 0.0};
    Real last_kinetic_energy_diag[3] = {0.0, 0.0, 0.0};
    bool virial_tensor_enabled_ = false;

public:
    virtual ~ForceCalculator() = default;
    virtual void compute_forces(MDSystem& system) = 0;
    virtual void initialize() = 0;
    virtual void finalize() = 0;
    virtual void set_nonbonded_kernel(const std::string& kernel) {
        nonbonded_kernel_ = kernel;
    }
    const std::string& get_nonbonded_kernel() const { return nonbonded_kernel_; }

    virtual void set_mixed_precision(bool /*enabled*/) {}

    // force_calculation.pme_grid_spacing (A); 0 = automatic. No-op on backends
    // (e.g. mlip) that don't run the builtin PME path.
    virtual void set_pme_grid_spacing(Real /*spacing_angstrom*/) {}

    virtual Real get_potential_energy() const { return last_potential_energy; }
    virtual Real get_kinetic_energy() const { return last_kinetic_energy; }
    virtual Real get_virial() const { return last_virial; }
    // The MLIP's contribution to the virial of this rank, added to the value the last force phase read back. Call it
    // only on a step whose force phase read the energy and virial back (it adds to that value).
    void add_external_virial(Real virial) { last_virial += virial; }
    // The same with the diagonal W_xx, W_yy, W_zz (while the virial tensor is enabled): adds them to the diagonal and their sum to
    // the trace.
    void add_external_virial_diag(const Real virial[3]) {
        for (int a = 0; a < 3; ++a) {
            last_virial_diag[a] += virial[a];
            last_virial += virial[a];
        }
    }

    // Virial and kinetic energy tensors (diagonals) for anisotropic pressure coupling.
    //   supports_virial_tensor()      the backend can give W_aa and KE_a (false: the default, e.g. SYCL, MLIP-only)
    //   set_virial_tensor_enabled()   switch the extra accumulation on; off (the default) leaves every kernel and every
    //                                 result exactly as before, so isotropic runs pay nothing
    //   get_virial_diag()             this rank's W_xx, W_yy, W_zz of the last force phase that read the virial back;
    //                                 summed over the three it equals get_virial() (the caller adds the ranks up, as for
    //                                 the scalar virial)
    //   get_kinetic_energy_diag()     the global KE_x, KE_y, KE_z computed by the last calculate_kinetic_energy() (reduced
    //                                 over the ranks like get_kinetic_energy(), raw unit: multiply by mvv2e)
    virtual bool supports_virial_tensor() const { return false; }
    virtual void set_virial_tensor_enabled(bool enabled) { virial_tensor_enabled_ = enabled; }
    bool virial_tensor_enabled() const { return virial_tensor_enabled_; }
    virtual void get_virial_diag(Real out[3]) const {
        for (int a = 0; a < 3; ++a) out[a] = last_virial_diag[a];
    }
    virtual void get_kinetic_energy_diag(Real out[3]) const {
        for (int a = 0; a < 3; ++a) out[a] = last_kinetic_energy_diag[a];
    }

    virtual void calculate_kinetic_energy(MDSystem& system) = 0;

    struct ForceComponentTiming { std::string name; double ms; };
    virtual void set_timing_breakdown_enabled(bool /*enabled*/) {}
    virtual bool is_timing_breakdown_enabled() const { return false; }
    virtual const std::vector<ForceComponentTiming>& get_last_force_component_timings() const {
        static const std::vector<ForceComponentTiming> empty_timings;
        return empty_timings;
    }

    virtual const std::vector<int>& get_lj_pair_counts() const {
        static const std::vector<int> empty_counts;
        return empty_counts;
    }

protected:
    std::string nonbonded_kernel_ = "tiled";
};

#endif // FORCE_CALCULATOR_H
