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
    // (e.g. python) that don't run the builtin PME path.
    virtual void set_pme_grid_spacing(Real /*spacing_angstrom*/) {}

    virtual Real get_potential_energy() const { return last_potential_energy; }
    virtual Real get_kinetic_energy() const { return last_kinetic_energy; }
    virtual Real get_virial() const { return last_virial; }
    // The MLIP's contribution to the virial of this rank, added to the value the last force phase read back. Call it
    // only on a step whose force phase read the energy and virial back (it adds to that value).
    void add_external_virial(Real virial) { last_virial += virial; }

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
