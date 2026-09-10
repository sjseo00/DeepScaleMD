#ifndef FORCE_CALCULATOR_CPU_H
#define FORCE_CALCULATOR_CPU_H

#include "force_calculator.h"
#include "md_system.h"
#include <vector>
#include <fftw3.h>

class ForceCalculatorCPU : public ForceCalculator {
public:
    ForceCalculatorCPU() = default;
    virtual ~ForceCalculatorCPU();

    virtual void initialize() override;
    virtual void finalize() override;
    virtual void compute_forces(MDSystem& system) override;
    virtual void calculate_kinetic_energy(MDSystem& system) override;

private:
    void compute_bond_forces(MDSystem& system);
    void compute_angle_forces(MDSystem& system);
    void compute_dihedral_forces(MDSystem& system);
    void compute_improper_forces(MDSystem& system);
    // Shared by compute_dihedral_forces/compute_improper_forces: propers and
    // impropers use the identical periodic-torsion functional form and
    // analytical gradient, just over different atom-quadruple lists.
    void compute_torsion_forces(MDSystem& system, const std::vector<MDSystem::LocalDihedral>& torsions);
    void compute_lj_forces(MDSystem& system);
    // Scaled 1-4 (or FF-specific) nonbonded pairs re-added on top of
    // [exclusions]; see Topology::SpecialPair for the exact semantics.
    void compute_special_pair_forces(MDSystem& system);

    // Real-space (erfc-screened) half of the Ewald sum, evaluated over the
    // existing neighbor list.
    void compute_electrostatic_forces_real_space(MDSystem& system);
    // Full Ewald summation: real-space + reciprocal-space (FFTW PME grid) +
    // self-energy correction. Replaces the old direct-Coulomb-only path.
    void compute_electrostatic_forces_ewald(MDSystem& system);
    void ensure_ewald_initialized(MDSystem& system);
    void free_ewald_resources();

    // Thread-private reverse-force buffers for Newton ON (avoids atomic writes)
    // Sized nthreads * local_particles; reused across calls to avoid reallocation.
    std::vector<Real> rfx_priv, rfy_priv, rfz_priv;

    // ===== EWALD / PME RECIPROCAL-SPACE ELECTROSTATICS (CPU, FFTW) =====
    bool ewald_initialized_ = false;
    int ewald_grid_x_ = 0, ewald_grid_y_ = 0, ewald_grid_z_ = 0;
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;
    Real ewald_prefactor_ = 0.0;  // 4*pi*k_coulomb / box_volume

    // Full complex-to-complex (C2C) grid, charge density in place becomes
    // Phi(k) after the Poisson solve (see .cpp for why C2C over R2C/C2R).
    fftw_complex* pme_charge_grid_ = nullptr;
    fftw_complex* pme_ex_grid_ = nullptr;
    fftw_complex* pme_ey_grid_ = nullptr;
    fftw_complex* pme_ez_grid_ = nullptr;
    Real* pme_force_grid_x_ = nullptr;
    Real* pme_force_grid_y_ = nullptr;
    Real* pme_force_grid_z_ = nullptr;
    std::vector<Real> pme_b_spline_mod_x_, pme_b_spline_mod_y_, pme_b_spline_mod_z_;
    fftw_plan pme_fft_plan_fwd_ = nullptr;
    fftw_plan pme_fft_plan_bwd_ = nullptr;  // reused for all 3 inverse transforms
};

#endif // FORCE_CALCULATOR_CPU_H
