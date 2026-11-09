#ifndef CONSTRAINT_SOLVER_H
#define CONSTRAINT_SOLVER_H

#include "md_types.h"
#include <vector>

class MDSystem;

// SHAKE (positions) and RATTLE (velocities) on the constraint list of an MDSystem.
//
// Several ranks: the list of a rank holds every constraint of every constraint cluster with an owned atom
// (MDSystem::refresh_local_bonded_topology), the cluster's other atoms are ghosts. The caller refreshes the ghost
// positions before apply_shake() and the ghost positions and velocities before apply_rattle(), so each rank sees the
// whole cluster up to date and solves it by itself; both solvers then also move the ghost copies (needed for the
// iteration to converge on coupled constraints), which the next halo exchange overwrites with the owners' values.
class ConstraintSolver {
public:
    ConstraintSolver(int max_iterations = 500,
                      Real position_tolerance = 1e-8,
                      Real velocity_tolerance = 1e-8);

    bool apply_shake(MDSystem& system, Real dt,
                      const std::vector<Real>& pre_step_x,
                      const std::vector<Real>& pre_step_y,
                      const std::vector<Real>& pre_step_z);

    // dt is the integrator step (the virial needs it, see get_virial_diag). ref_vx/vy/vz (all or none): the start-of-run
    // virial measurement, where the velocities are those of a virtual half step and the pair's r.v is to become
    // -dt/2 |v_ref|^2 instead of 0, with v_ref the velocities before the kick (see ForceCalculatorCUDA::
    // apply_rattle_on_device).
    bool apply_rattle(MDSystem& system, Real dt, const Real* ref_vx = nullptr, const Real* ref_vy = nullptr,
                      const Real* ref_vz = nullptr);

    // After a barostat step scaled every position by mu[a] along axis a (the box of `system` is the new one): puts the
    // atoms of every constraint cluster back to their rigid shape around the scaled centre of mass. Without it every
    // bond is off by a factor of about mu, and the next SHAKE takes the restoring displacement for a constraint force.
    void restore_rigid_clusters(MDSystem& system, const Real mu[3]);

    // Diagonal of the virial tensor of the constraint forces of the last apply_rattle() on this rank, sum over its
    // constraints c of r_c,a G_c,a (kcal/mol, r_c the bond vector at the end of the step, G_c the force that keeps the
    // pair at its length, from the velocity correction over the last half step). This belongs to the time of the
    // forces and the kinetic energy of the step, which the SHAKE force of the step (the force of the step's start)
    // does not. A constraint counts on the rank that owns its atom of the lower global id, so the ranks' values add up
    // to the system's. The trace is the scalar virial of the constraints.
    void get_virial_diag(Real out[3]) const {
        for (int a = 0; a < 3; ++a) out[a] = virial_diag_[a];
    }
    // Puts back a virial read with get_virial_diag (a velocity projection that is not part of the dynamics, such as the
    // one after a stochastic thermostat, must not replace the step's constraint virial).
    void set_virial_diag(const Real virial[3]) {
        for (int a = 0; a < 3; ++a) virial_diag_[a] = virial[a];
    }

private:
    int max_iterations_;
    Real position_tolerance_;   // relative to distance^2
    Real velocity_tolerance_;   // absolute, in the position*velocity units of r.v
    Real virial_diag_[3] = {0.0, 0.0, 0.0};
};

#endif // CONSTRAINT_SOLVER_H
