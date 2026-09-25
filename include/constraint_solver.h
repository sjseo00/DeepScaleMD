#ifndef CONSTRAINT_SOLVER_H
#define CONSTRAINT_SOLVER_H

#include "md_types.h"
#include <vector>

class MDSystem;

class ConstraintSolver {
public:
    ConstraintSolver(int max_iterations = 500,
                      Real position_tolerance = 1e-8,
                      Real velocity_tolerance = 1e-8);

    bool apply_shake(MDSystem& system, Real dt,
                      const std::vector<Real>& pre_step_x,
                      const std::vector<Real>& pre_step_y,
                      const std::vector<Real>& pre_step_z);

    bool apply_rattle(MDSystem& system);

private:
    int max_iterations_;
    Real position_tolerance_;   // relative to distance^2
    Real velocity_tolerance_;   // absolute, in the position*velocity units of r.v
};

#endif // CONSTRAINT_SOLVER_H
