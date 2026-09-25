#include "../include/constraint_solver.h"
#include "../include/md_system.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

ConstraintSolver::ConstraintSolver(int max_iterations, Real position_tolerance, Real velocity_tolerance)
    : max_iterations_(max_iterations),
      position_tolerance_(position_tolerance),
      velocity_tolerance_(velocity_tolerance) {}

bool ConstraintSolver::apply_shake(MDSystem& system, Real dt,
                                    const std::vector<Real>& pre_step_x,
                                    const std::vector<Real>& pre_step_y,
                                    const std::vector<Real>& pre_step_z) {
    const auto& constraints = system.get_local_constraints();
    if (constraints.empty()) return true;

    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    const size_t local_owned = system.get_local_particles();
    const Real inv_dt = 1.0 / dt;

    const char* allow_unsafe = std::getenv("MD_ALLOW_UNSAFE_CROSSRANK_CONSTRAINTS");
    const bool allow_unsafe_crossrank = (allow_unsafe != nullptr) && (std::string(allow_unsafe) == "1");
    static bool warned_cross_rank_shake = false;
    size_t skipped_cross_rank = 0;

    for (int iter = 0; iter < max_iterations_; ++iter) {
        Real max_relative_error = 0.0;

        for (const auto& c : constraints) {
            const size_t i = c.atom1;
            const size_t j = c.atom2;
            const Real d2 = c.distance * c.distance;

            if (i >= local_owned || j >= local_owned) {
                ++skipped_cross_rank;
                continue;
            }

            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            box.apply_minimum_image(dx, dy, dz);

            const Real diff = (dx * dx + dy * dy + dz * dz) - d2;
            const Real relative_error = std::abs(diff) / d2;
            max_relative_error = std::max(max_relative_error, relative_error);
            if (relative_error < position_tolerance_) continue;

            Real ox = pre_step_x[i] - pre_step_x[j];
            Real oy = pre_step_y[i] - pre_step_y[j];
            Real oz = pre_step_z[i] - pre_step_z[j];
            box.apply_minimum_image(ox, oy, oz);

            const Real g_dot = ox * dx + oy * dy + oz * dz;
            if (std::abs(g_dot) < 1e-12) continue; // Degenerate geometry; retry next sweep.

            const Real inv_mi = 1.0 / particles.mass[i];
            const Real inv_mj = 1.0 / particles.mass[j];
            const Real g = diff / (2.0 * (inv_mi + inv_mj) * g_dot);

            const Real corr_x = g * ox;
            const Real corr_y = g * oy;
            const Real corr_z = g * oz;

            particles.x[i] -= corr_x * inv_mi;
            particles.y[i] -= corr_y * inv_mi;
            particles.z[i] -= corr_z * inv_mi;
            particles.x[j] += corr_x * inv_mj;
            particles.y[j] += corr_y * inv_mj;
            particles.z[j] += corr_z * inv_mj;

            particles.vx[i] -= corr_x * inv_mi * inv_dt;
            particles.vy[i] -= corr_y * inv_mi * inv_dt;
            particles.vz[i] -= corr_z * inv_mi * inv_dt;
            particles.vx[j] += corr_x * inv_mj * inv_dt;
            particles.vy[j] += corr_y * inv_mj * inv_dt;
            particles.vz[j] += corr_z * inv_mj * inv_dt;
        }

        if (skipped_cross_rank > 0 && !warned_cross_rank_shake) {
            if (!allow_unsafe_crossrank) {
                throw std::runtime_error(
                    "[Constraints] MPI cross-rank constraints detected (ghost endpoint involved). "
                    "Current CPU SHAKE/RATTLE path cannot safely enforce these in distributed mode. "
                    "Use fewer MPI ranks so constrained partners stay local, disable constraints, "
                    "or set MD_ALLOW_UNSAFE_CROSSRANK_CONSTRAINTS=1 to continue with skipped cross-rank constraints.");
            } else {
                std::cerr << "[Constraints][rank " << system.get_rank()
                          << "] SHAKE skipped " << skipped_cross_rank
                          << " cross-rank constraint(s) (ghost endpoint involved)."
                          << std::endl;
                warned_cross_rank_shake = true;
            }
        }

        if (max_relative_error < position_tolerance_) return true;
    }

    return false; // Did not converge within max_iterations_.
}

bool ConstraintSolver::apply_rattle(MDSystem& system) {
    const auto& constraints = system.get_local_constraints();
    if (constraints.empty()) return true;

    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    const size_t local_owned = system.get_local_particles();

    const char* allow_unsafe = std::getenv("MD_ALLOW_UNSAFE_CROSSRANK_CONSTRAINTS");
    const bool allow_unsafe_crossrank = (allow_unsafe != nullptr) && (std::string(allow_unsafe) == "1");
    static bool warned_cross_rank_rattle = false;
    size_t skipped_cross_rank = 0;

    for (int iter = 0; iter < max_iterations_; ++iter) {
        Real max_error = 0.0;

        for (const auto& c : constraints) {
            const size_t i = c.atom1;
            const size_t j = c.atom2;

            if (i >= local_owned || j >= local_owned) {
                ++skipped_cross_rank;
                continue;
            }

            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            box.apply_minimum_image(dx, dy, dz);
            const Real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) continue;

            const Real dvx = particles.vx[i] - particles.vx[j];
            const Real dvy = particles.vy[i] - particles.vy[j];
            const Real dvz = particles.vz[i] - particles.vz[j];

            const Real k_dot = dx * dvx + dy * dvy + dz * dvz;
            max_error = std::max(max_error, std::abs(k_dot));
            if (std::abs(k_dot) < velocity_tolerance_) continue;

            const Real inv_mi = 1.0 / particles.mass[i];
            const Real inv_mj = 1.0 / particles.mass[j];
            const Real k = k_dot / ((inv_mi + inv_mj) * r2);

            particles.vx[i] -= k * inv_mi * dx;
            particles.vy[i] -= k * inv_mi * dy;
            particles.vz[i] -= k * inv_mi * dz;
            particles.vx[j] += k * inv_mj * dx;
            particles.vy[j] += k * inv_mj * dy;
            particles.vz[j] += k * inv_mj * dz;
        }

        if (skipped_cross_rank > 0 && !warned_cross_rank_rattle) {
            if (!allow_unsafe_crossrank) {
                throw std::runtime_error(
                    "[Constraints] MPI cross-rank constraints detected (ghost endpoint involved). "
                    "Current CPU SHAKE/RATTLE path cannot safely enforce these in distributed mode. "
                    "Use fewer MPI ranks so constrained partners stay local, disable constraints, "
                    "or set MD_ALLOW_UNSAFE_CROSSRANK_CONSTRAINTS=1 to continue with skipped cross-rank constraints.");
            } else {
                std::cerr << "[Constraints][rank " << system.get_rank()
                          << "] RATTLE skipped " << skipped_cross_rank
                          << " cross-rank constraint(s) (ghost endpoint involved)."
                          << std::endl;
                warned_cross_rank_rattle = true;
            }
        }

        if (max_error < velocity_tolerance_) return true;
    }

    return false; // Did not converge within max_iterations_.
}
