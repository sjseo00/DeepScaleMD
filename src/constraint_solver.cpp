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
    const Real inv_dt = 1.0 / dt;

    for (int iter = 0; iter < max_iterations_; ++iter) {
        Real max_relative_error = 0.0;

        for (const auto& c : constraints) {
            const size_t i = c.atom1;
            const size_t j = c.atom2;
            const Real d2 = c.distance * c.distance;

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

            // Both atoms move, a ghost atom too: its owner does the same on its side, and the copy here is only
            // needed so that the next constraint of a coupled cluster sees the corrected position.
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

        if (max_relative_error < position_tolerance_) return true;
    }

    return false; // Did not converge within max_iterations_.
}

bool ConstraintSolver::apply_rattle(MDSystem& system, Real dt, const Real* ref_vx, const Real* ref_vy,
                                    const Real* ref_vz) {
    virial_diag_[0] = virial_diag_[1] = virial_diag_[2] = 0.0;
    const auto& constraints = system.get_local_constraints();
    if (constraints.empty()) return true;

    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    const size_t local_owned = system.get_local_particles();
    // The velocity correction of the first atom of a pair is -k r / m; the force that did it over the half step dt/2
    // is G = m dv / (FORCE_TO_ACCEL dt/2) = -2 k r / (FORCE_TO_ACCEL dt), and the pair's virial r_a G_a is
    // virial_prefactor k r_a^2. The positions do not change during the velocity iteration, so every correction of a
    // constraint uses the same r and the sum over the iterations is the total.
    const Real virial_prefactor = -2.0 / (FORCE_TO_ACCEL * dt);

    for (int iter = 0; iter < max_iterations_; ++iter) {
        Real max_error = 0.0;

        for (const auto& c : constraints) {
            const size_t i = c.atom1;
            const size_t j = c.atom2;

            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];
            box.apply_minimum_image(dx, dy, dz);
            const Real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-20) continue;

            const Real dvx = particles.vx[i] - particles.vx[j];
            const Real dvy = particles.vy[i] - particles.vy[j];
            const Real dvz = particles.vz[i] - particles.vz[j];

            Real target = 0.0;
            if (ref_vx) {
                const Real rx = ref_vx[i] - ref_vx[j], ry = ref_vy[i] - ref_vy[j], rz = ref_vz[i] - ref_vz[j];
                target = -0.5 * dt * (rx * rx + ry * ry + rz * rz);
            }
            const Real k_dot = dx * dvx + dy * dvy + dz * dvz - target;
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

            // Counted once: by the rank that owns the atom of the lower global id.
            const size_t lower = (particles.global_id[i] < particles.global_id[j]) ? i : j;
            if (lower < local_owned) {
                virial_diag_[0] += virial_prefactor * k * dx * dx;
                virial_diag_[1] += virial_prefactor * k * dy * dy;
                virial_diag_[2] += virial_prefactor * k * dz * dz;
            }
        }

        if (max_error < velocity_tolerance_) return true;
    }

    return false; // Did not converge within max_iterations_.
}

void ConstraintSolver::restore_rigid_clusters(MDSystem& system, const Real mu[3]) {
    const auto& constraints = system.get_local_constraints();
    if (constraints.empty()) return;
    ParticleData& particles = system.get_particles();
    const SimulationBox& box = system.get_box();

    // The constraint graph over the atom slots (CSR), walked cluster by cluster from its first atom: every atom is put
    // next to the atom it is bonded to by the minimum image, so a cluster that straddles the periodic boundary is one
    // piece.
    size_t slots = 0;
    for (const auto& c : constraints) slots = std::max({slots, c.atom1 + 1, c.atom2 + 1});
    std::vector<size_t> first(slots + 1, 0);
    for (const auto& c : constraints) {
        ++first[c.atom1 + 1];
        ++first[c.atom2 + 1];
    }
    for (size_t a = 0; a < slots; ++a) first[a + 1] += first[a];
    std::vector<size_t> neighbor(2 * constraints.size()), cursor(first.begin(), first.end() - 1);
    for (const auto& c : constraints) {
        neighbor[cursor[c.atom1]++] = c.atom2;
        neighbor[cursor[c.atom2]++] = c.atom1;
    }

    const Real shrink[3] = {1.0 - 1.0 / mu[0], 1.0 - 1.0 / mu[1], 1.0 - 1.0 / mu[2]};
    std::vector<char> done(slots, 0);
    std::vector<size_t> members;
    std::vector<Real> ux, uy, uz;
    for (size_t seed = 0; seed < slots; ++seed) {
        if (done[seed] || first[seed] == first[seed + 1]) continue;
        members.assign(1, seed);
        ux.assign(1, particles.x[seed]);
        uy.assign(1, particles.y[seed]);
        uz.assign(1, particles.z[seed]);
        done[seed] = 1;
        for (size_t head = 0; head < members.size(); ++head) {
            const size_t a = members[head];
            for (size_t e = first[a]; e < first[a + 1]; ++e) {
                const size_t b = neighbor[e];
                if (done[b]) continue;
                Real dx = particles.x[b] - particles.x[a];
                Real dy = particles.y[b] - particles.y[a];
                Real dz = particles.z[b] - particles.z[a];
                box.apply_minimum_image(dx, dy, dz);
                members.push_back(b);
                ux.push_back(ux[head] + dx);
                uy.push_back(uy[head] + dy);
                uz.push_back(uz[head] + dz);
                done[b] = 1;
            }
        }
        Real total = 0.0, comx = 0.0, comy = 0.0, comz = 0.0;
        for (size_t m = 0; m < members.size(); ++m) {
            const Real mass = particles.mass[members[m]];
            total += mass;
            comx += mass * ux[m];
            comy += mass * uy[m];
            comz += mass * uz[m];
        }
        comx /= total; comy /= total; comz /= total;
        for (size_t m = 0; m < members.size(); ++m) {
            particles.x[members[m]] -= shrink[0] * (ux[m] - comx);
            particles.y[members[m]] -= shrink[1] * (uy[m] - comy);
            particles.z[members[m]] -= shrink[2] * (uz[m] - comz);
        }
    }
}
