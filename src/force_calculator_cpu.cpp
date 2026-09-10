
#include "../include/force_calculator_cpu.h"
#include "../include/md_system.h"
#include "../include/topology.h"
#include "../include/mpi_utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <iomanip>
#include <cstring>
#include <fftw3.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ForceCalculatorCPU::~ForceCalculatorCPU() {
    free_ewald_resources();
}

void ForceCalculatorCPU::initialize() {
    if (is_rank_zero()) {
        std::cout << "CPU Force Calculator initialized" << std::endl;
    }
}

void ForceCalculatorCPU::finalize() {
    // Nothing to cleanup for CPU version
}

void ForceCalculatorCPU::compute_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    
    // Clear forces
    std::fill(particles.fx.begin(), particles.fx.end(), 0.0);
    std::fill(particles.fy.begin(), particles.fy.end(), 0.0);
    std::fill(particles.fz.begin(), particles.fz.end(), 0.0);
    
    // Reset energies (last_virial is accumulated by LJ + electrostatics below)
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    last_virial = 0.0;
    
    // Get force field configuration from force manager
    ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    
    // Compute different force contributions based on configuration
    if (force_config.enable_bonds) {
        compute_bond_forces(system);         // Bonded interactions
    }
    if (force_config.enable_angles) {
        compute_angle_forces(system);        // Angle bending forces  
    }
    if (force_config.enable_dihedrals) {
        compute_dihedral_forces(system);     // Torsional forces
        compute_improper_forces(system);     // Improper torsions (same functional form)
    }
    if (force_config.enable_lennard_jones) {
        compute_lj_forces(system);           // Van der Waals forces
    }
    if (force_config.enable_coulomb) {
        compute_electrostatic_forces_ewald(system); // Full Ewald summation (real + reciprocal + self)
    }
    if (force_config.enable_lennard_jones || force_config.enable_coulomb) {
        compute_special_pair_forces(system); // Scaled 1-4 nonbonded pairs
    }

    // MPI-AWARE: Sum potential energy across all ranks
    // Each rank computes PE for pairs it "owns" (global_id[i] < global_id[j]),
    // so we need to sum these partial energies to get the total system PE
    // OPTIMIZATION: Batch the two MPI_Allreduce calls (PE reduction + particle count)
    // into a single call to reduce communication overhead.
    size_t local_owned = system.get_local_particles();
    Real batch_in[2]  = { last_potential_energy, static_cast<Real>(local_owned) };
    Real batch_out[2] = { 0.0, 0.0 };
    MPI_Allreduce(batch_in, batch_out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy = batch_out[0];
    (void)batch_out[1];

    // NOTE: Long-range tail correction is NOT applied here.
    // LAMMPS default: truncated LJ at cutoff with NO tail correction unless
    // "pair_modify tail yes" is explicitly specified.  Applying tail correction
    // would add ~-14.6 kcal/mol to PE (for 32K Ar @ cutoff=12Å), making DS CPU
    // diverge from LAMMPS and DS GPU by that amount.  Tail correction can be
    // enabled separately if needed, but must also be added to the virial/pressure
    // calculation to remain thermodynamically consistent.
}

void ForceCalculatorCPU::compute_lj_forces(MDSystem& system) {
    
    ParticleData& particles = system.get_particles();
    NeighborList& nlist = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
    SimulationBox& box = system.get_box();
    Topology& topology = system.get_topology();
            
    // Use the existing domain decomposition system
    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();
    (void)total_particles;  // used only in Newton-ON branch
    
    // Use physical cutoff, not neighbor list cutoff (which includes skin)
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;
    
    // Reset virial accumulator for pressure calculation
    Real local_virial = 0.0;

    // ======================================================================
    // PERFORMANCE FIXES applied here:
    //
    // 1) virial: W = r·F = r²·force_mag  (no sqrt → removes expensive op)
    //
    // 2) Newton ON half-list: neighbor list stores only j > i for local-local.
    //    Reverse force written directly to particles.fx[j] (SAFE for single thread).
    //    For multi-thread OMP, use omp critical or atomic — but single thread is
    //    the primary use case here (OMP_NUM_THREADS=1).
    // ======================================================================

    #pragma omp parallel for reduction(+:last_potential_energy,local_virial) schedule(dynamic, 64)
    for (size_t i = 0; i < local_particles; ++i) {
        // Load particle i data once (cache-friendly)
        Real xi = particles.x[i];
        Real yi = particles.y[i];
        Real zi = particles.z[i];
        int type_i = particles.type[i];
        int gid_i  = particles.global_id[i];
        
        // Prefetch epsilon/sigma row for this particle type
        const std::vector<Real>& eps_row = params.epsilon_mixed[type_i];
        const std::vector<Real>& sig_row = params.sigma_mixed[type_i];
        
        // Accumulate forces locally to reduce memory writes
        Real fx_acc = 0.0;
        Real fy_acc = 0.0;
        Real fz_acc = 0.0;
        Real pe_acc = 0.0;
        Real virial_acc = 0.0;
        
        size_t neighbor_start = static_cast<size_t>(nlist.neighbor_offset[i]);
        size_t neighbor_count = static_cast<size_t>(nlist.num_neighbors[i]);
        
        const int* neighbor_ptr = nlist.neighbors.data() + neighbor_start;
        
        const Real* __restrict__ px    = particles.x.data();
        const Real* __restrict__ py    = particles.y.data();
        const Real* __restrict__ pz    = particles.z.data();
        const int*  __restrict__ ptype = particles.type.data();
        const int*  __restrict__ pgid  = particles.global_id.data();
        
        for (size_t n = 0; n < neighbor_count; ++n) {
            int ji = neighbor_ptr[n];
            if (ji < 0) continue;
            size_t j = static_cast<size_t>(ji);

            if (topology.is_nonbonded_excluded(gid_i, pgid[j])) continue;
            
            // Half-list guarantees j > i for local-local pairs.
            // No skip check needed here.

            Real dx = xi - px[j];
            Real dy = yi - py[j];
            Real dz = zi - pz[j];
            box.apply_minimum_image(dx, dy, dz);
            Real r2 = dx*dx + dy*dy + dz*dz;
            
            if (r2 >= cutoff_sq || r2 < 1e-10) continue;

            int type_j = ptype[j];
            Real eps = eps_row[type_j];
            Real sig = sig_row[type_j];
            Real r2inv = 1.0 / r2;
            Real attractive;
            Real repulsive;
            if (params.mie_repulsive_exponent == 12) {
                const Real sig2 = sig * sig;
                const Real sig6 = sig2 * sig2 * sig2;
                attractive = sig6 * r2inv * r2inv * r2inv;
                repulsive = attractive * attractive;
            } else {
                const Real sig_r = sig * std::sqrt(r2inv);
                const Real sig3_r3 = sig_r * sig_r * sig_r;
                attractive = sig3_r3 * sig3_r3;
                repulsive = attractive * sig3_r3; // SPICA Mie(9,6)
            }
            Real force_mag = params.mie_prefactor * eps * r2inv *
                (params.mie_repulsive_exponent * repulsive -
                 params.mie_attractive_exponent * attractive);
            Real pe_pair = params.mie_prefactor * eps * (repulsive - attractive);
            if (params.lj_switch_distance > 0.0 &&
                r2 > params.lj_switch_distance * params.lj_switch_distance) {
                const Real switch_sq = params.lj_switch_distance * params.lj_switch_distance;
                const Real cutoff_delta = cutoff_sq - switch_sq;
                const Real a = cutoff_sq - r2;
                const Real b = cutoff_sq + 2.0 * r2 - 3.0 * switch_sq;
                const Real denom = cutoff_delta * cutoff_delta * cutoff_delta;
                const Real switch_value = a * a * b / denom;
                const Real force_correction = 12.0 * pe_pair * a * (r2 - switch_sq) / denom;
                force_mag = switch_value * force_mag + force_correction;
                pe_pair *= switch_value;
            }
            // Virial FIX: W = r·F = r²*force_mag (no sqrt, correct formula)
            Real virial_pair = r2 * force_mag;

            // Count energy/virial once per pair
            bool j_is_local = (j < local_particles);
            if (j_is_local) {
                pe_acc     += pe_pair;
                virial_acc += virial_pair;
            } else {
                if (gid_i < pgid[j]) {
                    pe_acc     += pe_pair;
                    virial_acc += virial_pair;
                }
            }

            Real fxi = force_mag * dx;
            Real fyi = force_mag * dy;
            Real fzi = force_mag * dz;

            fx_acc += fxi;
            fy_acc += fyi;
            fz_acc += fzi;
            
            // NEWTON ON: apply reverse force to local j
            // Safe for single-thread (OMP_NUM_THREADS=1), race-free because
            // half-list has j > i and only one thread processes i at a time.
            if (j_is_local) {
                particles.fx[j] -= fxi;
                particles.fy[j] -= fyi;
                particles.fz[j] -= fzi;
            }
        }
        
        // Write accumulated forces once (reduced memory access)
        particles.fx[i] += fx_acc;
        particles.fy[i] += fy_acc;
        particles.fz[i] += fz_acc;
        last_potential_energy += pe_acc;
        local_virial          += virial_acc;
    }
    
    // Store virial for pressure calculation.
    // virial_pair = r² * force_mag, where force_mag is in kcal/(mol·Å²) and r² in Å²,
    // so virial has units kcal/mol — same as the GPU path.  Do NOT multiply by mvv2e.
    // Accumulate (not assign): electrostatics also contributes to last_virial,
    // and last_virial is reset to 0 once per step in compute_forces().
    last_virial += local_virial;
    
    // === [DEBUG VIRIAL] Periodic reporting ===
    // OPTIMIZATION: Disabled to reduce MPI overhead (was causing extra Allreduce every 100 steps)
    // Uncomment for debugging virial calculations
    /*
    static int virial_report_count = 0;
    if (virial_report_count % 100 == 0) {
        Real global_virial = 0.0;
        MPI_Allreduce(&last_virial, &global_virial, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "[DEBUG VIRIAL] virial_global=" << global_virial 
                      << " kcal/mol (local=" << last_virial << ")" << std::endl;
        }
    }
    virial_report_count++;
    */
    
}

// Real-space (near-field) half of the Ewald sum: erfc(alpha*r)-screened
// Coulomb interaction over the neighbor list. Must always be combined with
// the reciprocal-space PME grid contribution (compute_electrostatic_forces_ewald)
// -- together they reproduce the exact 1/r Coulomb sum over all periodic images.
void ForceCalculatorCPU::compute_electrostatic_forces_real_space(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    NeighborList& nlist = system.get_neighbor_list();
    SimulationBox& box = system.get_box();
    Topology& topology = system.get_topology();

    size_t local_particles = system.get_local_particles();
    size_t total_particles = system.get_num_particles();

    // Use physical cutoff, not neighbor list cutoff (which includes skin)
    Real physical_cutoff = system.get_physical_cutoff();
    Real cutoff_sq = physical_cutoff * physical_cutoff;
    const Real k_coulomb = 332.0636; // kcal/mol·Å·e^-2 (conversion factor)
    const Real alpha = ewald_alpha_;
    const Real two_alpha_over_sqrt_pi = 1.1283791670955126 * alpha; // 2/sqrt(pi) * alpha

    Real local_pe = 0.0;
    Real local_virial = 0.0;

    // Use neighbor list for efficient force calculation
    for (size_t i = 0; i < local_particles; i++) {
        Real qi = particles.charge[i];
        if (std::abs(qi) < 1e-10) continue; // Skip neutral particles

        int gid_i = particles.global_id[i];

        // Get neighbor list for particle i using pre-computed offsets
        size_t neighbor_start = nlist.neighbor_offset[i];
        size_t neighbor_end = neighbor_start + nlist.num_neighbors[i];

        for (size_t n_idx = neighbor_start; n_idx < neighbor_end; n_idx++) {
            if (n_idx >= nlist.neighbors.size()) break;

            size_t j = nlist.neighbors[n_idx];
            if (j >= total_particles) continue;
            const bool excluded = topology.is_nonbonded_excluded(
                gid_i, particles.global_id[j]);

            Real qj = particles.charge[j];
            if (std::abs(qj) < 1e-10) continue; // Skip neutral particles

            Real dx = particles.x[i] - particles.x[j];
            Real dy = particles.y[i] - particles.y[j];
            Real dz = particles.z[i] - particles.z[j];

            // Apply periodic boundary conditions
            box.apply_minimum_image(dx, dy, dz);

            Real r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < cutoff_sq && r2 > 0.36) { // Minimum distance ~0.6 Å to avoid overlaps
                Real r = std::sqrt(r2);
                Real qq = k_coulomb * qi * qj;
                Real erf_ar = std::erf(alpha * r);
                Real erfc_ar = 1.0 - erf_ar;
                Real exp_term = std::exp(-alpha * alpha * r2);
                // PME reciprocal space contains the erf(alpha*r)/r part for
                // every charged pair.  For an excluded pair, add its negative
                // here so real + reciprocal is exactly zero.  Normal pairs
                // receive the usual erfc real-space contribution.
                Real u_pair;
                Real force_mag;
                if (excluded) {
                    u_pair = -qq * erf_ar / r;
                    force_mag = -qq * (erf_ar / (r2 * r) -
                        two_alpha_over_sqrt_pi * exp_term / r2);
                } else {
                    u_pair = qq * erfc_ar / r;
                    force_mag = qq * (erfc_ar / (r2 * r) +
                        two_alpha_over_sqrt_pi * exp_term / r2);
                }

                Real fx = force_mag * dx;
                Real fy = force_mag * dy;
                Real fz = force_mag * dz;

                particles.fx[i] += fx;
                particles.fy[i] += fy;
                particles.fz[i] += fz;

                // NOTE: this shares system.get_neighbor_list() with compute_lj_forces,
                // which builds a Newton-ON HALF list on CPU/SYCL (local-local pairs
                // stored only once, as j > i -- see VerletCellListNeighborBuilder's
                // use_half_list). So, matching compute_lj_forces exactly: local-local
                // pairs get the FULL pair energy/virial (not halved, since each such
                // pair is visited exactly once) plus an explicit reverse force on j;
                // local-ghost pairs use the global-id ownership rule instead (that
                // pair is visited once per rank, so no reverse force is needed --
                // the ghost's force is recovered via halo exchange).
                bool j_is_local = (j < local_particles);
                int gid_j = particles.global_id[j];
                if (j_is_local) {
                    local_pe += u_pair;
                    local_virial += (dx * fx + dy * fy + dz * fz);
                    particles.fx[j] -= fx;
                    particles.fy[j] -= fy;
                    particles.fz[j] -= fz;
                } else if (gid_i < gid_j) {
                    local_pe += u_pair;
                    local_virial += (dx * fx + dy * fy + dz * fz);
                }
            }
        }
    }

    last_potential_energy += local_pe;
    last_virial += local_virial;
}

// ============================================================================
// EWALD / PME RECIPROCAL-SPACE ELECTROSTATICS (CPU, FFTW)
//
// Mirrors the CUDA implementation in force_calculator_cuda.cu: full
// complex-to-complex (C2C) FFT grid (charge density in place becomes Phi(k)
// after the Poisson solve), then E(k) = -i*k*Phi(k) is computed explicitly
// and inverse-transformed per component -- this is the fix for the bug in
// the original (quarantined) CPU PME prototype, which inverse-transformed
// the potential itself and used it as if it were the force field, and never
// populated the y/z reciprocal force grids at all.
//
// TODO(ExaFMM): see the CUDA implementation's note -- an FMM-based solve
// would replace this spread/FFT/solve/interpolate sequence for multi-node
// scaling.
// ============================================================================

namespace {

Real ewald_solve_alpha_cpu(Real cutoff, Real tolerance) {
    Real lo = 1e-4, hi = 10.0 / cutoff;
    for (int iter = 0; iter < 100; ++iter) {
        Real mid = 0.5 * (lo + hi);
        if (std::erfc(mid * cutoff) > tolerance) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

int ewald_next_good_fft_size_cpu(int minimum) {
    if (minimum < 8) minimum = 8;
    for (int n = minimum; ; ++n) {
        int m = n;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return n;
    }
}

std::vector<Real> ewald_compute_bspline_moduli_cpu(int grid_size_1d, int order) {
    std::vector<Real> b_mod(grid_size_1d);
    for (int k = 0; k < grid_size_1d; ++k) {
        Real sum_real = 0.0, sum_imag = 0.0;
        for (int l = 0; l < order; ++l) {
            Real angle = 2.0 * M_PI * k * l / grid_size_1d;
            Real weight = 0.0;
            if (order == 4) {
                if (l == 0) weight = 1.0/6.0;
                else if (l == 1) weight = 2.0/3.0;
                else if (l == 2) weight = 1.0/6.0;
            } else if (order == 6) {
                if (l == 0) weight = 1.0/120.0;
                else if (l == 1) weight = 26.0/120.0;
                else if (l == 2) weight = 66.0/120.0;
                else if (l == 3) weight = 26.0/120.0;
                else if (l == 4) weight = 1.0/120.0;
            }
            sum_real += weight * std::cos(angle);
            sum_imag += weight * std::sin(angle);
        }
        Real m = std::sqrt(sum_real*sum_real + sum_imag*sum_imag);
        b_mod[k] = (m < 1e-10) ? 1.0 : m;
    }
    return b_mod;
}

// Cardinal cubic B-spline weights + grid offsets (matches the CUDA device
// function pme_compute_bspline_weights exactly, order 4 only -- the only
// spline order this code path uses).
void ewald_bspline_weights_cpu(Real u, int order, Real* weights, int* grid_indices) {
    int base_index = static_cast<int>(std::floor(u));
    Real w = u - static_cast<Real>(base_index);
    if (order == 4) {
        weights[0] = (1.0 - w) * (1.0 - w) * (1.0 - w) / 6.0;
        weights[1] = (3.0 * w * w * w - 6.0 * w * w + 4.0) / 6.0;
        weights[2] = (-3.0 * w * w * w + 3.0 * w * w + 3.0 * w + 1.0) / 6.0;
        weights[3] = w * w * w / 6.0;
        grid_indices[0] = base_index - 1;
        grid_indices[1] = base_index;
        grid_indices[2] = base_index + 1;
        grid_indices[3] = base_index + 2;
    }
}

inline int ewald_pbc_index_cpu(int idx, int grid_size) {
    idx %= grid_size;
    if (idx < 0) idx += grid_size;
    return idx;
}

void ewald_spread_charges_cpu(
    const Real* x, const Real* y, const Real* z, const Real* charge,
    fftw_complex* charge_grid,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    Real wx[8] = {}, wy[8] = {}, wz[8] = {};
    int ix[8] = {}, iy[8] = {}, iz[8] = {};
    for (size_t i = 0; i < num_particles; ++i) {
        Real qi = charge[i];
        if (std::fabs(qi) < 1e-12) continue;

        Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
        Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
        Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
        u_x -= std::floor(u_x / grid_nx) * grid_nx;
        u_y -= std::floor(u_y / grid_ny) * grid_ny;
        u_z -= std::floor(u_z / grid_nz) * grid_nz;

        ewald_bspline_weights_cpu(u_x, spline_order, wx, ix);
        ewald_bspline_weights_cpu(u_y, spline_order, wy, iy);
        ewald_bspline_weights_cpu(u_z, spline_order, wz, iz);

        for (int a = 0; a < spline_order; ++a) {
            int gx = ewald_pbc_index_cpu(ix[a], grid_nx);
            Real wax = wx[a] * qi;
            for (int b = 0; b < spline_order; ++b) {
                int gy = ewald_pbc_index_cpu(iy[b], grid_ny);
                Real waxy = wy[b] * wax;
                for (int c = 0; c < spline_order; ++c) {
                    int gz = ewald_pbc_index_cpu(iz[c], grid_nz);
                    Real w = wz[c] * waxy;
                    size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                               + static_cast<size_t>(gy) * grid_nz + gz;
                    charge_grid[idx][0] += w;  // imaginary part stays 0
                }
            }
        }
    }
}

// Solve the Poisson equation in place (rho(k) -> Phi(k)) and return the
// reciprocal-space energy. `prefactor` already includes k_coulomb.
Real ewald_solve_poisson_cpu(
    fftw_complex* fft_grid,
    const Real* b_mod_x, const Real* b_mod_y, const Real* b_mod_z,
    Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    Real energy = 0.0;
    Real alpha_sq = alpha * alpha;
    for (int kx = 0; kx < grid_nx; ++kx) {
        Real kxw = (kx <= grid_nx/2 ? kx : kx - grid_nx) * 2.0 * M_PI * inv_box_x;
        for (int ky = 0; ky < grid_ny; ++ky) {
            Real kyw = (ky <= grid_ny/2 ? ky : ky - grid_ny) * 2.0 * M_PI * inv_box_y;
            for (int kz = 0; kz < grid_nz; ++kz) {
                size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nz
                           + static_cast<size_t>(ky) * grid_nz + kz;
                if (kx == 0 && ky == 0 && kz == 0) {
                    fft_grid[idx][0] = 0.0;
                    fft_grid[idx][1] = 0.0;
                    continue;
                }
                Real kzw = (kz <= grid_nz/2 ? kz : kz - grid_nz) * 2.0 * M_PI * inv_box_z;
                Real k_sq = kxw*kxw + kyw*kyw + kzw*kzw;

                Real mx = b_mod_x[kx], my = b_mod_y[ky], mz = b_mod_z[kz];
                Real m_sq = mx*mx * my*my * mz*mz;

                Real gauss = std::exp(-k_sq / (4.0 * alpha_sq));
                Real green = prefactor * gauss / (k_sq * m_sq + 1e-12);

                Real re = fft_grid[idx][0], im = fft_grid[idx][1];
                energy += 0.5 * green * (re*re + im*im);
                fft_grid[idx][0] = green * re;
                fft_grid[idx][1] = green * im;
            }
        }
    }
    return energy;
}

Real ewald_reciprocal_virial_cpu(
    const fftw_complex* fft_grid,
    const Real* b_mod_x, const Real* b_mod_y, const Real* b_mod_z,
    Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    Real virial = 0.0;
    Real alpha_sq = alpha * alpha;
    for (int kx = 0; kx < grid_nx; ++kx) {
        Real kxw = (kx <= grid_nx/2 ? kx : kx - grid_nx) * 2.0 * M_PI * inv_box_x;
        for (int ky = 0; ky < grid_ny; ++ky) {
            Real kyw = (ky <= grid_ny/2 ? ky : ky - grid_ny) * 2.0 * M_PI * inv_box_y;
            for (int kz = 0; kz < grid_nz; ++kz) {
                if (kx == 0 && ky == 0 && kz == 0) continue;
                Real kzw = (kz <= grid_nz/2 ? kz : kz - grid_nz) * 2.0 * M_PI * inv_box_z;
                Real k_sq = kxw*kxw + kyw*kyw + kzw*kzw;

                Real mx = b_mod_x[kx], my = b_mod_y[ky], mz = b_mod_z[kz];
                Real m_sq = mx*mx * my*my * mz*mz;
                Real gauss = std::exp(-k_sq / (4.0 * alpha_sq));
                Real green = prefactor * gauss / (k_sq * m_sq + 1e-12);

                size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nz
                           + static_cast<size_t>(ky) * grid_nz + kz;
                Real re = fft_grid[idx][0], im = fft_grid[idx][1];
                Real rho_sq = re*re + im*im;
                Real virial_factor = 1.0 + k_sq / (2.0 * alpha_sq);
                virial += virial_factor * green * rho_sq;
            }
        }
    }
    return virial;
}

// E(k) = -i * k * Phi(k)
void ewald_compute_electric_field_cpu(
    const fftw_complex* phi_grid,
    fftw_complex* ex_grid, fftw_complex* ey_grid, fftw_complex* ez_grid,
    int grid_nx, int grid_ny, int grid_nz,
    Real box_x, Real box_y, Real box_z)
{
    for (int kx = 0; kx < grid_nx; ++kx) {
        Real kxw = (kx <= grid_nx/2 ? kx : kx - grid_nx) * 2.0 * M_PI / box_x;
        for (int ky = 0; ky < grid_ny; ++ky) {
            Real kyw = (ky <= grid_ny/2 ? ky : ky - grid_ny) * 2.0 * M_PI / box_y;
            for (int kz = 0; kz < grid_nz; ++kz) {
                Real kzw = (kz <= grid_nz/2 ? kz : kz - grid_nz) * 2.0 * M_PI / box_z;
                size_t idx = static_cast<size_t>(kx) * grid_ny * grid_nz
                           + static_cast<size_t>(ky) * grid_nz + kz;
                Real pr = phi_grid[idx][0], pi = phi_grid[idx][1];
                ex_grid[idx][0] = pi * kxw;  ex_grid[idx][1] = -pr * kxw;
                ey_grid[idx][0] = pi * kyw;  ey_grid[idx][1] = -pr * kyw;
                ez_grid[idx][0] = pi * kzw;  ez_grid[idx][1] = -pr * kzw;
            }
        }
    }
}

void ewald_interpolate_forces_cpu(
    const Real* x, const Real* y, const Real* z, const Real* charge,
    const Real* field_x, const Real* field_y, const Real* field_z,
    Real* fx, Real* fy, Real* fz,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    Real wx[8] = {}, wy[8] = {}, wz[8] = {};
    int ix[8] = {}, iy[8] = {}, iz[8] = {};
    for (size_t i = 0; i < num_particles; ++i) {
        Real qi = charge[i];
        if (std::fabs(qi) < 1e-12) continue;

        Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
        Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
        Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
        u_x -= std::floor(u_x / grid_nx) * grid_nx;
        u_y -= std::floor(u_y / grid_ny) * grid_ny;
        u_z -= std::floor(u_z / grid_nz) * grid_nz;

        ewald_bspline_weights_cpu(u_x, spline_order, wx, ix);
        ewald_bspline_weights_cpu(u_y, spline_order, wy, iy);
        ewald_bspline_weights_cpu(u_z, spline_order, wz, iz);

        Real ex = 0.0, ey = 0.0, ez = 0.0;
        for (int a = 0; a < spline_order; ++a) {
            int gx = ewald_pbc_index_cpu(ix[a], grid_nx);
            Real wax = wx[a];
            for (int b = 0; b < spline_order; ++b) {
                int gy = ewald_pbc_index_cpu(iy[b], grid_ny);
                Real waxy = wy[b] * wax;
                for (int c = 0; c < spline_order; ++c) {
                    int gz = ewald_pbc_index_cpu(iz[c], grid_nz);
                    Real w = wz[c] * waxy;
                    size_t idx = static_cast<size_t>(gx) * grid_ny * grid_nz
                               + static_cast<size_t>(gy) * grid_nz + gz;
                    ex += w * field_x[idx];
                    ey += w * field_y[idx];
                    ez += w * field_z[idx];
                }
            }
        }
        fx[i] += qi * ex;
        fy[i] += qi * ey;
        fz[i] += qi * ez;
    }
}

}  // namespace

void ForceCalculatorCPU::ensure_ewald_initialized(MDSystem& system) {
    SimulationBox& box = system.get_box();
    const Real k_coulomb = 332.0636;

    if (ewald_initialized_) {
        // Grid size and alpha stay fixed after first init; only the
        // volume-dependent Green's-function prefactor is refreshed (NPT).
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
        return;
    }

    Real cutoff = system.get_physical_cutoff();
    ewald_alpha_ = ewald_solve_alpha_cpu(cutoff, 1e-5);
    ewald_spline_order_ = 4;

    // Grid spacing must resolve the reciprocal-space Gaussian width, which
    // narrows as alpha grows: target alpha*h <= 0.25 (standard PME accuracy
    // rule of thumb), capped at 1.0 Angstrom for small alpha/large boxes
    // where that would be needlessly fine.
    const Real target_spacing = std::min(1.0, 0.25 / ewald_alpha_);
    ewald_grid_x_ = ewald_next_good_fft_size_cpu(static_cast<int>(std::ceil(box.lx / target_spacing)));
    ewald_grid_y_ = ewald_next_good_fft_size_cpu(static_cast<int>(std::ceil(box.ly / target_spacing)));
    ewald_grid_z_ = ewald_next_good_fft_size_cpu(static_cast<int>(std::ceil(box.lz / target_spacing)));

    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;

    pme_charge_grid_ = fftw_alloc_complex(grid_size);
    pme_ex_grid_     = fftw_alloc_complex(grid_size);
    pme_ey_grid_     = fftw_alloc_complex(grid_size);
    pme_ez_grid_     = fftw_alloc_complex(grid_size);
    pme_force_grid_x_ = new Real[grid_size];
    pme_force_grid_y_ = new Real[grid_size];
    pme_force_grid_z_ = new Real[grid_size];

    // One forward plan (charge grid, in place) and one backward plan reused
    // via fftw_execute_dft's new-array execute for all three field grids
    // (they share the same size/alignment since all are fftw_alloc_complex).
    pme_fft_plan_fwd_ = fftw_plan_dft_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_charge_grid_, pme_charge_grid_, FFTW_FORWARD, FFTW_ESTIMATE);
    pme_fft_plan_bwd_ = fftw_plan_dft_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_ex_grid_, pme_ex_grid_, FFTW_BACKWARD, FFTW_ESTIMATE);

    pme_b_spline_mod_x_ = ewald_compute_bspline_moduli_cpu(ewald_grid_x_, ewald_spline_order_);
    pme_b_spline_mod_y_ = ewald_compute_bspline_moduli_cpu(ewald_grid_y_, ewald_spline_order_);
    pme_b_spline_mod_z_ = ewald_compute_bspline_moduli_cpu(ewald_grid_z_, ewald_spline_order_);

    ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
    ewald_initialized_ = true;

    if (is_rank_zero()) {
        std::cout << "[Ewald/PME][CPU] grid " << ewald_grid_x_ << "x" << ewald_grid_y_ << "x" << ewald_grid_z_
                  << ", alpha=" << ewald_alpha_ << " 1/A, spline_order=" << ewald_spline_order_
                  << ", cutoff=" << cutoff << " A" << std::endl;
    }
}

void ForceCalculatorCPU::free_ewald_resources() {
    if (pme_fft_plan_fwd_) { fftw_destroy_plan(pme_fft_plan_fwd_); pme_fft_plan_fwd_ = nullptr; }
    if (pme_fft_plan_bwd_) { fftw_destroy_plan(pme_fft_plan_bwd_); pme_fft_plan_bwd_ = nullptr; }
    if (pme_charge_grid_) { fftw_free(pme_charge_grid_); pme_charge_grid_ = nullptr; }
    if (pme_ex_grid_) { fftw_free(pme_ex_grid_); pme_ex_grid_ = nullptr; }
    if (pme_ey_grid_) { fftw_free(pme_ey_grid_); pme_ey_grid_ = nullptr; }
    if (pme_ez_grid_) { fftw_free(pme_ez_grid_); pme_ez_grid_ = nullptr; }
    if (pme_force_grid_x_) { delete[] pme_force_grid_x_; pme_force_grid_x_ = nullptr; }
    if (pme_force_grid_y_) { delete[] pme_force_grid_y_; pme_force_grid_y_ = nullptr; }
    if (pme_force_grid_z_) { delete[] pme_force_grid_z_; pme_force_grid_z_ = nullptr; }
    ewald_initialized_ = false;
}

void ForceCalculatorCPU::compute_electrostatic_forces_ewald(MDSystem& system) {
    ensure_ewald_initialized(system);

    // ---- Real-space (near-field, erfc-screened) part ----
    compute_electrostatic_forces_real_space(system);

    // ---- Reciprocal-space (long-range) part ----
    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    const Real k_coulomb = 332.0636;
    // Local particles only: ghosts are periodic copies of atoms owned by
    // other MPI ranks and must not be double-counted in the charge density.
    size_t local_particles = system.get_local_particles();

    Real inv_box_x = 1.0/box.lx, inv_box_y = 1.0/box.ly, inv_box_z = 1.0/box.lz;
    const size_t grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * ewald_grid_z_;

    std::memset(pme_charge_grid_, 0, grid_size * sizeof(fftw_complex));
    ewald_spread_charges_cpu(
        particles.x.data(), particles.y.data(), particles.z.data(), particles.charge.data(),
        pme_charge_grid_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles);

    fftw_execute(pme_fft_plan_fwd_);

    Real recip_energy = ewald_solve_poisson_cpu(
        pme_charge_grid_, pme_b_spline_mod_x_.data(), pme_b_spline_mod_y_.data(), pme_b_spline_mod_z_.data(),
        ewald_prefactor_, ewald_alpha_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z);
    last_potential_energy += recip_energy;

    Real recip_virial = ewald_reciprocal_virial_cpu(
        pme_charge_grid_, pme_b_spline_mod_x_.data(), pme_b_spline_mod_y_.data(), pme_b_spline_mod_z_.data(),
        ewald_prefactor_, ewald_alpha_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z);
    last_virial += recip_virial;

    ewald_compute_electric_field_cpu(
        pme_charge_grid_, pme_ex_grid_, pme_ey_grid_, pme_ez_grid_,
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, box.lx, box.ly, box.lz);

    fftw_execute_dft(pme_fft_plan_bwd_, pme_ex_grid_, pme_ex_grid_);
    fftw_execute_dft(pme_fft_plan_bwd_, pme_ey_grid_, pme_ey_grid_);
    fftw_execute_dft(pme_fft_plan_bwd_, pme_ez_grid_, pme_ez_grid_);

    // FFTW does not normalize inverse transforms; divide by grid_size. The
    // imaginary part after the inverse transform is numerical noise only
    // (E(k) has Hermitian symmetry for a real charge distribution).
    Real norm = 1.0 / static_cast<Real>(grid_size);
    for (size_t idx = 0; idx < grid_size; ++idx) {
        pme_force_grid_x_[idx] = pme_ex_grid_[idx][0] * norm;
        pme_force_grid_y_[idx] = pme_ey_grid_[idx][0] * norm;
        pme_force_grid_z_[idx] = pme_ez_grid_[idx][0] * norm;
    }

    ewald_interpolate_forces_cpu(
        particles.x.data(), particles.y.data(), particles.z.data(), particles.charge.data(),
        pme_force_grid_x_, pme_force_grid_y_, pme_force_grid_z_,
        particles.fx.data(), particles.fy.data(), particles.fz.data(),
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles);

    // ---- Self-energy correction (local particles only) ----
    Real self_energy_sum_q2 = 0.0;
    for (size_t i = 0; i < local_particles; ++i) {
        self_energy_sum_q2 += particles.charge[i] * particles.charge[i];
    }
    last_potential_energy -= k_coulomb * ewald_alpha_ / 1.7724538509055159 * self_energy_sum_q2;  // sqrt(pi)
}

void ForceCalculatorCPU::compute_bond_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    
    // Check if we have bonds
    const auto& bonds = system.get_local_bonds();
    if (bonds.empty()) {
        return; // No bonds defined
    }
    
    for (const auto& bond : bonds) {
        size_t i = bond.atom1;
        size_t j = bond.atom2;
        
        if (i >= particles.x.size() || j >= particles.x.size()) continue;
        
        Real dx = particles.x[i] - particles.x[j];
        Real dy = particles.y[i] - particles.y[j];
        Real dz = particles.z[i] - particles.z[j];
        
        // Apply periodic boundary conditions
        SimulationBox& box = system.get_box();
        box.apply_minimum_image(dx, dy, dz);
        
        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-10) continue; // Avoid division by zero
        
        // Get bond parameters from topology
        const BondType& bond_type = topology.get_bond_type(bond.type);
        Real k_bond = bond_type.k;    // Force constant
        Real r0 = bond_type.r0;       // Equilibrium distance
        
        // Harmonic bond potential: V = 0.5 * k * (r - r0)^2
        // Force: F = -k * (r - r0) * r_hat
        Real dr = r - r0;
        last_potential_energy += 0.5 * k_bond * dr * dr;
        Real force_mag = -k_bond * dr / r;
        
        Real fx = force_mag * dx;
        Real fy = force_mag * dy;
        Real fz = force_mag * dz;
        
        particles.fx[i] += fx;
        particles.fy[i] += fy;
        particles.fz[i] += fz;
        particles.fx[j] -= fx;
        particles.fy[j] -= fy;
        particles.fz[j] -= fz;
    }
}

void ForceCalculatorCPU::compute_angle_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    
    // Check if we have angles
    const auto& angles = system.get_local_angles();
    if (angles.empty()) {
        return; // No angles defined
    }
    
    for (const auto& angle : angles) {
        size_t i = angle.atom1;  // First atom
        size_t j = angle.atom2;  // Central atom
        size_t k = angle.atom3;  // Third atom
        
        if (i >= particles.x.size() || j >= particles.x.size() || k >= particles.x.size()) continue;
        
        // Vectors from central atom j to atoms i and k
        Real dx_ij = particles.x[i] - particles.x[j];
        Real dy_ij = particles.y[i] - particles.y[j];
        Real dz_ij = particles.z[i] - particles.z[j];
        
        Real dx_kj = particles.x[k] - particles.x[j];
        Real dy_kj = particles.y[k] - particles.y[j];
        Real dz_kj = particles.z[k] - particles.z[j];
        
        // Apply periodic boundary conditions
        SimulationBox& box = system.get_box();
        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
        box.apply_minimum_image(dx_kj, dy_kj, dz_kj);
        
        Real r_ij = std::sqrt(dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij);
        Real r_kj = std::sqrt(dx_kj*dx_kj + dy_kj*dy_kj + dz_kj*dz_kj);
        
        if (r_ij < 1e-10 || r_kj < 1e-10) continue; // Avoid division by zero
        
        // Calculate angle
        Real dot_product = dx_ij*dx_kj + dy_ij*dy_kj + dz_ij*dz_kj;
        Real cos_theta = dot_product / (r_ij * r_kj);
        
        // Clamp to avoid numerical issues
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        
        Real theta = std::acos(cos_theta);
        
        // Get angle parameters from topology
        const AngleType& angle_type = topology.get_angle_type(angle.type);
        Real k_angle = angle_type.k;         // Force constant
        Real theta0 = angle_type.theta0;     // Equilibrium angle
        Real dtheta = theta - theta0;
        // Harmonic angle potential: V = 0.5 * k * (theta - theta0)^2
        last_potential_energy += 0.5 * k_angle * dtheta * dtheta;

        if (std::abs(dtheta) < 1e-10) continue; // No force if at equilibrium
        
        // Use more numerically stable force calculation
        Real sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        if (sin_theta < 1e-10) continue; // Avoid division by zero
        
        // -dV/dtheta times dtheta/dr.  The vector expressions below are
        // (v_hat - cos(theta) u_hat)/r, i.e. the negative angular gradient,
        // so the scalar coefficient must be positive here.
        Real force_constant = k_angle * dtheta / sin_theta;
        
        // Force calculation (simplified - this is a complex derivation)
        Real r_ij_inv = 1.0 / r_ij;
        Real r_kj_inv = 1.0 / r_kj;
        
        Real cos_theta_ij = cos_theta / r_ij;
        Real cos_theta_kj = cos_theta / r_kj;
        
        Real fx_i = force_constant * (dx_kj * r_ij_inv * r_kj_inv - dx_ij * cos_theta_ij * r_ij_inv);
        Real fy_i = force_constant * (dy_kj * r_ij_inv * r_kj_inv - dy_ij * cos_theta_ij * r_ij_inv);
        Real fz_i = force_constant * (dz_kj * r_ij_inv * r_kj_inv - dz_ij * cos_theta_ij * r_ij_inv);
        
        Real fx_k = force_constant * (dx_ij * r_ij_inv * r_kj_inv - dx_kj * cos_theta_kj * r_kj_inv);
        Real fy_k = force_constant * (dy_ij * r_ij_inv * r_kj_inv - dy_kj * cos_theta_kj * r_kj_inv);
        Real fz_k = force_constant * (dz_ij * r_ij_inv * r_kj_inv - dz_kj * cos_theta_kj * r_kj_inv);
        
        // Forces on central atom j (Newton's third law)
        Real fx_j = -(fx_i + fx_k);
        Real fy_j = -(fy_i + fy_k);
        Real fz_j = -(fz_i + fz_k);
        
        particles.fx[i] += fx_i;
        particles.fy[i] += fy_i;
        particles.fz[i] += fz_i;
        particles.fx[j] += fx_j;
        particles.fy[j] += fy_j;
        particles.fz[j] += fz_j;
        particles.fx[k] += fx_k;
        particles.fy[k] += fy_k;
        particles.fz[k] += fz_k;
    }
}

void ForceCalculatorCPU::compute_dihedral_forces(MDSystem& system) {
    compute_torsion_forces(system, system.get_local_dihedrals());
}

void ForceCalculatorCPU::compute_improper_forces(MDSystem& system) {
    compute_torsion_forces(system, system.get_local_impropers());
}

void ForceCalculatorCPU::compute_torsion_forces(MDSystem& system,
                                                 const std::vector<MDSystem::LocalDihedral>& dihedrals) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();

    if (dihedrals.empty()) {
        return; // No torsions of this kind defined
    }

    SimulationBox& box = system.get_box();

    for (const auto& dihedral : dihedrals) {
        size_t i = dihedral.atom1;
        size_t j = dihedral.atom2;
        size_t k = dihedral.atom3;
        size_t l = dihedral.atom4;

        if (i >= particles.x.size() || j >= particles.x.size() ||
            k >= particles.x.size() || l >= particles.x.size()) continue;

        // Vectors for dihedral calculation
        Real dx_ij = particles.x[i] - particles.x[j];
        Real dy_ij = particles.y[i] - particles.y[j];
        Real dz_ij = particles.z[i] - particles.z[j];

        Real dx_jk = particles.x[j] - particles.x[k];
        Real dy_jk = particles.y[j] - particles.y[k];
        Real dz_jk = particles.z[j] - particles.z[k];

        Real dx_kl = particles.x[k] - particles.x[l];
        Real dy_kl = particles.y[k] - particles.y[l];
        Real dz_kl = particles.z[k] - particles.z[l];

        // Apply periodic boundary conditions
        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
        box.apply_minimum_image(dx_jk, dy_jk, dz_jk);
        box.apply_minimum_image(dx_kl, dy_kl, dz_kl);

        // Cross products for plane normals -- kept UNNORMALIZED, the exact
        // analytical force formula below needs the raw vectors and |.|^2.
        Real n1_x = dy_ij * dz_jk - dz_ij * dy_jk;  // ij x jk
        Real n1_y = dz_ij * dx_jk - dx_ij * dz_jk;
        Real n1_z = dx_ij * dy_jk - dy_ij * dx_jk;

        Real n2_x = dy_jk * dz_kl - dz_jk * dy_kl;  // jk x kl
        Real n2_y = dz_jk * dx_kl - dx_jk * dz_kl;
        Real n2_z = dx_jk * dy_kl - dy_jk * dx_kl;

        Real n1_sq = n1_x*n1_x + n1_y*n1_y + n1_z*n1_z;
        Real n2_sq = n2_x*n2_x + n2_y*n2_y + n2_z*n2_z;
        Real jk_sq = dx_jk*dx_jk + dy_jk*dy_jk + dz_jk*dz_jk;

        if (n1_sq < 1e-20 || n2_sq < 1e-20 || jk_sq < 1e-20) continue; // Degenerate geometry

        Real jk_mag = std::sqrt(jk_sq);

        // Dihedral angle via atan2(sin, cos) computed directly from the
        // unnormalized cross products (equivalent to normalizing first, but
        // avoids the extra sqrt/division and keeps n1/n2 intact for the
        // force formula below).
        Real cos_phi_num = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
        Real sin_phi_num = (n1_y*n2_z - n1_z*n2_y) * dx_jk +
                            (n1_z*n2_x - n1_x*n2_z) * dy_jk +
                            (n1_x*n2_y - n1_y*n2_x) * dz_jk;
        Real phi = std::atan2(sin_phi_num, cos_phi_num * jk_mag);

        // Get dihedral parameters from topology
        const DihedralType& dihedral_type = topology.get_dihedral_type(dihedral.type);
        Real k_dihedral = dihedral_type.k;
        Real phi0 = dihedral_type.phi0;  // Phase shift
        int n = dihedral_type.n;         // Multiplicity

        Real dVdphi;
        if (n == 0) {
            // CHARMM harmonic improper: V = Kpsi*(phi-phi0)^2. Keep the
            // angular displacement on the principal branch.
            Real dphi = std::remainder(phi - phi0, 2.0 * M_PI);
            last_potential_energy += k_dihedral * dphi * dphi;
            dVdphi = 2.0 * k_dihedral * dphi;
        } else {
            // CHARMM-style periodic torsion: V = k*(1+cos(n*phi-phi0)).
            Real dphi = n * phi - phi0;
            last_potential_energy += k_dihedral * (1.0 + std::cos(dphi));
            dVdphi = -k_dihedral * n * std::sin(dphi);
        }

        // Exact analytical gradient (Blondel-Karplus / Bekker formula),
        // momentum- and torque-conserving by construction. Using the
        // standard notation r_ij, r_kj = -dx_jk, r_kl, m = r_ij x r_kj,
        // n_std = r_kj x r_kl (note m = -n1, n_std = -n2 for the ij x jk /
        // jk x kl cross products already computed above), the textbook
        // With the signed angle convention used by atan2 above, dphi/dr_i
        // points along +n1 and dphi/dr_l along -n2. Force is -dV/dphi times
        // that gradient.
        Real fi_scale = -dVdphi * jk_mag / n1_sq;
        Real fl_scale =  dVdphi * jk_mag / n2_sq;

        Real fix = fi_scale * n1_x, fiy = fi_scale * n1_y, fiz = fi_scale * n1_z;
        Real flx = fl_scale * n2_x, fly = fl_scale * n2_y, flz = fl_scale * n2_z;

        // p, q are the fractional projections of r_ij and r_kl onto r_kj
        // (= -dx_jk here, but the sign cancels in the dot-product ratio).
        Real p = -(dx_ij*dx_jk + dy_ij*dy_jk + dz_ij*dz_jk) / jk_sq;
        Real q = -(dx_kl*dx_jk + dy_kl*dy_jk + dz_kl*dz_jk) / jk_sq;

        Real fjx = (p - 1.0) * fix - q * flx;
        Real fjy = (p - 1.0) * fiy - q * fly;
        Real fjz = (p - 1.0) * fiz - q * flz;

        Real fkx = -p * fix + (q - 1.0) * flx;
        Real fky = -p * fiy + (q - 1.0) * fly;
        Real fkz = -p * fiz + (q - 1.0) * flz;

        particles.fx[i] += fix; particles.fy[i] += fiy; particles.fz[i] += fiz;
        particles.fx[j] += fjx; particles.fy[j] += fjy; particles.fz[j] += fjz;
        particles.fx[k] += fkx; particles.fy[k] += fky; particles.fz[k] += fkz;
        particles.fx[l] += flx; particles.fy[l] += fly; particles.fz[l] += flz;
    }
}

void ForceCalculatorCPU::compute_special_pair_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    ForceFieldParams& params = system.get_ff_params();
    SimulationBox& box = system.get_box();

    const auto& pairs = system.get_local_special_pairs();
    if (pairs.empty()) {
        return; // No special pairs defined
    }

    const ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    const bool do_lj = force_config.enable_lennard_jones;
    const bool do_coulomb = force_config.enable_coulomb;
    if (!do_lj && !do_coulomb) {
        return;
    }

    // Direct, unscreened LJ + Coulomb -- never subject to the normal
    // neighbor-list cutoff. Requires the pair to also be listed in
    // [exclusions] so the main pairwise loops skip it. Under Ewald/PME, the
    // reciprocal-space sum includes the erf(alpha*r)/r term unconditionally
    // for every charged pair, and compute_electrostatic_forces_real_space()
    // subtracts that same term for excluded pairs, so the net Coulomb
    // contribution from those two routines is exactly zero here -- only the
    // scaled direct term added below survives.
    const Real k_coulomb = 332.0636; // kcal/mol*Angstrom / e^2 (matches the Ewald real-space routine)
    Real local_pe = 0.0;
    Real local_virial = 0.0;

    for (const auto& pair : pairs) {
        size_t i = pair.atom1;
        size_t j = pair.atom2;
        if (i >= particles.x.size() || j >= particles.x.size()) continue;

        Real dx = particles.x[i] - particles.x[j];
        Real dy = particles.y[i] - particles.y[j];
        Real dz = particles.z[i] - particles.z[j];
        box.apply_minimum_image(dx, dy, dz);
        Real r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-10) continue; // Avoid division by zero

        Real force_mag = 0.0;

        if (do_lj) {
            int type_i = particles.type[i];
            int type_j = particles.type[j];
            Real sigma = pair.sigma >= 0.0 ? pair.sigma : params.sigma_mixed[type_i][type_j];
            Real epsilon = pair.epsilon >= 0.0 ? pair.epsilon : params.epsilon_mixed[type_i][type_j];
            if (epsilon != 0.0) {
                Real r2inv = 1.0 / r2;
                Real attractive;
                Real repulsive;
                if (params.mie_repulsive_exponent == 12) {
                    const Real sig2 = sigma * sigma;
                    const Real sig6 = sig2 * sig2 * sig2;
                    attractive = sig6 * r2inv * r2inv * r2inv;
                    repulsive = attractive * attractive;
                } else {
                    const Real sig_r = sigma * std::sqrt(r2inv);
                    const Real sig3_r3 = sig_r * sig_r * sig_r;
                    attractive = sig3_r3 * sig3_r3;
                    repulsive = attractive * sig3_r3;
                }
                force_mag += params.mie_prefactor * epsilon * r2inv *
                    (params.mie_repulsive_exponent * repulsive -
                     params.mie_attractive_exponent * attractive) * pair.lj_scale;
                local_pe += params.mie_prefactor * epsilon *
                    (repulsive - attractive) * pair.lj_scale;
            }
        }

        if (do_coulomb) {
            Real qi = particles.charge[i];
            Real qj = particles.charge[j];
            if (qi != 0.0 && qj != 0.0) {
                Real r = std::sqrt(r2);
                Real qq = k_coulomb * qi * qj * pair.coulomb_scale;
                force_mag += qq / (r2 * r);
                local_pe += qq / r;
            }
        }

        if (force_mag == 0.0) continue;

        local_virial += r2 * force_mag; // W = r.F = r^2 * force_mag

        Real fx = force_mag * dx;
        Real fy = force_mag * dy;
        Real fz = force_mag * dz;

        particles.fx[i] += fx;
        particles.fy[i] += fy;
        particles.fz[i] += fz;
        particles.fx[j] -= fx;
        particles.fy[j] -= fy;
        particles.fz[j] -= fz;
    }

    last_potential_energy += local_pe;
    last_virial += local_virial;
}

void ForceCalculatorCPU::calculate_kinetic_energy(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    last_kinetic_energy = 0.0;
    
    for (size_t i = 0; i < local_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        last_kinetic_energy += 0.5 * particles.mass[i] * v2;
    }
    
    if (system.get_rank() == 0) {
        // Debug removed
    }
    
    // Sum across all MPI ranks
    Real global_ke = 0.0;
    MPI_Allreduce(&last_kinetic_energy, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_kinetic_energy = global_ke;
}
