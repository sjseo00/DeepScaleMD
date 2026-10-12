#include "pme_reciprocal_engine.h"
#include "mpi_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

Real pme_solve_alpha(Real cutoff, Real tolerance) {
    Real lo = 1e-4, hi = 10.0 / cutoff;
    for (int iter = 0; iter < 100; ++iter) {
        Real mid = 0.5 * (lo + hi);
        if (std::erfc(mid * cutoff) > tolerance) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

int pme_next_good_fft_size(int minimum, bool require_even = false) {
    if (minimum < 8) minimum = 8;
    for (int n = minimum; ; ++n) {
        if (require_even && (n % 2 != 0)) continue;
        int m = n;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return n;
    }
}

std::vector<Real> pme_compute_bspline_moduli(int grid_size_1d, int order) {
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

void pme_bspline_weights(Real u, int order, Real* weights, int* grid_indices) {
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

void pme_bspline_weights_and_derivatives(
    Real u, int order, Real* weights, Real* derivatives, int* grid_indices) {
    pme_bspline_weights(u, order, weights, grid_indices);
    const Real w = u - std::floor(u);
    if (order == 4) {
        const Real omw = 1.0 - w;
        const Real w2 = w * w;
        derivatives[0] = -0.5 * omw * omw;
        derivatives[1] =  1.5 * w2 - 2.0 * w;
        derivatives[2] = -1.5 * w2 + w + 0.5;
        derivatives[3] =  0.5 * w2;
    }
}

inline int pme_pbc_index(int idx, int grid_size) {
    idx %= grid_size;
    if (idx < 0) idx += grid_size;
    return idx;
}

void pme_spread_charges(
    const Real* x, const Real* y, const Real* z, const Real* charge,
    Real* charge_grid,
    int grid_nx, int grid_ny, int grid_nz, int grid_nz_pad,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles,
    std::vector<Real>& thread_scratch)
{
    const size_t grid_size = static_cast<size_t>(grid_nx) * grid_ny * grid_nz_pad;
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    const size_t scratch_size = static_cast<size_t>(nthreads) * grid_size;

    if (thread_scratch.size() != scratch_size) {
        thread_scratch.assign(scratch_size, 0.0);
    }

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Real* grid = thread_scratch.data() + static_cast<size_t>(tid) * grid_size;
        Real wx[8] = {}, wy[8] = {}, wz[8] = {};
        int ix[8] = {}, iy[8] = {}, iz[8] = {};
        int gx[8] = {}, gy[8] = {}, gz[8] = {};

        #pragma omp for schedule(static)
        for (long ii = 0; ii < static_cast<long>(num_particles); ++ii) {
            size_t i = static_cast<size_t>(ii);
            Real qi = charge[i];
            if (std::fabs(qi) < 1e-12) continue;

            Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
            Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
            Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
            u_x -= std::floor(u_x / grid_nx) * grid_nx;
            u_y -= std::floor(u_y / grid_ny) * grid_ny;
            u_z -= std::floor(u_z / grid_nz) * grid_nz;

            pme_bspline_weights(u_x, spline_order, wx, ix);
            pme_bspline_weights(u_y, spline_order, wy, iy);
            pme_bspline_weights(u_z, spline_order, wz, iz);

            for (int a = 0; a < spline_order; ++a) gx[a] = pme_pbc_index(ix[a], grid_nx);
            for (int b = 0; b < spline_order; ++b) gy[b] = pme_pbc_index(iy[b], grid_ny);
            for (int c = 0; c < spline_order; ++c) gz[c] = pme_pbc_index(iz[c], grid_nz);

            for (int a = 0; a < spline_order; ++a) {
                const size_t off_x = static_cast<size_t>(gx[a]) * grid_ny * grid_nz_pad;
                const Real wax = wx[a] * qi;
                for (int b = 0; b < spline_order; ++b) {
                    Real* __restrict__ row = grid + off_x + static_cast<size_t>(gy[b]) * grid_nz_pad;
                    const Real waxy = wy[b] * wax;
                    for (int c = 0; c < spline_order; ++c) {
                        row[gz[c]] += wz[c] * waxy;
                    }
                }
            }
        }
    }

    constexpr long kChunk = 4096;
    const long nchunks = static_cast<long>((grid_size + kChunk - 1) / kChunk);
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < nchunks; ++c) {
        const long begin = c * kChunk;
        const long end = std::min(begin + kChunk, static_cast<long>(grid_size));
        for (int t = 0; t < nthreads; ++t) {
            Real* __restrict__ slice = thread_scratch.data() + static_cast<size_t>(t) * grid_size;
            if (t == 0) {
                #pragma omp simd
                for (long idx = begin; idx < end; ++idx) {
                    charge_grid[idx] = slice[idx];
                    slice[idx] = 0.0;
                }
            } else {
                #pragma omp simd
                for (long idx = begin; idx < end; ++idx) {
                    charge_grid[idx] += slice[idx];
                    slice[idx] = 0.0;
                }
            }
        }
    }
}

Real pme_apply_green(
    fftw_complex* fft_grid,
    const Real* __restrict__ green,
    const Real* __restrict__ green_energy,
    const Real* __restrict__ green_virial,
    size_t num_k, Real& virial_out)
{
    Real energy = 0.0;
    Real virial = 0.0;
    #pragma omp parallel for reduction(+:energy, virial) schedule(static)
    for (long idx = 0; idx < static_cast<long>(num_k); ++idx) {
        const Real re = fft_grid[idx][0];
        const Real im = fft_grid[idx][1];
        const Real rho_sq = re * re + im * im;
        energy += green_energy[idx] * rho_sq;
        virial += green_virial[idx] * rho_sq;
        const Real g = green[idx];
        fft_grid[idx][0] = g * re;
        fft_grid[idx][1] = g * im;
    }
    virial_out = virial;
    return energy;
}

void pme_interpolate_forces_from_potential(
    const Real* x, const Real* y, const Real* z, const Real* charge,
    const Real* potential_grid,
    Real* fx, Real* fy, Real* fz,
    int grid_nx, int grid_ny, int grid_nz, int grid_nz_pad,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    const Real inv_grid_size =
        1.0 / (static_cast<Real>(grid_nx) * grid_ny * grid_nz);
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(num_particles); ++ii) {
        size_t i = static_cast<size_t>(ii);
        Real wx[8] = {}, wy[8] = {}, wz[8] = {};
        Real dwx[8] = {}, dwy[8] = {}, dwz[8] = {};
        int ix[8] = {}, iy[8] = {}, iz[8] = {};
        int gx[8] = {}, gy[8] = {}, gz[8] = {};
        Real qi = charge[i];
        if (std::fabs(qi) < 1e-12) continue;

        Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
        Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
        Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
        u_x -= std::floor(u_x / grid_nx) * grid_nx;
        u_y -= std::floor(u_y / grid_ny) * grid_ny;
        u_z -= std::floor(u_z / grid_nz) * grid_nz;

        pme_bspline_weights_and_derivatives(u_x, spline_order, wx, dwx, ix);
        pme_bspline_weights_and_derivatives(u_y, spline_order, wy, dwy, iy);
        pme_bspline_weights_and_derivatives(u_z, spline_order, wz, dwz, iz);

        for (int a = 0; a < spline_order; ++a) gx[a] = pme_pbc_index(ix[a], grid_nx);
        for (int b = 0; b < spline_order; ++b) gy[b] = pme_pbc_index(iy[b], grid_ny);
        for (int c = 0; c < spline_order; ++c) gz[c] = pme_pbc_index(iz[c], grid_nz);

        Real dphi_du_x = 0.0, dphi_du_y = 0.0, dphi_du_z = 0.0;
        for (int a = 0; a < spline_order; ++a) {
            const size_t off_x = static_cast<size_t>(gx[a]) * grid_ny * grid_nz_pad;
            for (int b = 0; b < spline_order; ++b) {
                const Real* __restrict__ row = potential_grid + off_x
                                             + static_cast<size_t>(gy[b]) * grid_nz_pad;
                const Real wab   = wx[a]  * wy[b];
                const Real dwa_b = dwx[a] * wy[b];
                const Real wa_db = wx[a]  * dwy[b];
                for (int c = 0; c < spline_order; ++c) {
                    const Real phi = row[gz[c]];
                    dphi_du_x += dwa_b * wz[c]  * phi;
                    dphi_du_y += wa_db * wz[c]  * phi;
                    dphi_du_z += wab   * dwz[c] * phi;
                }
            }
        }
        fx[i] += -qi * dphi_du_x * (grid_nx * inv_box_x) * inv_grid_size;
        fy[i] += -qi * dphi_du_y * (grid_ny * inv_box_y) * inv_grid_size;
        fz[i] += -qi * dphi_du_z * (grid_nz * inv_box_z) * inv_grid_size;
    }
}

}  // namespace

PmeReciprocalEngine::~PmeReciprocalEngine() {
    free_resources();
}

void PmeReciprocalEngine::ensure_initialized(Real cutoff, const SimulationBox& box) {
    const Real k_coulomb = 332.0636;

    if (initialized_) {
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
        return;
    }

    ewald_alpha_ = pme_solve_alpha(cutoff, 1e-5);
    ewald_spline_order_ = 4;

    Real target_spacing = std::min(static_cast<Real>(1.2), static_cast<Real>(0.4) / ewald_alpha_);
    if (grid_spacing_override_ > 0.0) target_spacing = grid_spacing_override_;
    ewald_grid_x_ = pme_next_good_fft_size(static_cast<int>(std::ceil(box.lx / target_spacing)));
    ewald_grid_y_ = pme_next_good_fft_size(static_cast<int>(std::ceil(box.ly / target_spacing)));
    ewald_grid_z_ = pme_next_good_fft_size(static_cast<int>(std::ceil(box.lz / target_spacing)), true);

    pme_grid_z_half_ = ewald_grid_z_ / 2 + 1;
    pme_grid_z_pad_ = 2 * pme_grid_z_half_;

    const size_t real_grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_pad_;
    pme_real_grid_ = fftw_alloc_real(real_grid_size);
    pme_complex_grid_ = reinterpret_cast<fftw_complex*>(pme_real_grid_);
    std::memset(pme_real_grid_, 0, real_grid_size * sizeof(Real));

#ifdef DEEPSCALEMD_HAVE_FFTW_OMP
    {
        int threads_ok = fftw_init_threads();
        if (threads_ok) {
#ifdef _OPENMP
            fftw_plan_with_nthreads(omp_get_max_threads());
#else
            fftw_plan_with_nthreads(1);
#endif
        } else if (is_rank_zero()) {
            std::cerr << "[Ewald/PME][CPU] WARNING: fftw_init_threads() failed; "
                         "PME FFTs will run single-threaded." << std::endl;
        }
    }
#endif

    pme_fft_plan_fwd_ = fftw_plan_dft_r2c_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_real_grid_, pme_complex_grid_, FFTW_MEASURE);
    pme_fft_plan_bwd_ = fftw_plan_dft_c2r_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_complex_grid_, pme_real_grid_, FFTW_MEASURE);

    pme_b_spline_mod_x_ = pme_compute_bspline_moduli(ewald_grid_x_, ewald_spline_order_);
    pme_b_spline_mod_y_ = pme_compute_bspline_moduli(ewald_grid_y_, ewald_spline_order_);
    pme_b_spline_mod_z_ = pme_compute_bspline_moduli(ewald_grid_z_, ewald_spline_order_);

    ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
    initialized_ = true;

    if (is_rank_zero()) {
        std::cout << "[Ewald/PME][reciprocal] grid " << ewald_grid_x_ << "x" << ewald_grid_y_
                  << "x" << ewald_grid_z_ << " (r2c), alpha=" << ewald_alpha_
                  << " 1/A, spline_order=" << ewald_spline_order_
                  << ", spacing<=" << target_spacing << " A, cutoff=" << cutoff << " A" << std::endl;
    }
}

#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
void PmeReciprocalEngine::ensure_initialized_distributed(Real cutoff, const SimulationBox& box,
                                                          MPI_Comm pme_comm) {
    const Real k_coulomb = 332.0636;

    if (initialized_) {
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
        return;
    }

    distributed_ = true;
    pme_comm_ = pme_comm;
    fftw_mpi_init();

    ewald_alpha_ = pme_solve_alpha(cutoff, 1e-5);
    ewald_spline_order_ = 4;

    Real target_spacing = std::min(static_cast<Real>(1.2), static_cast<Real>(0.4) / ewald_alpha_);
    if (grid_spacing_override_ > 0.0) target_spacing = grid_spacing_override_;
    ewald_grid_x_ = pme_next_good_fft_size(static_cast<int>(std::ceil(box.lx / target_spacing)));
    ewald_grid_y_ = pme_next_good_fft_size(static_cast<int>(std::ceil(box.ly / target_spacing)));
    ewald_grid_z_ = pme_next_good_fft_size(static_cast<int>(std::ceil(box.lz / target_spacing)), true);

    pme_grid_z_half_ = ewald_grid_z_ / 2 + 1;
    pme_grid_z_pad_ = 2 * pme_grid_z_half_;

    // fftw_mpi_local_size_3d wants the r2c "half" extent (N2/2+1) for the last
    // dimension and returns the COMPLEX-element count this rank must
    // allocate -- per FFTW's own docs this may exceed
    // local_nx_*ewald_grid_y_*pme_grid_z_half_ (internal load-balancing
    // padding), so the allocation always uses this returned value; only the
    // local_nx_*ewald_grid_y_*pme_grid_z_pad_ REAL prefix of that allocation
    // is ours to read/write (spread/Green/interpolate all index within that
    // prefix only, exactly like the non-distributed case's full-grid layout).
    alloc_local_complex_ = fftw_mpi_local_size_3d(
        ewald_grid_x_, ewald_grid_y_, pme_grid_z_half_, pme_comm_,
        &local_nx_, &local_x_start_);

    const size_t alloc_reals = 2 * static_cast<size_t>(alloc_local_complex_);
    pme_real_grid_ = fftw_alloc_real(alloc_reals);
    pme_complex_grid_ = reinterpret_cast<fftw_complex*>(pme_real_grid_);
    std::memset(pme_real_grid_, 0, alloc_reals * sizeof(Real));

#ifdef DEEPSCALEMD_HAVE_FFTW_OMP
    {
        int threads_ok = fftw_init_threads();
        if (threads_ok) {
#ifdef _OPENMP
            fftw_plan_with_nthreads(omp_get_max_threads());
#else
            fftw_plan_with_nthreads(1);
#endif
        }
    }
#endif

    pme_fft_plan_fwd_ = fftw_mpi_plan_dft_r2c_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_real_grid_, pme_complex_grid_, pme_comm_, FFTW_MEASURE);
    pme_fft_plan_bwd_ = fftw_mpi_plan_dft_c2r_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_complex_grid_, pme_real_grid_, pme_comm_, FFTW_MEASURE);

    pme_b_spline_mod_x_ = pme_compute_bspline_moduli(ewald_grid_x_, ewald_spline_order_);
    pme_b_spline_mod_y_ = pme_compute_bspline_moduli(ewald_grid_y_, ewald_spline_order_);
    pme_b_spline_mod_z_ = pme_compute_bspline_moduli(ewald_grid_z_, ewald_spline_order_);

    ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
    initialized_ = true;

    int pme_rank = 0;
    MPI_Comm_rank(pme_comm_, &pme_rank);
    std::cout << "[Ewald/PME][reciprocal][distributed] pme_rank " << pme_rank
              << " grid " << ewald_grid_x_ << "x" << ewald_grid_y_ << "x" << ewald_grid_z_
              << " (r2c), local_nx=" << local_nx_ << " at x_start=" << local_x_start_
              << ", alpha=" << ewald_alpha_ << " 1/A, spline_order=" << ewald_spline_order_
              << ", cutoff=" << cutoff << " A" << std::endl;
}
#endif  // DEEPSCALEMD_HAVE_FFTW_MPI

void PmeReciprocalEngine::free_resources() {
    if (pme_fft_plan_fwd_) { fftw_destroy_plan(pme_fft_plan_fwd_); pme_fft_plan_fwd_ = nullptr; }
    if (pme_fft_plan_bwd_) { fftw_destroy_plan(pme_fft_plan_bwd_); pme_fft_plan_bwd_ = nullptr; }
    if (pme_real_grid_) { fftw_free(pme_real_grid_); pme_real_grid_ = nullptr; }
    pme_complex_grid_ = nullptr;
    pme_green_.clear();
    pme_green_energy_.clear();
    pme_green_virial_.clear();
    pme_green_cached_lx_ = pme_green_cached_ly_ = pme_green_cached_lz_ = 0.0;
    initialized_ = false;
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    distributed_ = false;
    pme_comm_ = MPI_COMM_NULL;
    local_nx_ = 0;
    local_x_start_ = 0;
    alloc_local_complex_ = 0;
#endif
}

void PmeReciprocalEngine::ensure_green_cached(const SimulationBox& box) {
    if (!pme_green_.empty() &&
        pme_green_cached_lx_ == box.lx &&
        pme_green_cached_ly_ == box.ly &&
        pme_green_cached_lz_ == box.lz) {
        return;
    }

    const int nx = ewald_grid_x_, ny = ewald_grid_y_, nz = ewald_grid_z_;
    const int nzh = pme_grid_z_half_;
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    const int kx_begin = distributed_ ? static_cast<int>(local_x_start_) : 0;
    const int kx_count  = distributed_ ? static_cast<int>(local_nx_) : nx;
#else
    const int kx_begin = 0;
    const int kx_count = nx;
#endif
    const size_t num_k_local = static_cast<size_t>(kx_count) * ny * nzh;
    pme_green_.assign(num_k_local, 0.0);
    pme_green_energy_.assign(num_k_local, 0.0);
    pme_green_virial_.assign(num_k_local, 0.0);

    const Real inv_box_x = 1.0 / box.lx, inv_box_y = 1.0 / box.ly, inv_box_z = 1.0 / box.lz;
    const Real alpha_sq = ewald_alpha_ * ewald_alpha_;
    const Real inv_4alpha_sq = 1.0 / (4.0 * alpha_sq);
    const Real inv_2alpha_sq = 1.0 / (2.0 * alpha_sq);
    const Real two_pi = 2.0 * M_PI;

    // lx indexes this rank's LOCAL k-slab array; kx is the corresponding
    // GLOBAL wavevector index the physics formula needs (identical to the
    // non-distributed loop when kx_begin==0 and kx_count==nx).
    #pragma omp parallel for schedule(static)
    for (int lx = 0; lx < kx_count; ++lx) {
        const int kx = kx_begin + lx;
        const Real kxw = (kx <= nx / 2 ? kx : kx - nx) * two_pi * inv_box_x;
        const Real bx = pme_b_spline_mod_x_[kx];
        for (int ky = 0; ky < ny; ++ky) {
            const Real kyw = (ky <= ny / 2 ? ky : ky - ny) * two_pi * inv_box_y;
            const Real by = pme_b_spline_mod_y_[ky];
            const Real bxy_sq = (bx * bx) * (by * by);
            for (int kz = 0; kz < nzh; ++kz) {
                const size_t idx = (static_cast<size_t>(lx) * ny + ky) * nzh + kz;
                if (kx == 0 && ky == 0 && kz == 0) continue;  // arrays already 0 at DC
                const Real kzw = kz * two_pi * inv_box_z;     // kz <= nz/2 by construction
                const Real k_sq = kxw * kxw + kyw * kyw + kzw * kzw;
                const Real bz = pme_b_spline_mod_z_[kz];
                const Real m_sq = bxy_sq * (bz * bz);
                const Real green = ewald_prefactor_ * std::exp(-k_sq * inv_4alpha_sq)
                                 / (k_sq * m_sq + 1e-12);

                const Real w = (kz == 0 || 2 * kz == nz) ? 1.0 : 2.0;

                pme_green_[idx] = green;
                pme_green_energy_[idx] = 0.5 * w * green;
                pme_green_virial_[idx] = w * green * (1.0 + k_sq * inv_2alpha_sq);
            }
        }
    }

    pme_green_cached_lx_ = box.lx;
    pme_green_cached_ly_ = box.ly;
    pme_green_cached_lz_ = box.lz;
}

void PmeReciprocalEngine::spread_charges(const ParticleData& particles, size_t local_particles,
                                         const SimulationBox& box) {
    const Real inv_box_x = 1.0/box.lx, inv_box_y = 1.0/box.ly, inv_box_z = 1.0/box.lz;
    pme_spread_charges(
        particles.x.data(), particles.y.data(), particles.z.data(), particles.charge.data(),
        pme_real_grid_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, pme_grid_z_pad_,
        inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles,
        pme_charge_spread_priv_);
}

size_t PmeReciprocalEngine::real_grid_size() const {
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    if (distributed_) {
        return static_cast<size_t>(local_nx_) * ewald_grid_y_ * pme_grid_z_pad_;
    }
#endif
    return static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_pad_;
}

size_t PmeReciprocalEngine::num_k() const {
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    if (distributed_) {
        return static_cast<size_t>(local_nx_) * ewald_grid_y_ * pme_grid_z_half_;
    }
#endif
    return static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_half_;
}

Real PmeReciprocalEngine::solve(Real& virial_out) {
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    if (distributed_) {
        // Collective: every rank in pme_comm_ must call this together.
        fftw_mpi_execute_dft_r2c(pme_fft_plan_fwd_, pme_real_grid_, pme_complex_grid_);
        Real d_virial = 0.0;
        Real d_energy = pme_apply_green(
            pme_complex_grid_, pme_green_.data(), pme_green_energy_.data(),
            pme_green_virial_.data(), num_k(), d_virial);
        fftw_mpi_execute_dft_c2r(pme_fft_plan_bwd_, pme_complex_grid_, pme_real_grid_);
        virial_out = d_virial;
        return d_energy;  // LOCAL partial sum -- caller must MPI_Allreduce over pme_comm_
    }
#endif
    fftw_execute(pme_fft_plan_fwd_);   // rho(r) -> rho(k), R2C
    Real virial = 0.0;
    Real energy = pme_apply_green(
        pme_complex_grid_, pme_green_.data(), pme_green_energy_.data(),
        pme_green_virial_.data(), num_k(), virial);
    fftw_execute(pme_fft_plan_bwd_);   // Phi(k) -> Phi(r), C2R
    virial_out = virial;
    return energy;
}

void PmeReciprocalEngine::interpolate_forces(ParticleData& particles, size_t local_particles,
                                             const SimulationBox& box) {
    const Real inv_box_x = 1.0/box.lx, inv_box_y = 1.0/box.ly, inv_box_z = 1.0/box.lz;
    pme_interpolate_forces_from_potential(
        particles.x.data(), particles.y.data(), particles.z.data(), particles.charge.data(),
        pme_real_grid_,
        particles.fx.data(), particles.fy.data(), particles.fz.data(),
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, pme_grid_z_pad_,
        inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles);
}
