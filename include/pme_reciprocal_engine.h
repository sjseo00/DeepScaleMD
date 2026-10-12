#ifndef PME_RECIPROCAL_ENGINE_H
#define PME_RECIPROCAL_ENGINE_H

#include "md_system.h"
#include <vector>
#include <fftw3.h>
#include <mpi.h>
#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
#include <fftw3-mpi.h>
#endif

// Owns the PME reciprocal-space grid/FFT/Green's-function machinery: charge
// spreading, forward FFT, Green's-function (Poisson) solve, inverse FFT, and
// force interpolation. Extracted out of ForceCalculatorCPU so it has two
// callers: the legacy in-process (no rank split) path and the dedicated
// PME-group server (PmeServer) used when force_calculation.pme_ranks > 0.
//
// This class does NOT do any MPI itself -- callers own the cross-rank
// charge-reduce / potential-broadcast (see ForceCalculatorCPU::pme_reciprocal_begin/end
// and PmeServer::step()), since those differ depending on whether the caller
// is running in the legacy single-group mode or the split PP/PME mode.
//
// NOTE ON HISTORY AND RISK: a genuinely distributed FFT (fftw_mpi_*, every
// rank owning and transforming its own x-slab, MPI_Ireduce_scatter/
// MPI_Allgatherv instead of reduce-to-rank-0/broadcast) was tried in this
// codebase once before and reverted: both the plain (non-transposed) and
// FFTW_MPI_TRANSPOSED_OUT/IN variants produced intermittent, run-to-run
// non-deterministic energy blow-ups (same input, same rank count, sometimes
// stable, sometimes diverging by step ~100) that could not be root-caused
// without a build/test environment available at the time. That prior attempt
// ran on the SAME ranks that were also doing real-space work; this class's
// distributed mode (ensure_initialized_distributed/below) is used only by a
// fully dedicated PME rank group (see PmeServer), which removes one plausible
// cause (interaction with concurrent real-space OpenMP/MPI activity in the
// same process) but does NOT remove the more fundamental risk: this is the
// same class of mechanism (custom MPI exchange around a distributed 3D FFT),
// re-implemented without the ability to compile or run it here. Distrust a
// single "looks fine" run -- verify with many repeated identical runs on
// real hardware before trusting this for production results, exactly as the
// project's own prior incident demands.
class PmeReciprocalEngine {
public:
    PmeReciprocalEngine() = default;
    ~PmeReciprocalEngine();
    PmeReciprocalEngine(const PmeReciprocalEngine&) = delete;
    PmeReciprocalEngine& operator=(const PmeReciprocalEngine&) = delete;

    // Idempotent: allocates the grid/FFT plans on first call for a given
    // cutoff; on later calls (box volume changed under a barostat) just
    // refreshes the cached prefactor. Must be called before any other method.
    void ensure_initialized(Real cutoff, const SimulationBox& box);

    // force_calculation.pme_grid_spacing (A) from YAML. Must be set before the
    // first ensure_initialized()/ensure_initialized_distributed() call to take effect.
    void set_grid_spacing_override(Real spacing_angstrom) { grid_spacing_override_ = spacing_angstrom; }

#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    // Distributed variant: this rank owns only an x-slab of the grid
    // (local_nx() planes starting at local_x_start()), determined by
    // fftw_mpi_local_size_3d() over pme_comm. Used only by PmeServer when
    // its PME group has more than one rank (force_calculation.pme_ranks > 1)
    // -- see the class-level NOTE ON HISTORY AND RISK above before touching
    // this. Idempotent like ensure_initialized(); mutually exclusive with it
    // (a given engine instance is either distributed or not, for its whole
    // lifetime).
    void ensure_initialized_distributed(Real cutoff, const SimulationBox& box, MPI_Comm pme_comm);
    bool is_distributed() const { return distributed_; }
    ptrdiff_t local_nx() const { return local_nx_; }
    ptrdiff_t local_x_start() const { return local_x_start_; }
#endif

    void free_resources();
    bool initialized() const { return initialized_; }

    // Recomputes the cached Green's-function tables if box dimensions changed
    // since the last call. Must be called after ensure_initialized() and
    // before solve(), every step (cheap no-op when the box hasn't changed).
    void ensure_green_cached(const SimulationBox& box);

    // Spreads local atoms' charges onto the internal real grid, OpenMP-parallel.
    // Fully overwrites the grid (no separate clear needed).
    void spread_charges(const ParticleData& particles, size_t local_particles,
                        const SimulationBox& box);

    Real* grid_data() { return pme_real_grid_; }
    // In distributed mode, these are this rank's LOCAL slab sizes
    // (local_nx() planes), not the global grid size.
    size_t real_grid_size() const;
    size_t num_k() const;

    // Forward FFT -> Green's-function (Poisson) solve -> inverse FFT, in place
    // on grid_data(). Returns this rank's reciprocal energy and writes its
    // virial contribution. Non-distributed: caller must ensure grid_data()
    // holds the true GLOBAL (already-reduced) charge density; this executes
    // on one rank only. Distributed: grid_data() must hold this rank's
    // already-reduced LOCAL slab; solve() is a COLLECTIVE call that every
    // rank in pme_comm must make together (it runs fftw_mpi_execute_dft_*),
    // and the returned energy/virial are this rank's LOCAL partial sums --
    // the caller must MPI_Allreduce them over pme_comm to get the true total.
    Real solve(Real& virial_out);

    // Interpolates forces from grid_data() (already inverse-FFT'd potential)
    // into particles' fx/fy/fz for its local atoms. Adds to existing forces.
    void interpolate_forces(ParticleData& particles, size_t local_particles,
                            const SimulationBox& box);

    Real alpha() const { return ewald_alpha_; }
    int spline_order() const { return ewald_spline_order_; }
    int grid_x() const { return ewald_grid_x_; }
    int grid_y() const { return ewald_grid_y_; }
    int grid_z() const { return ewald_grid_z_; }
    int grid_z_pad() const { return pme_grid_z_pad_; }

private:
    Real grid_spacing_override_ = 0.0;  // 0 = automatic
    bool initialized_ = false;
    int ewald_grid_x_ = 0, ewald_grid_y_ = 0, ewald_grid_z_ = 0;
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;
    Real ewald_prefactor_ = 0.0;  // 4*pi*k_coulomb / box_volume

#ifdef DEEPSCALEMD_HAVE_FFTW_MPI
    bool distributed_ = false;
    MPI_Comm pme_comm_ = MPI_COMM_NULL;      // not owned; caller's communicator
    ptrdiff_t local_nx_ = 0;                 // this rank's slab: x in [local_x_start_, +local_nx_)
    ptrdiff_t local_x_start_ = 0;
    ptrdiff_t alloc_local_complex_ = 0;      // complex-element count fftw_mpi_local_size_3d told us to allocate
#endif

    Real* pme_real_grid_ = nullptr;
    fftw_complex* pme_complex_grid_ = nullptr;  // aliases pme_real_grid_
    int pme_grid_z_pad_ = 0;   // 2*(nz/2+1), the real grid's z stride
    int pme_grid_z_half_ = 0;  // nz/2+1, the complex grid's z extent
    std::vector<Real> pme_b_spline_mod_x_, pme_b_spline_mod_y_, pme_b_spline_mod_z_;
    fftw_plan pme_fft_plan_fwd_ = nullptr;  // r2c, rho(r) -> rho(k)
    fftw_plan pme_fft_plan_bwd_ = nullptr;  // c2r, Phi(k) -> Phi(r)

    std::vector<Real> pme_green_, pme_green_energy_, pme_green_virial_;
    Real pme_green_cached_lx_ = 0.0, pme_green_cached_ly_ = 0.0, pme_green_cached_lz_ = 0.0;

    std::vector<Real> pme_charge_spread_priv_;
};

#endif // PME_RECIPROCAL_ENGINE_H
