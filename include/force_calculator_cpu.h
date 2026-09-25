#ifndef FORCE_CALCULATOR_CPU_H
#define FORCE_CALCULATOR_CPU_H

#include "force_calculator.h"
#include "md_system.h"
#include "pme_reciprocal_engine.h"
#include <vector>
#include <cstdint>
#include <chrono>
#include <fftw3.h>
#include <mpi.h>

class Topology;
class CpuSpatialVerletList;

class ForceCalculatorCPU : public ForceCalculator {
public:
    ForceCalculatorCPU() = default;
    virtual ~ForceCalculatorCPU();

    virtual void initialize() override;
    virtual void finalize() override;
    virtual void compute_forces(MDSystem& system) override;
    virtual void calculate_kinetic_energy(MDSystem& system) override;

    void set_timing_breakdown_enabled(bool enabled) override { timing_breakdown_enabled_ = enabled; }
    bool is_timing_breakdown_enabled() const override { return timing_breakdown_enabled_; }
    const std::vector<ForceComponentTiming>& get_last_force_component_timings() const override {
        return last_force_component_timings_;
    }

    const std::vector<int>& get_lj_pair_counts() const override {
        return lj_pair_counts_;
    }

    void set_mixed_precision(bool enabled) override;

private:
    void compute_bond_forces(MDSystem& system);
    void compute_angle_forces(MDSystem& system);
    void compute_dihedral_forces(MDSystem& system);
    void compute_improper_forces(MDSystem& system);
    void compute_torsion_forces(MDSystem& system, const std::vector<MDSystem::LocalDihedral>& torsions);
    void compute_special_pair_forces(MDSystem& system);

    void compute_electrostatic_forces_reciprocal(MDSystem& system);

    void pme_reciprocal_begin(MDSystem& system);
    void pme_reciprocal_end(MDSystem& system);
    MPI_Request pme_reduce_req_ = MPI_REQUEST_NULL;
    bool pme_reduce_pending_ = false;
    double pme_time_ms_ = 0.0;

    template <typename Fn>
    void run_nonbonded_with_pme_overlap(MDSystem& system, bool do_coulomb, const char* label, Fn&& body) {
        if (do_coulomb) {
            ensure_ewald_initialized(system);
            pme_reciprocal_begin(system);
        }
        using clock = std::chrono::high_resolution_clock;
        clock::time_point t0;
        if (timing_breakdown_enabled_) t0 = clock::now();

        body();

        if (timing_breakdown_enabled_) {
            last_force_component_timings_.push_back(
                {label, std::chrono::duration<double, std::milli>(clock::now() - t0).count()});
        }
        if (do_coulomb) {
            pme_reciprocal_end(system);
        }
    }
    void compute_tiled_nonbonded(MDSystem& system, bool do_lj, bool do_coulomb);
    void compute_graph_nonbonded(MDSystem& system, bool do_lj, bool do_coulomb);
    void ensure_ewald_initialized(MDSystem& system);
    void free_ewald_resources();

    void ensure_exclusion_cache_built(const Topology& topology);
    std::vector<int> excl_cache_offsets_;   // size num_atoms + 1, indexed by global_id
    std::vector<int> excl_cache_partners_;  // flat, sorted per atom's [offset, offset+1) range
    bool excl_cache_built_ = false;

    void compute_electrostatic_forces_fmm(MDSystem& system);

    bool timing_breakdown_enabled_ = false;
    std::vector<ForceComponentTiming> last_force_component_timings_;
    std::vector<int> lj_pair_counts_;
    bool graph_kernel_logged_ = false;
    bool ewald_initialized_ = false;  // true once the real-space table + pme_engine_ are set up
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;

    // Reciprocal-space grid/FFT/Green's-function machinery, extracted into
    // its own class since it now has two callers: this one (legacy in-process
    // path, and the PP side of a PP/PME rank split) and PmeServer (the
    // dedicated PME-group side of a split, see pme_server.h). See
    // pme_reciprocal_engine.h for the distributed-FFT revert history this
    // class's FFT usage must not repeat.
    PmeReciprocalEngine pme_engine_;

    std::vector<Real> ewald_rs_table_;       // erfc-screened (normal pairs), interleaved e/f
    std::vector<Real> ewald_rs_table_excl_;  // erf-based (excluded pairs), interleaved e/f
    Real ewald_rs_table_r2_min_ = 0.0;
    Real ewald_rs_table_inv_spacing_ = 0.0;
    int ewald_rs_table_size_ = 0;            // number of NODES (not doubles)

    void ensure_lj_coeff_table_built(const ForceFieldParams& params);
    std::vector<Real> lj_c6c12_;
    int lj_coeff_num_types_ = 0;

    std::vector<Real> tile_cx_, tile_cy_, tile_cz_, tile_rad_;

    void repack_tile_soa(const ParticleData& particles, const CpuSpatialVerletList& nlist);
    std::vector<Real> tile_px_, tile_py_, tile_pz_, tile_q_;
    std::vector<int>  tile_type_, tile_gid_;

    void ensure_mixed_precision_tables();
    bool mixed_precision_nonbonded_ = false;
    bool mixed_precision_probed_ = false;
    std::vector<float> tile_fx_, tile_fy_, tile_fz_, tile_fq_;
    std::vector<float> lj_c6c12_f_;
    std::vector<float> ewald_rs_table_f_;

    void ensure_tile_exclusion_masks(const CpuSpatialVerletList& nlist);
    std::vector<std::uint64_t> tile_pair_excl_;
    std::uint64_t tile_excl_generation_ = ~static_cast<std::uint64_t>(0);

    void ensure_lj_thread_buffers(int nthreads, size_t local_particles);
    std::vector<Real> lj_thread_fx_, lj_thread_fy_, lj_thread_fz_;  // size nthreads*local_particles

    void ensure_lj_tile_masks(const CpuSpatialVerletList& nlist, size_t local_particles);
    std::vector<uint8_t> lj_tile_valid_mask_, lj_tile_local_mask_;
    std::vector<std::uint64_t> lj_same_tile_mask_;
    std::uint64_t lj_tile_mask_generation_ = ~static_cast<std::uint64_t>(0);
    size_t lj_tile_mask_local_particles_ = ~static_cast<size_t>(0);

    void ensure_lj_tile_excl_col(const CpuSpatialVerletList& nlist);
    std::vector<std::uint64_t> lj_tile_excl_col_;
    std::uint64_t lj_tile_excl_col_generation_ = ~static_cast<std::uint64_t>(0);
};

#endif // FORCE_CALCULATOR_CPU_H
