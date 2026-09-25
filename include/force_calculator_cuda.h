#ifndef FORCE_CALCULATOR_CUDA_H
#define FORCE_CALCULATOR_CUDA_H

#include "../include/force_calculator.h"
#include "../include/md_types.h"
#include "../include/pme_rank_split.h"
#include <dsdecompx.hpp>
#include <vector>
#include <string>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

class MDSystem;
class ParticleData;
class NeighborList;
class Topology;
class ForceFieldParams;
class SimulationBox;

class ForceCalculatorCUDA : public ForceCalculator,
                            public dsdecompx::ParticleInfo::Provider {
public:
#ifdef USE_CUDA
    ForceCalculatorCUDA();
    virtual ~ForceCalculatorCUDA();

    void initialize() override;
    void finalize() override;
    void compute_forces(MDSystem& system) override;
    void calculate_kinetic_energy(MDSystem& system) override;

    void initial_upload_to_device(MDSystem& system);     // Initial 1-time upload
    void download_from_device(MDSystem& system);         // Download for output/checkpoint
    void build_neighbor_list_on_device(MDSystem& system); // Build rank-local spatial tile graph on GPU
    void refresh_bonded_topology_on_device(MDSystem& system);
    
    void integrate_on_device(MDSystem& system, Real dt, const std::string& integrator_type);
    void complete_velocity_update_on_device(MDSystem& system, Real dt, const std::string& integrator_type);

    void snapshot_constraint_positions(MDSystem& system);
    bool apply_shake_on_device(MDSystem& system, Real dt,
                                int max_iterations = 500, Real position_tolerance = 1e-8);
    bool apply_rattle_on_device(MDSystem& system,
                                 int max_iterations = 500, Real velocity_tolerance = 1e-8);

    Real compute_max_displacement_on_device(size_t num_local_particles, const SimulationBox& box);
    void append_ghosts_to_device(MDSystem& system, size_t previous_local_count);
    void update_reference_positions_on_device(size_t num_local_particles);  // NEW: Update reference positions after neighbor list rebuild

    struct FireReduction {
        Real power = 0.0;      // sum_i F_i . v_i
        Real f_norm_sq = 0.0;  // sum_i F_i . F_i
        Real v_norm_sq = 0.0;  // sum_i v_i . v_i
        Real f_max_sq = 0.0;   // max_i (F_i . F_i)
    };
    FireReduction compute_fire_reduction_on_device(size_t local_particles);
    void apply_fire_update_on_device(size_t local_particles, const SimulationBox& box,
                                     bool reset_velocity, bool mix_velocity,
                                     Real alpha, Real v_norm, Real f_norm, Real dt);

    Real* get_device_x() { return d_x; }
    Real* get_device_y() { return d_y; }
    Real* get_device_z() { return d_z; }
    Real* get_device_vx() { return d_vx; }
    Real* get_device_vy() { return d_vy; }
    Real* get_device_vz() { return d_vz; }
    Real* get_device_mass() { return d_mass; }
    Real* get_device_charge() { return d_charge; }
    int* get_device_type() { return d_type; }
    int* get_device_global_id() { return d_global_id; }
    cudaStream_t get_force_stream() const { return force_stream; }
    int get_device_id() const { return device_id; }

    dsdecompx::ParticleView particle_info_view(size_t count) override;
    bool ensure_device_particle_capacity(size_t required, size_t preserve_count);
    dsdecompx::StorageResizeResult resize_particle_info(
        size_t required, size_t preserve_count) override;
    
    Real* get_device_fx() { return d_fx; }
    Real* get_device_fy() { return d_fy; }
    Real* get_device_fz() { return d_fz; }
    
    void upload_forces_to_device(const ParticleData& particles, size_t num_particles);

    void download_positions_for_ml(ParticleData& particles, size_t num_particles);

    void download_forces_for_ml(ParticleData& particles, size_t num_particles);

    void upload_forces_from_ml(const ParticleData& particles, size_t num_particles);

    void register_ml_pinned_buffers(ParticleData& particles, size_t num_particles);
    void unregister_ml_pinned_buffers(ParticleData& particles);

    void pre_register_mlip_sources();
    void set_mlip_runtime(void* raw_runtime);
    void set_builtin_forces_enabled(bool enabled) { builtin_forces_enabled_ = enabled; }
    void set_mixed_precision(bool enabled) { mixed_precision_enabled_ = enabled; }
    bool is_mixed_precision() const { return mixed_precision_enabled_; }
    void set_pme_cluster_spread_enabled(bool enabled) { pme_use_cluster_spread_ = enabled; }
    bool is_pme_cluster_spread_enabled() const { return pme_use_cluster_spread_; }
    void compute_mlip_runtime_forces(MDSystem& system);

    void set_timing_breakdown_enabled(bool enabled) override { timing_breakdown_enabled_ = enabled; }
    bool is_timing_breakdown_enabled() const override { return timing_breakdown_enabled_; }
    const std::vector<ForceComponentTiming>& get_last_force_component_timings() const override {
        return last_force_component_timings_;
    }

    bool nlist_dirty_ = true;

    friend void mlip_bridge_register_md_sources(void* fc_cuda_ptr);

protected:
    void compute_electrostatic_forces_real_space(MDSystem& system);
    void compute_electrostatic_forces_reciprocal(MDSystem& system);
    void compute_electrostatic_forces_ewald(MDSystem& system);
    void ensure_ewald_initialized(MDSystem& system);
    void free_ewald_resources();
    void compute_bond_forces(MDSystem& system);
    void compute_angle_forces(MDSystem& system);
    void compute_dihedral_forces(MDSystem& system);
    void compute_improper_forces(MDSystem& system);
    void compute_special_pair_forces(MDSystem& system);

    void build_coo_edge_list_on_device(MDSystem& system);
    void interleave_positions_on_device(int num_total);
    void ensure_graph_on_device(MDSystem& system);
    void rebuild_graph_from_tiles_on_device(MDSystem& system);
    void ensure_graph_edge_capacity(int64_t edges);

    void allocate_device_memory(size_t num_particles);

    void ensure_tile_pair_stats_enabled();
    void log_tile_pair_stats_gpu();
    
    void upload_bonded_topology_to_device(MDSystem& system);
    void free_bonded_topology_device();

    Real* d_x;              // Particle x coordinates
    Real* d_y;              // Particle y coordinates  
    Real* d_z;              // Particle z coordinates
    Real* d_fx;             // Force x components
    Real* d_fy;             // Force y components
    Real* d_fz;             // Force z components
    Real* d_vx;             // Velocity x components
    Real* d_vy;             // Velocity y components
    Real* d_vz;             // Velocity z components
    Real* d_mass;           // Particle masses
    Real* d_charge;         // Particle charges
    int* d_type;            // Particle types
    int* d_global_id;       // Global particle IDs (for MPI) - int to match ParticleData
    
    Real* d_x_ref;          // Reference positions for displacement check
    Real* d_y_ref;
    Real* d_z_ref;
    Real* d_max_disp_sq;    // Device scalar for max displacement squared
    Real* d_fire_reduce_;   // [power, f_norm_sq, v_norm_sq, f_max_sq], lazily allocated

    int* d_tile_atoms = nullptr;    // [num_lj_tiles_ * LJ_TILE_SIZE], atom index per slot (-1 = padding)
    int* d_tile_pair_i = nullptr;   // [num_lj_tile_pairs_], tile id in the "i" role (sorted by this key)
    int* d_tile_pair_j = nullptr;   // [num_lj_tile_pairs_], tile id in the "j" role, grouped by d_tile_pair_offsets
    int* d_tile_pair_offsets = nullptr; // [num_lj_tiles_ + 1] CSR offsets into d_tile_pair_j, keyed by home tile
    unsigned long long* d_tile_pair_verlet_mask_ = nullptr; // [num_lj_tile_pairs_], bit=(j_slot*8+i_lane)
    int  num_lj_tiles_ = 0;
    int  num_lj_tile_pairs_ = 0;
    unsigned long long pairs_inside_rc_ = 0;
    unsigned long long pairs_inside_rc_plus_skin_ = 0;

    unsigned long long* d_tile_pair_stats_ = nullptr;
    bool tile_pair_stats_probed_ = false;
    bool tile_pair_stats_enabled_ = false;


    long long* d_force_fixed_x_ = nullptr;
    long long* d_force_fixed_y_ = nullptr;
    long long* d_force_fixed_z_ = nullptr;

    Real* d_epsilon_mixed; // LJ epsilon mixing rules (1D row-major array)
    Real* d_sigma_mixed;   // LJ sigma mixing rules (1D row-major array)
    unsigned long long* d_exclusion_keys; // sorted global atom-pair keys
    size_t num_exclusions;
    int* d_exclusion_offsets;  // global-ID indexed CSR offsets
    int* d_exclusion_partners; // short per-atom exclusion lists
    size_t exclusion_atom_count;

    Real* d_potential_energy;
    Real* d_kinetic_energy;
    Real* d_virial;              // NEW: Device memory for virial calculation

    bool ml_buffers_pinned = false;

    size_t num_bonds_cached;
    size_t* d_bond_atom1;
    size_t* d_bond_atom2;
    Real*   d_bond_k;
    Real*   d_bond_r0;
    
    size_t num_angles_cached;
    size_t* d_angle_atom1;
    size_t* d_angle_atom2;
    size_t* d_angle_atom3;
    Real*   d_angle_k;
    Real*   d_angle_theta0;
    
    size_t num_dihedrals_cached;
    size_t* d_dihedral_atom1;
    size_t* d_dihedral_atom2;
    size_t* d_dihedral_atom3;
    size_t* d_dihedral_atom4;
    Real*   d_dihedral_k;
    Real*   d_dihedral_phi0;
    int*    d_dihedral_n;

    size_t num_impropers_cached;
    size_t* d_improper_atom1;
    size_t* d_improper_atom2;
    size_t* d_improper_atom3;
    size_t* d_improper_atom4;
    Real*   d_improper_k;
    Real*   d_improper_phi0;
    int*    d_improper_n;

    size_t num_special_pairs_cached;
    size_t* d_special_pair_atom1;
    size_t* d_special_pair_atom2;
    Real*   d_special_pair_lj_scale;
    Real*   d_special_pair_coulomb_scale;
    Real*   d_special_pair_sigma;
    Real*   d_special_pair_epsilon;

    size_t num_constraints_cached;
    size_t* d_constraint_atom1;
    size_t* d_constraint_atom2;
    Real*   d_constraint_distance;

    size_t constraint_scratch_capacity_;
    Real* d_constraint_pre_x_;
    Real* d_constraint_pre_y_;
    Real* d_constraint_pre_z_;
    Real* d_constraint_max_error_; // single device scalar, reused every iteration
    void ensure_constraint_scratch_capacity(size_t n);

    size_t allocated_particles;
    int max_types;
    bool device_data_initialized;  // Track if initial upload completed
    bool bonded_topology_on_device; // Track if bonded topology is uploaded
    std::vector<int> bonded_layout_global_ids_;
    size_t bonded_layout_local_count_ = static_cast<size_t>(-1);
    
    int block_size;         // CUDA block size for kernels
    int device_id;          // GPU device ID
    
    cudaStream_t force_stream;
    cudaStream_t energy_stream;
    
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float kernel_time_ms;
    
    int rank;
    int mpi_size;
    size_t allocated_size;
    
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    
    int* d_debug_counters;

    void*    mlip_runtime_raw_      = nullptr;  // non-owning ptr to MlipRuntime
    bool     mlip_runtime_enabled_  = false;
    bool     builtin_forces_enabled_ = true;
    bool     mixed_precision_enabled_ = false;

    int      num_local_cached_      = 0;
    int      num_ghost_cached_      = 0;

    int32_t* d_edge_src_         = nullptr;
    int32_t* d_edge_dst_         = nullptr;
    int32_t* d_edge_coo_packed_  = nullptr;
    int32_t* d_tmp_edge_src_     = nullptr;
    int32_t* d_tmp_edge_dst_     = nullptr;
    int64_t  num_edges_          = 0;
    int64_t  cap_edges_          = 0;

    int*     d_graph_offsets_     = nullptr; // [num_local + 1]
    int*     d_graph_degrees_     = nullptr; // [num_local]
    int*     d_graph_cursors_     = nullptr; // [num_local]
    int      graph_atom_capacity_ = 0;
    Real*    d_graph_ref_x_       = nullptr;
    Real*    d_graph_ref_y_       = nullptr;
    Real*    d_graph_ref_z_       = nullptr;
    int      graph_ref_capacity_  = 0;
    bool     graph_dirty_         = true;
    Real     graph_active_skin_   = 0.0;

    double* d_pos_interleaved_   = nullptr;
    int     cap_pos_interleaved_ = 0;

    double* d_mlip_forces_       = nullptr;
    int     cap_mlip_forces_     = 0;

    float*   d_edge_vectors_     = nullptr;
    int64_t  cap_edge_vectors_   = 0;

    int64_t* d_batch_            = nullptr;
    int      cap_batch_          = 0;
    int64_t* d_num_atoms_        = nullptr;

    double*  d_cell_             = nullptr;
    int*     d_pbc_buf_          = nullptr;

    int*     d_mlip_error_flag_  = nullptr;
    
    size_t max_neighbors_allocated;
    
    size_t current_compact_size;
    std::vector<int> current_compact_map;
    std::vector<bool> current_is_local;
    
    bool ewald_initialized_ = false;
    int ewald_grid_x_ = 0, ewald_grid_y_ = 0, ewald_grid_z_ = 0;
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;
    Real ewald_prefactor_ = 0.0;  // 4*pi*k_coulomb / box_volume (updated on box change)

    Real* d_pme_charge_grid_real_          = nullptr;
    cufftDoubleComplex* d_pme_charge_grid_freq_ = nullptr;
    Real* d_pme_b_spline_mod_x_ = nullptr;
    Real* d_pme_b_spline_mod_y_ = nullptr;
    Real* d_pme_b_spline_mod_z_ = nullptr;
    cufftHandle pme_fft_plan_forward_ = 0;   // D2Z: real charge grid -> half-spectrum
    cufftHandle pme_fft_plan_inverse_ = 0;   // Z2D: half-spectrum -> real field grid
    cudaStream_t pme_stream_ = nullptr;
    cudaEvent_t force_clear_event_ = nullptr;
    cudaEvent_t pme_grid_ready_event_ = nullptr;
    // Host staging buffer for the PP-side of a PP/PME rank split
    // (force_calculation.pme_ranks > 0): packs/unpacks d_pme_charge_grid_real_
    // around the intercommunicator MPI_Ireduce/MPI_Bcast to the PME group --
    // same protocol regardless of PME group size (a PME group of more than
    // one rank, always the CPU FFTW-MPI fallback here, funnels through its
    // own leader internally; see PmeServer::step_distributed()). Unused
    // (empty) when no split is configured.
    std::vector<Real> pme_split_host_grid_buffer_;

    bool device_supports_clusters_ = false;
    bool pme_use_cluster_spread_ = false;

    bool timing_breakdown_enabled_ = false;
    bool graph_kernel_logged_ = false;
    bool force_timing_events_ready_ = false;
    cudaEvent_t force_timing_event_start_ = nullptr;
    cudaEvent_t force_timing_event_bonded_ = nullptr;
    cudaEvent_t force_timing_event_mlip_ = nullptr;
    cudaEvent_t force_timing_event_lj_ = nullptr;
    cudaEvent_t force_timing_event_electrostatic_ = nullptr;
    cudaEvent_t force_timing_event_special_ = nullptr;
    std::vector<ForceComponentTiming> last_force_component_timings_;
#else
    ForceCalculatorCUDA() {
        throw std::runtime_error("ForceCalculatorCUDA was instantiated without USE_CUDA defined");
    }
    virtual ~ForceCalculatorCUDA() = default;

    void initialize() override {}
    void finalize() override {}
    void compute_forces(MDSystem&) override {
        throw std::runtime_error("CUDA backend not available (compiled without USE_CUDA)");
    }
    void calculate_kinetic_energy(MDSystem&) override {
        throw std::runtime_error("CUDA backend not available (compiled without USE_CUDA)");
    }
    dsdecompx::StorageResizeResult resize_particle_info(
        size_t, size_t) override {
        throw std::runtime_error("CUDA backend not available (compiled without USE_CUDA)");
    }
    dsdecompx::ParticleView particle_info_view(size_t) override {
        throw std::runtime_error("CUDA backend not available (compiled without USE_CUDA)");
    }
#endif
};

#ifdef USE_CUDA

__global__ void clear_forces_kernel(Real* fx, Real* fy, Real* fz, size_t n);


__device__ void pme_compute_bspline_weights(Real u, int order, Real* weights, int* grid_indices);

__device__ void pme_compute_bspline_weights_and_derivatives(
    Real u, int order, Real* weights, Real* derivatives, int* grid_indices);

__global__ void pme_clear_grid_kernel(Real* grid, size_t grid_size);

__global__ void pme_spread_charges_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, Real* charge_grid,
    const int* __restrict__ spatial_particle_order, size_t spread_work_items,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles);

__global__ void pme_solve_poisson_kernel(
    cufftDoubleComplex* fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* reciprocal_energy_out, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz, int grid_nzh,
    Real inv_box_x, Real inv_box_y, Real inv_box_z);

__global__ void pme_interpolate_forces_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge,
    const Real* __restrict__ potential_grid,
    Real* fx, Real* fy, Real* fz,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles);

__global__ void pme_self_energy_kernel(
    const Real* __restrict__ charge, Real* potential_energy,
    Real k_coulomb, Real alpha, size_t num_particles);

__global__ void pme_reciprocal_virial_kernel(
    const cufftDoubleComplex* __restrict__ fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* virial, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz, int grid_nzh,
    Real inv_box_x, Real inv_box_y, Real inv_box_z);

__global__ void compute_bond_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    size_t* bond_atom1, size_t* bond_atom2,
    Real* bond_k, Real* bond_r0,
    size_t num_bonds,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles);

__global__ void compute_angle_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    size_t* angle_atom1, size_t* angle_atom2, size_t* angle_atom3,
    Real* angle_k, Real* angle_theta0,
    size_t num_angles,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles);

__global__ void compute_dihedral_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    size_t* dihedral_atom1, size_t* dihedral_atom2,
    size_t* dihedral_atom3, size_t* dihedral_atom4,
    Real* dihedral_k, Real* dihedral_phi0, int* dihedral_n,
    size_t num_dihedrals,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles);

__global__ void calculate_kinetic_energy_kernel(
    Real* vx, Real* vy, Real* vz,
    Real* mass,
    Real* kinetic_energy,
    size_t n);

template <typename T>
__device__ inline void apply_minimum_image_device(
    T& dx, T& dy, T& dz,
    T box_x, T box_y, T box_z,
    T inv_box_x, T inv_box_y, T inv_box_z);

// Drives the dedicated PME-group side of a PP/PME rank split
// (force_calculation.pme_ranks > 0) on a CUDA build: owns no atoms, runs no
// dsDecompX, and does nothing but receive a reduced charge grid from the PP
// group, run it through the SAME cuFFT kernels ForceCalculatorCUDA uses
// (pme_spread_charges_kernel is not used here -- PP ranks spread their own
// atoms; this class only clears/reduces-into/solves/broadcasts the grid),
// and broadcast the potential grid + reciprocal energy/virial back.
//
// Implemented in force_calculator_cuda.cu (same translation unit as the
// pme_*_kernel definitions it launches) rather than a separate .cu file:
// this project does not build with CUDA_SEPARABLE_COMPILATION, so a kernel
// launch from a different translation unit than the kernel's definition
// would not link. See pme_reciprocal_engine.h for the CPU-side equivalent
// and the distributed-FFT revert history that both must avoid repeating.
//
// Cross-rank transfer uses host-staged buffers (cudaMemcpy D2H/H2D around a
// plain MPI_Ireduce/MPI_Bcast over the intercommunicator), not GPU-aware
// MPI -- safer default given this path cannot be verified against a real
// CUDA-aware MPI build in this environment; revisit if profiling shows the
// staging copy matters at the target PME grid sizes.
class PmeServerCuda {
public:
    explicit PmeServerCuda(PmePpSplit& split);
    ~PmeServerCuda();
    PmeServerCuda(const PmeServerCuda&) = delete;
    PmeServerCuda& operator=(const PmeServerCuda&) = delete;

    void run(int total_steps, Real cutoff);

private:
    void step();
    void ensure_initialized();
    void free_resources();

    PmePpSplit& split_;
    SimulationBox box_{};
    Real cutoff_ = 0.0;
    int device_id_ = 0;

    bool ewald_initialized_ = false;
    int ewald_grid_x_ = 0, ewald_grid_y_ = 0, ewald_grid_z_ = 0;
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;
    Real ewald_prefactor_ = 0.0;

    Real* d_pme_charge_grid_real_ = nullptr;
    cufftDoubleComplex* d_pme_charge_grid_freq_ = nullptr;
    Real* d_pme_b_spline_mod_x_ = nullptr;
    Real* d_pme_b_spline_mod_y_ = nullptr;
    Real* d_pme_b_spline_mod_z_ = nullptr;
    Real* d_recip_energy_ = nullptr;
    Real* d_recip_virial_ = nullptr;
    cufftHandle pme_fft_plan_forward_ = 0;
    cufftHandle pme_fft_plan_inverse_ = 0;
    cudaStream_t pme_stream_ = nullptr;

    std::vector<Real> host_grid_buffer_;  // host staging for the intercomm reduce/bcast
};

#endif // USE_CUDA

#endif // FORCE_CALCULATOR_CUDA_H
