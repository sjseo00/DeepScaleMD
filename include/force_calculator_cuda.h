#ifndef FORCE_CALCULATOR_CUDA_H
#define FORCE_CALCULATOR_CUDA_H

#include "../include/force_calculator.h"
#include "../include/md_types.h"
#include <dsdecompx.hpp>
#include <vector>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

// Forward declarations
class MDSystem;
class ParticleData;
class NeighborList;
class Topology;
class ForceFieldParams;
class SimulationBox;

/**
 * @brief CUDA-accelerated force calculator for molecular dynamics simulations
 * 
 * This class implements force calculations using CUDA kernels for GPU acceleration.
 * It supports:
 * - Lennard-Jones non-bonded interactions
 * - Coulomb electrostatic forces
 * - Bonded interactions (bonds, angles, dihedrals)
 * - MPI domain decomposition
 * - Efficient neighbor list handling
 * 
 * The implementation uses:
 * - Coalesced memory access patterns
 * - Shared memory for reductions
 * - Atomic operations for force accumulation
 * - Texture memory for force field parameters (optional)
 */
class ForceCalculatorCUDA : public ForceCalculator,
                            public dsdecompx::ParticleInfo::Provider {
public:
#ifdef USE_CUDA
    ForceCalculatorCUDA();
    virtual ~ForceCalculatorCUDA();

    // Override base class methods
    void initialize() override;
    void finalize() override;
    void compute_forces(MDSystem& system) override;
    void calculate_kinetic_energy(MDSystem& system) override;

    // GPU-resident data management (public for external access)
    void initial_upload_to_device(MDSystem& system);     // Initial 1-time upload
    void download_from_device(MDSystem& system);         // Download for output/checkpoint
    void update_neighbor_list_on_device(MDSystem& system); // Only update neighbor list
    void build_neighbor_list_on_device(MDSystem& system); // Build neighbor list directly on GPU
    // Rebuild global-ID -> current owned/ghost-slot bonded lists after a
    // dsDecompX halo or migration changes the device particle layout.
    void refresh_bonded_topology_on_device(MDSystem& system);
    
    // GPU-resident integrator support
    void integrate_on_device(MDSystem& system, Real dt, const std::string& integrator_type);
    void complete_velocity_update_on_device(MDSystem& system, Real dt, const std::string& integrator_type);

    // ── SHAKE/RATTLE constraints (GPU-resident) ────────────────────────────
    // Call snapshot_constraint_positions() BEFORE integrate_on_device() (it
    // captures r(t) for SHAKE's fixed reference direction), apply_shake_on_device()
    // right after it, and apply_rattle_on_device() right after
    // complete_velocity_update_on_device(). velocity_verlet only; see
    // include/constraint_solver.h for the CPU-side derivation this mirrors.
    void snapshot_constraint_positions(MDSystem& system);
    bool apply_shake_on_device(MDSystem& system, Real dt,
                                int max_iterations = 500, Real position_tolerance = 1e-8);
    bool apply_rattle_on_device(MDSystem& system,
                                 int max_iterations = 500, Real velocity_tolerance = 1e-8);

    // GPU-resident neighbor list management
    Real compute_max_displacement_on_device(size_t num_local_particles, const SimulationBox& box);
    void append_ghosts_to_device(MDSystem& system, size_t previous_local_count);
    void update_reference_positions_on_device(size_t num_local_particles);  // NEW: Update reference positions after neighbor list rebuild

    // ── FIRE energy minimizer (GPU-resident) ────────────────────────────────
    // See include/minimizer.h for the algorithm this drives. Both methods
    // operate on LOCAL particles only (ghosts are excluded from the sums, and
    // never get their position/velocity mutated here) and never touch the
    // host mirror -- x/v stay device-resident for the whole minimization.
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

    // Device pointer accessors for GPU-resident thermostat/barostat/integrator
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

    // Non-owning views used by the DeepScaleMD -> dsDecompX CUDA adapter.
    // No host copy is performed when constructing or consuming these views.
    dsdecompx::ParticleView particle_info_view(size_t count) override;
    // Grows the SoA allocation while preserving the requested owned prefix.
    // Returns true when pointers changed and dependent GPU layouts must rebuild.
    bool ensure_device_particle_capacity(size_t required, size_t preserve_count);
    dsdecompx::StorageResizeResult resize_particle_info(
        size_t required, size_t preserve_count) override;
    
    // NEW: Force accessors for complete GPU-resident pipeline
    Real* get_device_fx() { return d_fx; }
    Real* get_device_fy() { return d_fy; }
    Real* get_device_fz() { return d_fz; }
    
    // Upload forces from host to device (for Python/ML force integration)
    void upload_forces_to_device(const ParticleData& particles, size_t num_particles);

    // ── MLP GPU-efficient path ─────────────────────────────────────────────
    // Download ONLY positions (D→H) before MLP call so Python sees fresh data.
    // Uses the existing force_stream and cudaStreamSynchronize so it is safe
    // to call while in the GPU-resident loop.
    void download_positions_for_ml(ParticleData& particles, size_t num_particles);

    // Download the accumulated device forces before a host/Python component
    // adds its contribution.  The matching upload replaces the device buffer.
    void download_forces_for_ml(ParticleData& particles, size_t num_particles);

    // Upload ONLY forces (H→D) after MLP call, async on force_stream.
    void upload_forces_from_ml(const ParticleData& particles, size_t num_particles);

    // Register host particle vectors as pinned (page-locked) memory so that
    // DMA transfers run without staging through a temporary pinned buffer.
    // Call once after all host vectors have been sized (after initial_upload).
    void register_ml_pinned_buffers(ParticleData& particles, size_t num_particles);
    void unregister_ml_pinned_buffers(ParticleData& particles);

    // ── C++ LibTorch MLIP path ────────────────────────────────────────────
    // Register MD device pointers so MlipRuntime YAML can address them.
    void pre_register_mlip_sources();
    // Attach MlipRuntime pointer (non-owning).
    void set_mlip_runtime(void* raw_runtime);
    // Controls the built-in bonded/non-bonded component independently from an
    // attached LibTorch MLIP component.
    void set_builtin_forces_enabled(bool enabled) { builtin_forces_enabled_ = enabled; }
    // Run the C++ MLIP runtime for one MD step.
    void compute_mlip_runtime_forces(MDSystem& system);

    // PERF FIX: Track whether the neighbor list was rebuilt since the last
    // COO edge-list construction (only relevant for the LibTorch MLIP path).
    bool nlist_dirty_ = true;

    // Bridge function (mlip_force_bridge.cpp) needs access to device buffers.
    friend void mlip_bridge_register_md_sources(void* fc_cuda_ptr);

protected:
    // Internal force computation methods
    void compute_lj_forces(MDSystem& system);
    // Full Ewald summation: real-space (erfc-screened, neighbor list) +
    // reciprocal-space (B-spline PME grid via cuFFT) + self-energy correction.
    void compute_electrostatic_forces_ewald(MDSystem& system);
    void ensure_ewald_initialized(MDSystem& system);
    void free_ewald_resources();
    void compute_bond_forces(MDSystem& system);
    void compute_angle_forces(MDSystem& system);
    void compute_dihedral_forces(MDSystem& system);
    void compute_improper_forces(MDSystem& system);
    // Scaled 1-4 (or FF-specific) nonbonded pairs; see Topology::SpecialPair.
    void compute_special_pair_forces(MDSystem& system);

    // MLIP COO edge-list and position helpers (mlip_runtime path)
    void build_coo_edge_list_on_device(MDSystem& system);
    void interleave_positions_on_device(int num_total);

    // Device memory management
    void allocate_device_memory(size_t num_particles, size_t max_neighbors);
    
    // Bonded topology management (GPU-resident for efficiency)
    void upload_bonded_topology_to_device(MDSystem& system);
    void free_bonded_topology_device();

    // Device memory pointers for particle data
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
    
    // Device memory for displacement tracking (Verlet list rebuild check)
    Real* d_x_ref;          // Reference positions for displacement check
    Real* d_y_ref;
    Real* d_z_ref;
    Real* d_max_disp_sq;    // Device scalar for max displacement squared
    Real* d_fire_reduce_;   // [power, f_norm_sq, v_norm_sq, f_max_sq], lazily allocated

    // Device memory for neighbor list
    int* d_neighbors;       // Neighbor list
    int* d_num_neighbors;   // Number of neighbors per particle
    int* d_neighbor_offset; // Offset into neighbor list for each particle

    // Device memory for force field parameters
    Real* d_epsilon_mixed; // LJ epsilon mixing rules (1D row-major array)
    Real* d_sigma_mixed;   // LJ sigma mixing rules (1D row-major array)
    unsigned long long* d_exclusion_keys; // sorted global atom-pair keys
    size_t num_exclusions;

    // Device memory for energies
    Real* d_potential_energy;
    Real* d_kinetic_energy;
    Real* d_virial;              // NEW: Device memory for virial calculation

    // Pinned-memory registration flag for MLP path
    bool ml_buffers_pinned = false;

    // ===== GPU-RESIDENT BONDED TOPOLOGY =====
    // Bond topology (allocated once in initialize, freed in finalize)
    size_t num_bonds_cached;
    size_t* d_bond_atom1;
    size_t* d_bond_atom2;
    Real*   d_bond_k;
    Real*   d_bond_r0;
    
    // Angle topology (allocated once)
    size_t num_angles_cached;
    size_t* d_angle_atom1;
    size_t* d_angle_atom2;
    size_t* d_angle_atom3;
    Real*   d_angle_k;
    Real*   d_angle_theta0;
    
    // Dihedral topology (allocated once)
    size_t num_dihedrals_cached;
    size_t* d_dihedral_atom1;
    size_t* d_dihedral_atom2;
    size_t* d_dihedral_atom3;
    size_t* d_dihedral_atom4;
    Real*   d_dihedral_k;
    Real*   d_dihedral_phi0;
    int*    d_dihedral_n;

    // Improper torsion topology (same shape/kernel as dihedrals above)
    size_t num_impropers_cached;
    size_t* d_improper_atom1;
    size_t* d_improper_atom2;
    size_t* d_improper_atom3;
    size_t* d_improper_atom4;
    Real*   d_improper_k;
    Real*   d_improper_phi0;
    int*    d_improper_n;

    // Scaled 1-4 (special) nonbonded pairs
    size_t num_special_pairs_cached;
    size_t* d_special_pair_atom1;
    size_t* d_special_pair_atom2;
    Real*   d_special_pair_lj_scale;
    Real*   d_special_pair_coulomb_scale;
    Real*   d_special_pair_sigma;
    Real*   d_special_pair_epsilon;

    // SHAKE/RATTLE distance constraints
    size_t num_constraints_cached;
    size_t* d_constraint_atom1;
    size_t* d_constraint_atom2;
    Real*   d_constraint_distance;

    // Per-step SHAKE reference positions r(t): self-managed capacity (grown
    // lazily to the current local+ghost particle count), separate from the
    // shared particle-array growth path since this scratch space is only
    // needed when constraints are active.
    size_t constraint_scratch_capacity_;
    Real* d_constraint_pre_x_;
    Real* d_constraint_pre_y_;
    Real* d_constraint_pre_z_;
    Real* d_constraint_max_error_; // single device scalar, reused every iteration
    void ensure_constraint_scratch_capacity(size_t n);
    // ========================================

    // Memory allocation tracking
    size_t allocated_particles;
    size_t allocated_neighbors;
    int max_types;
    bool device_data_initialized;  // Track if initial upload completed
    bool bonded_topology_on_device; // Track if bonded topology is uploaded
    std::vector<int> bonded_layout_global_ids_;
    size_t bonded_layout_local_count_ = static_cast<size_t>(-1);
    
    // CUDA configuration
    int block_size;         // CUDA block size for kernels
    int device_id;          // GPU device ID
    
    // Stream management (for async operations)
    cudaStream_t force_stream;
    cudaStream_t energy_stream;
    
    // Performance monitoring
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float kernel_time_ms;
    
    // MPI variables
    int rank;
    int mpi_size;
    size_t allocated_size;
    
    // CUDA stream and cuBLAS
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    
    // Debug counters
    int* d_debug_counters;

    // ── C++ MlipRuntime device buffers ────────────────────────────────────
    void*    mlip_runtime_raw_      = nullptr;  // non-owning ptr to MlipRuntime
    bool     mlip_runtime_enabled_  = false;
    bool     builtin_forces_enabled_ = true;
    int      num_local_cached_      = 0;
    int      num_ghost_cached_      = 0;

    // COO half-list edge buffers (half → full directed)
    int32_t* d_edge_src_         = nullptr;
    int32_t* d_edge_dst_         = nullptr;
    int32_t* d_edge_coo_packed_  = nullptr;  // [src(E_half) | dst(E_half)]
    int32_t* d_tmp_edge_src_     = nullptr;
    int32_t* d_tmp_edge_dst_     = nullptr;
    int64_t  num_edges_          = 0;
    int64_t  cap_edges_          = 0;

    // Interleaved position buffer [N,3] float64
    double* d_pos_interleaved_   = nullptr;
    int     cap_pos_interleaved_ = 0;

    // MLIP force output buffer [N,3] float64 AoS
    double* d_mlip_forces_       = nullptr;
    int     cap_mlip_forces_     = 0;

    // Edge vector buffer [E,3] float32
    float*   d_edge_vectors_     = nullptr;
    int64_t  cap_edge_vectors_   = 0;

    // Batch / num_atoms tensors for MLIP
    int64_t* d_batch_            = nullptr;
    int      cap_batch_          = 0;
    int64_t* d_num_atoms_        = nullptr;

    // Box cell and PBC tensors on device
    double*  d_cell_             = nullptr;
    int*     d_pbc_buf_          = nullptr;

    // MLIP error flag
    int*     d_mlip_error_flag_  = nullptr;
    // ─────────────────────────────────────────────────────────────────────
    
    size_t max_neighbors_allocated;
    
    // Compact array tracking for MPI
    size_t current_compact_size;
    std::vector<int> current_compact_map;
    std::vector<bool> current_is_local;
    
    // ===== EWALD / PME RECIPROCAL-SPACE ELECTROSTATICS =====
    // Grid + cuFFT state for the long-range part of the Ewald sum. Parameters
    // (alpha, grid size) are derived automatically from the physical cutoff
    // and box size on first use (see ensure_ewald_initialized).
    bool ewald_initialized_ = false;
    int ewald_grid_x_ = 0, ewald_grid_y_ = 0, ewald_grid_z_ = 0;
    Real ewald_alpha_ = 0.0;
    int ewald_spline_order_ = 4;
    Real ewald_prefactor_ = 0.0;  // 4*pi*k_coulomb / box_volume (updated on box change)

    // Charge density grid, reused in place to hold Phi(k) after the Poisson
    // solve. Full complex-to-complex (C2C) layout -- see plan notes on why
    // R2C/C2R half-spectrum packing was dropped.
    cufftDoubleComplex* d_pme_charge_grid_ = nullptr;
    cufftDoubleComplex* d_pme_ex_grid_     = nullptr;
    cufftDoubleComplex* d_pme_ey_grid_     = nullptr;
    cufftDoubleComplex* d_pme_ez_grid_     = nullptr;
    Real* d_pme_force_grid_x_ = nullptr;   // real part of ifft(Ex(k)) etc.
    Real* d_pme_force_grid_y_ = nullptr;
    Real* d_pme_force_grid_z_ = nullptr;
    Real* d_pme_b_spline_mod_x_ = nullptr;
    Real* d_pme_b_spline_mod_y_ = nullptr;
    Real* d_pme_b_spline_mod_z_ = nullptr;
    cufftHandle pme_fft_plan_ = 0;   // single Z2Z plan, direction chosen at exec time
    cudaStream_t pme_stream_ = nullptr;
#else
    // Stub implementation when CUDA is not enabled
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

// CUDA kernel function declarations (implemented in .cu file)
#ifdef USE_CUDA

/**
 * @brief Clear force arrays on device
 */
__global__ void clear_forces_kernel(Real* fx, Real* fy, Real* fz, size_t n);

/**
 * @brief Compute Lennard-Jones forces
 */
__global__ void compute_lj_forces_kernel(
    Real* x, Real* y, Real* z, 
    Real* fx, Real* fy, Real* fz,
    int* type, int* global_id,
    int* neighbors, int* num_neighbors, int* neighbor_offset,
    Real* epsilon_mixed, Real* sigma_mixed,
    Real* potential_energy,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    int max_types,
    Real cutoff_sq);

/**
 * @brief Compute real-space (erfc-screened) Ewald electrostatic forces.
 *
 * This is the near-field half of the Ewald sum: U = k_coulomb * qi*qj *
 * erfc(alpha*r) / r, evaluated over the existing neighbor list. It must
 * always be paired with the reciprocal-space PME grid contribution
 * (see compute_electrostatic_forces_ewald) -- neither half is physically
 * meaningful alone.
 */
__global__ void compute_electrostatic_forces_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    Real* charge, int* global_id,
    int* neighbors, int* num_neighbors, int* neighbor_offset,
    Real* potential_energy,
    Real* virial,  // NEW: virial for pressure calculation
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    size_t local_particles, size_t total_particles,
    Real cutoff_sq, Real k_coulomb, Real alpha);

// ============================================================================
// PME (Particle Mesh Ewald) reciprocal-space kernels.
// Full complex-to-complex (C2C) grid layout throughout -- see
// force_calculator_cuda.cu for the driving routine
// (ForceCalculatorCUDA::compute_electrostatic_forces_ewald).
// ============================================================================

__device__ void pme_compute_bspline_weights(Real u, int order, Real* weights, int* grid_indices);

__global__ void pme_clear_grid_kernel(cufftDoubleComplex* grid, size_t grid_size);

__global__ void pme_spread_charges_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge, cufftDoubleComplex* charge_grid,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles);

__global__ void pme_solve_poisson_kernel(
    cufftDoubleComplex* fft_grid,
    const Real* __restrict__ b_mod_x, const Real* __restrict__ b_mod_y, const Real* __restrict__ b_mod_z,
    Real* reciprocal_energy_out, Real prefactor, Real alpha,
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z);

__global__ void pme_compute_electric_field_kernel(
    const cufftDoubleComplex* __restrict__ phi_grid,
    cufftDoubleComplex* ex_grid, cufftDoubleComplex* ey_grid, cufftDoubleComplex* ez_grid,
    int grid_nx, int grid_ny, int grid_nz,
    Real box_x, Real box_y, Real box_z);

__global__ void pme_extract_real_part_kernel(const cufftDoubleComplex* complex_grid,
                                              Real* real_grid, size_t grid_size);

__global__ void pme_interpolate_forces_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const Real* __restrict__ charge,
    const Real* __restrict__ field_grid_x, const Real* __restrict__ field_grid_y,
    const Real* __restrict__ field_grid_z,
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
    int grid_nx, int grid_ny, int grid_nz,
    Real inv_box_x, Real inv_box_y, Real inv_box_z);

/**
 * @brief Compute bond forces
 */
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

/**
 * @brief Compute angle forces
 */
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

/**
 * @brief Compute dihedral torsion forces
 */
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

/**
 * @brief Calculate kinetic energy with reduction
 */
__global__ void calculate_kinetic_energy_kernel(
    Real* vx, Real* vy, Real* vz,
    Real* mass,
    Real* kinetic_energy,
    size_t n);

/**
 * @brief Device helper function for minimum image convention
 */
__device__ inline void apply_minimum_image_device(
    Real& dx, Real& dy, Real& dz,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z);

#endif // USE_CUDA

#endif // FORCE_CALCULATOR_CUDA_H
