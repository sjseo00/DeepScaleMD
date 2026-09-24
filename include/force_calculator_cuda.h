#ifndef FORCE_CALCULATOR_CUDA_H
#define FORCE_CALCULATOR_CUDA_H

#include "../include/force_calculator.h"
#include "../include/md_types.h"
#include <dsdecompx.hpp>
#include <vector>
#include <string>
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
    void build_neighbor_list_on_device(MDSystem& system); // Build rank-local spatial tile graph on GPU
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
    // Mixed precision: the LJ and Ewald real-space pair kernels do their
    // per-pair math (distance, r^-2, erf/exp, force magnitude) in float
    // instead of double. Positions are still differenced in double and
    // forces/energy/virial still accumulate in double, so this trades a
    // little per-pair accuracy for throughput
    void set_mixed_precision(bool enabled) { mixed_precision_enabled_ = enabled; }
    bool is_mixed_precision() const { return mixed_precision_enabled_; }
    // Hopper (sm_90+) thread-block-cluster PME charge-spread path (see
    // pme_spread_charges_cluster_kernel in the .cu file). UNVERIFIED: no
    // Hopper-class GPU was available to run or benchmark it. Only takes
    // effect when the runtime device is actually compute capability >= 9.0
    // (device_supports_clusters_, detected in initialize()); on any older
    // GPU this flag is silently ignored and the existing warp-aggregated
    // kernel runs instead.
    void set_pme_cluster_spread_enabled(bool enabled) { pme_use_cluster_spread_ = enabled; }
    bool is_pme_cluster_spread_enabled() const { return pme_use_cluster_spread_; }
    // Run the C++ MLIP runtime for one MD step.
    void compute_mlip_runtime_forces(MDSystem& system);

    // Diagnostic per-force-component timing (MD_TIMING_TRACE_FORCE=1, wired
    // up in MDSimulation). Uses cudaEvent markers on force_stream rather than
    // extra cudaStreamSynchronize calls, so turning it on does not perturb
    // the deliberate LJ/PME stream overlap it's trying to measure -- the
    // elapsed times are all read back at the single synchronization point
    // compute_forces() already does at the end for the energy/virial
    // download. Off by default (near-zero cost when disabled: skipped event
    // creation, no extra kernel launches).
    void set_timing_breakdown_enabled(bool enabled) override { timing_breakdown_enabled_ = enabled; }
    bool is_timing_breakdown_enabled() const override { return timing_breakdown_enabled_; }
    // Only the components that actually ran this step are present (e.g. no
    // "mlip" entry when no LibTorch runtime is attached). Empty when
    // set_timing_breakdown_enabled(false) or before the first compute_forces().
    const std::vector<ForceComponentTiming>& get_last_force_component_timings() const override {
        return last_force_component_timings_;
    }

    // PERF FIX: Track whether the neighbor list was rebuilt since the last
    // COO edge-list construction (only relevant for the LibTorch MLIP path).
    bool nlist_dirty_ = true;

    // Bridge function (mlip_force_bridge.cpp) needs access to device buffers.
    friend void mlip_bridge_register_md_sources(void* fc_cuda_ptr);

protected:
    // Internal force computation methods
    // Real-space (erfc-screened) half of the Ewald sum, evaluated over the
    // existing neighbor list. Split out of compute_electrostatic_forces_ewald()
    // so compute_nonbonded_forces_fused() can skip it (it does the equivalent
    // work itself, fused with LJ) while still sharing compute_electrostatic_
    // forces_reciprocal() below.
    void compute_electrostatic_forces_real_space(MDSystem& system);
    // Reciprocal-space (B-spline PME grid via cuFFT) part + self-energy
    // correction. Split out of compute_electrostatic_forces_ewald() so the
    // fused LJ+Coulomb real-space path (below) can call it standalone.
    void compute_electrostatic_forces_reciprocal(MDSystem& system);
    // Full Ewald summation: real-space + reciprocal-space + self-energy
    // correction. Used when Coulomb is enabled without LJ (see compute_forces()).
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
    // Build/reuse the compact directed atom graph entirely on the GPU.  The
    // spatial tile list remains the broad-phase accelerator; this method
    // compacts its Verlet masks into atom CSR/COO at rc + skin/2.  The CSR is
    // consumed by the classical graph kernel and the same COO storage is
    // handed directly to LibTorch, avoiding the old D2H -> CPU prune -> H2D
    // path.
    void ensure_graph_on_device(MDSystem& system);
    void rebuild_graph_from_tiles_on_device(MDSystem& system);
    void ensure_graph_edge_capacity(int64_t edges);

    // Device memory management
    void allocate_device_memory(size_t num_particles);

    // Per-(re)build/per-step diagnostics for the GPU tile-pair graph:
    // neighbor-list inflation, interaction density, load imbalance (purely
    // structural, computed from num_lj_tiles_/num_lj_tile_pairs_/
    // d_tile_pair_offsets_, no kernel changes needed) and, when
    // MD_LOG_TILE_PAIR_STATS=1, tile-pruning efficiency and half-list atomic
    // cost (needs the opt-in d_tile_pair_stats_ counters populated by the
    // tiled LJ/nonbonded kernels). See force_calculator_cuda.cu.
    void log_tile_graph_stats(size_t local_particles);
    // Probes MD_LOG_TILE_PAIR_STATS and allocates d_tile_pair_stats_ on first
    // call; must run BEFORE the LJ/fused-nonbonded kernel launch that reads
    // d_tile_pair_stats_, since a null pointer at launch time means that
    // launch's counts are lost.
    void ensure_tile_pair_stats_enabled();
    void log_tile_pair_stats_gpu();
    
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

    // Owned by the persistent buffers inside launch_cuda_tile_list_build
    // (neighbor_list_kernels.cu); these are just the current pointers/counts
    // handed back after each build.
    int* d_tile_atoms = nullptr;    // [num_lj_tiles_ * LJ_TILE_SIZE], atom index per slot (-1 = padding)
    int* d_tile_pair_i = nullptr;   // [num_lj_tile_pairs_], tile id in the "i" role (sorted by this key)
    int* d_tile_pair_j = nullptr;   // [num_lj_tile_pairs_], tile id in the "j" role, grouped by d_tile_pair_offsets
    int* d_tile_pair_offsets = nullptr; // [num_lj_tiles_ + 1] CSR offsets into d_tile_pair_j, keyed by home tile
    unsigned long long* d_tile_pair_verlet_mask_ = nullptr; // [num_lj_tile_pairs_], bit=(j_slot*8+i_lane)
    int  num_lj_tiles_ = 0;
    int  num_lj_tile_pairs_ = 0;
    unsigned long long pairs_inside_rc_ = 0;
    unsigned long long pairs_inside_rc_plus_skin_ = 0;

    // Opt-in (MD_LOG_TILE_PAIR_STATS=1) diagnostic counters for the tiled LJ
    // kernels: d_tile_pair_stats_[0] = candidate atom pairs tested (accepted
    // slot, before the cutoff/exclusion check), [1] = real atom pairs (passed
    // it). Left null when the env var isn't set, so the kernels' extra
    // `if (pair_stats)` branch is the only cost paid by default. See
    // log_tile_pair_stats_gpu() in force_calculator_cuda.cu.
    unsigned long long* d_tile_pair_stats_ = nullptr;
    bool tile_pair_stats_probed_ = false;
    bool tile_pair_stats_enabled_ = false;


    long long* d_force_fixed_x_ = nullptr;
    long long* d_force_fixed_y_ = nullptr;
    long long* d_force_fixed_z_ = nullptr;

    // Device memory for force field parameters
    Real* d_epsilon_mixed; // LJ epsilon mixing rules (1D row-major array)
    Real* d_sigma_mixed;   // LJ sigma mixing rules (1D row-major array)
    unsigned long long* d_exclusion_keys; // sorted global atom-pair keys
    size_t num_exclusions;
    int* d_exclusion_offsets;  // global-ID indexed CSR offsets
    int* d_exclusion_partners; // short per-atom exclusion lists
    size_t exclusion_atom_count;

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
    bool     mixed_precision_enabled_ = false;

    int      num_local_cached_      = 0;
    int      num_ghost_cached_      = 0;

    // COO half-list edge buffers (half → full directed)
    int32_t* d_edge_src_         = nullptr;
    int32_t* d_edge_dst_         = nullptr;
    // Packed edge_index storage. Tiled conversion stores the source/dest
    // half-list here; graph mode stores [src(E_active) | dst(E_active)].
    int32_t* d_edge_coo_packed_  = nullptr;
    int32_t* d_tmp_edge_src_     = nullptr;
    int32_t* d_tmp_edge_dst_     = nullptr;
    int64_t  num_edges_          = 0;
    int64_t  cap_edges_          = 0;

    // Directed graph CSR.  d_edge_dst_ is also the CSR column array because
    // graph fill writes COO in row-major order.  Separate degree/cursor
    // scratch is retained and reused so active-graph refreshes allocate no
    // memory in the steady state.
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

    // Charge density: real-valued spread grid (nx*ny*nz) transformed by an
    // out-of-place D2Z (real-to-complex) plan into the half-spectrum
    // frequency-domain buffer (nx*ny*(nz/2+1)), which is then overwritten in
    // place to hold Phi(k) after the Poisson solve. R2C/C2R halves both the
    // FFT work and the grid memory versus a full complex-to-complex
    // transform by exploiting the Hermitian symmetry of a real charge
    // density -- see compute_electrostatic_forces_ewald for the energy/
    // virial double-counting this packing requires.
    // The real grid is reused as the inverse-transformed potential grid after
    // the forward transform; this keeps the reciprocal path to one real grid.
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

    // Runtime capability/opt-in for the Hopper cluster PME spread kernel
    // (see set_pme_cluster_spread_enabled above). device_supports_clusters_
    // is detected once from the actual GPU in initialize(); a fat binary
    // built with sm_90 in CMAKE_CUDA_ARCHITECTURES still runs fine on older
    // GPUs, it just never takes this path.
    bool device_supports_clusters_ = false;
    bool pme_use_cluster_spread_ = false;

    // Per-force-component timing (see set_timing_breakdown_enabled above).
    // force_clear_event_ can't be reused as the start marker: it's created
    // with cudaEventDisableTiming (cheaper for its actual job, gating
    // pme_stream_), which makes it unusable with cudaEventElapsedTime.
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

// ============================================================================
// PME (Particle Mesh Ewald) reciprocal-space kernels.
// Real-to-complex (D2Z/Z2D) half-spectrum grid layout -- see
// force_calculator_cuda.cu for the driving routine
// (ForceCalculatorCUDA::compute_electrostatic_forces_ewald) and the packing
// notes above the kernel definitions.
// ============================================================================

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
 * (templated so callers can wrap in float on the mixed-precision hot path --
 * see the definition in force_calculator_cuda.cu)
 */
template <typename T>
__device__ inline void apply_minimum_image_device(
    T& dx, T& dy, T& dz,
    T box_x, T box_y, T box_z,
    T inv_box_x, T inv_box_y, T inv_box_z);

#endif // USE_CUDA

#endif // FORCE_CALCULATOR_CUDA_H
