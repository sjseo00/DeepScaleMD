#ifndef MD_SYSTEM_H
#define MD_SYSTEM_H

#include "dsdecompx.hpp"
#include "force_manager.h"
#include "md_types.h"
#include "topology.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mpi.h>

class NeighborList;

class MDSystem {
public:
    // The type name is a pointer into the topology's term list (it outlives these lists),
    // and the force-field parameters are copied in when the list is built, so neither a
    // string copy nor a map lookup per term is needed on a rebuild.
    struct LocalBond {
        size_t atom1, atom2;
        const std::string* type;
        Real k, r0;
    };
    struct LocalAngle {
        size_t atom1, atom2, atom3;
        const std::string* type;
        Real k, theta0;
    };
    struct LocalDihedral {
        size_t atom1, atom2, atom3, atom4;
        const std::string* type;
        Real k, phi0;
        int n;
    };
    struct LocalSpecialPair {
        size_t atom1, atom2;
        Real lj_scale, coulomb_scale;
        Real sigma, epsilon;
    };
    struct LocalConstraint {
        size_t atom1, atom2;
        Real distance;
    };

private:
    ParticleData particles;
    SimulationBox box;
    ForceFieldParams ff_params;
    NeighborList* neighbor_list_ptr_ = nullptr;

    std::unique_ptr<Topology> topology;
    std::unique_ptr<ForceManager> force_manager;

    Real physical_cutoff;
    Real verlet_skin_ = 2.0;
    Real mlip_ghost_depth_ = 0.0;   // set when an MLIP that needs real ghost atoms (libtorch) is selected
    bool mlip_replicated_ = false;  // a subdomain is narrower than the MLIP's ghost layer: every rank computes all atoms
    bool builtin_forces_enabled_ = true;   // force_calculation.mode has "builtin" (the classical short-range kernels run)
    bool mlip_replicated_reported_ = false;

    // decomposition: (advanced, optional YAML, nested as advanced.decomposition.*)
    // overrides for the dsDecompX process-grid policy; empty/sentinel means "unset",
    // leaving the library default (MPI_Dims_create) in charge. See
    // set_decomposition_policy() and initialize_decomposition_runtime().
    std::string decomposition_grid_policy_;
    double decomposition_inter_node_weight_ = -1.0;
    double decomposition_intra_node_weight_ = -1.0;
    long decomposition_dynamic_interval_ = -1;
    double decomposition_dynamic_imbalance_threshold_ = -1.0;
    double decomposition_dynamic_relaxation_ = -1.0;

    int rank;
    int size;
    // The communicator this MDSystem's collectives run on. Defaults to
    // MPI_COMM_WORLD (legacy, no PP/PME split); set_pp_communicator()
    // repoints it (and rank/size) at the PP sub-communicator once
    // PmePpSplit has run (see MDSimulation::initialize_impl). MDSystem is
    // only ever constructed/used on PP ranks -- PME-group ranks never reach
    // any code that touches this.
    MPI_Comm pp_comm_ = MPI_COMM_WORLD;
    std::vector<std::string> coordinate_elements_;
    // By MD type id: the element symbol the coordinate file gives the atoms of that type ("" if they disagree). Same on every PP rank.
    std::vector<std::string> type_coordinate_elements_;

    std::unique_ptr<dsdecompx::Decomp> decomp_runtime_;
    bool last_device_halo_plan_reused_ = false;

    std::vector<LocalBond> local_bonds_;
    std::vector<LocalAngle> local_angles_;
    std::vector<LocalDihedral> local_dihedrals_;
    std::vector<LocalDihedral> local_impropers_;
    std::vector<LocalSpecialPair> local_special_pairs_;
    std::vector<LocalConstraint> local_constraints_;
    std::vector<int> bonded_layout_global_ids_;
    size_t bonded_layout_local_count_ = static_cast<size_t>(-1);
    // Parameters of every topology term, looked up once (by term index) instead of by
    // type name on every neighbor rebuild; ok flags mark terms whose type is unknown.
    bool bonded_param_cache_ready_ = false;
    std::vector<Real> bond_k_all_, bond_r0_all_, angle_k_all_, angle_theta0_all_;
    std::vector<Real> dihedral_k_all_, dihedral_phi0_all_, improper_k_all_, improper_phi0_all_;
    std::vector<int> dihedral_n_all_, improper_n_all_;
    std::vector<char> bond_ok_all_, angle_ok_all_, dihedral_ok_all_, improper_ok_all_;
    std::vector<int> bonded_slot_scratch_;   // global id -> local slot, reused between rebuilds
    void ensure_bonded_parameter_cache();
    friend class ForceCalculatorCUDA;  // builds the local bonded lists on the GPU from the caches above
    // true (CUDA): the local lists hold every term that touches an owned atom, because the
    // kernels apply forces to owned atoms only. false: only terms owned by the lower-id
    // atom, whose forces on ghosts are sent back by a reverse halo exchange (CPU path).
    bool bonded_redundant_ = false;
    void resize_particle_storage(size_t count);
    void initialize_decomposition_runtime();
    // True when every decomposed axis has a subdomain at least as wide as the halo (this rank only; see sync_decomposition_domain).
    bool mlip_halo_fits(Real halo_width) const;

public:
    MDSystem();
    virtual ~MDSystem();

    virtual void initialize();
    virtual void finalize();

    void initialize_mpi();
    void load_config(const std::string& config_file);
    void load_coordinates(const std::string& coord_file);
    void load_topology(const std::string& topo_file,
                       const std::string& parameter_file = "");
    void initialize_velocities(Real temperature);
    void initialize_distributed_velocities(Real temperature);
    void setup_domain_decomposition();

    void load_standard_xyz(const std::string& coord_file);
    void assign_masses_from_topology();
    void assign_charges_from_topology();
    void setup_force_field_params();
    void initialize_force_manager(const std::map<std::string, std::string>& config_map);

    void compute_forces();

    ParticleData& get_particles() { return particles; }
    SimulationBox& get_simulation_box() { return box; }
    SimulationBox& get_box() { return box; }
    ForceFieldParams& get_ff_params() { return ff_params; }
    NeighborList& get_neighbor_list() { return *neighbor_list_ptr_; }
    void set_neighbor_list(NeighborList* nlist) { neighbor_list_ptr_ = nlist; }
    Topology& get_topology() { return *topology; }
    bool has_topology() const { return static_cast<bool>(topology); }
    const std::vector<std::string>& get_type_coordinate_elements() const { return type_coordinate_elements_; }
    const std::vector<LocalBond>& get_local_bonds() const { return local_bonds_; }
    const std::vector<LocalAngle>& get_local_angles() const { return local_angles_; }
    const std::vector<LocalDihedral>& get_local_dihedrals() const { return local_dihedrals_; }
    const std::vector<LocalDihedral>& get_local_impropers() const { return local_impropers_; }
    const std::vector<LocalSpecialPair>& get_local_special_pairs() const { return local_special_pairs_; }
    const std::vector<LocalConstraint>& get_local_constraints() const { return local_constraints_; }
    ForceManager& get_force_manager() { return *force_manager; }

    size_t get_num_particles() const { return particles.size(); }
    size_t get_local_particles() const { return particles.owned_count(); }
    size_t get_ghost_count() const { return particles.ghost_count(); }
    size_t get_total_particles() const { return particles.size(); }

    // Repoints this system's collectives at pp_comm (and refreshes rank/size
    // to be pp_comm-relative). Must be called before load_coordinates() /
    // load_topology() / setup_domain_decomposition(), i.e. right after
    // PmePpSplit::instance().initialize() runs, on PP ranks only.
    void set_pp_communicator(MPI_Comm pp_comm) {
        pp_comm_ = pp_comm;
        MPI_Comm_rank(pp_comm_, &rank);
        MPI_Comm_size(pp_comm_, &size);
    }

    int get_rank() const { return rank; }
    int get_size() const { return size; }
    Real get_physical_cutoff() const { return physical_cutoff; }
    void set_physical_cutoff(Real cutoff) { physical_cutoff = cutoff; }
    void set_verlet_skin(Real skin) { verlet_skin_ = skin; }
    // False for an MLIP-only run (mode: python): no classical short-range kernel runs, so the warning about ghost images that
    // are received twice (warn_if_domain_narrower_than_two_halos) does not apply: the MLIP edge search uses the real positions
    // of the atoms and images, no minimum-image convention, and the two copies of an atom are different periodic images.
    void set_builtin_forces_enabled(bool enabled) { builtin_forces_enabled_ = enabled; }

    // An MLIP with LibTorch needs ghost atoms as deep as its receptive field (num_layers * cutoff, design 5.1), not just
    // the force cutoff. Re-applies the halo width right away. 0 (the default) leaves the halo to the force cutoff, so
    // runs without such an MLIP are unchanged.
    void set_mlip_ghost_depth(Real depth);
    // Halo width in A: the larger of the force cutoff and the MLIP depth, plus the Verlet skin and a margin. When the MLIP's
    // ghost layer does not fit into the subdomains (mlip_replicated()) the MLIP takes no ghosts from the halo, so the depth
    // does not count.
    Real get_halo_width() const {
        const Real depth = mlip_replicated_ ? 0.0 : mlip_ghost_depth_;
        return (physical_cutoff > depth ? physical_cutoff : depth) + verlet_skin_ + 1.0;
    }
    // True when a halo exchange cannot give the MLIP its ghost layer (a subdomain along a decomposed axis is narrower than
    // ghost depth + skin + margin: one exchange reaches only the neighboring ranks). The same on all ranks; re-judged whenever the
    // box changes. The libtorch MLIP then runs replicated: every rank computes all atoms (MLIPReplica), correct but no speed-up.
    bool mlip_replicated() const { return mlip_replicated_; }
    // True when the box is split into more than one subdomain along the axis (0, 1, 2); the halo then provides ghosts there.
    bool axis_is_decomposed(int axis) const;

    // decomposition.* (advanced, optional YAML section nested under advanced:); must be
    // called before setup_domain_decomposition() to take effect. An empty policy string
    // means "unset", leaving the library default (MPI_Dims_create) in charge.
    void set_decomposition_policy(const std::string& policy, double inter_node_weight,
                                  double intra_node_weight, long dynamic_interval,
                                  double dynamic_imbalance_threshold, double dynamic_relaxation) {
        decomposition_grid_policy_ = policy;
        decomposition_inter_node_weight_ = inter_node_weight;
        decomposition_intra_node_weight_ = intra_node_weight;
        decomposition_dynamic_interval_ = dynamic_interval;
        decomposition_dynamic_imbalance_threshold_ = dynamic_imbalance_threshold;
        decomposition_dynamic_relaxation_ = dynamic_relaxation;
    }

    // The halo exchange runs on dsDecompX's MPI backend and the MPI library takes device pointers
    // (dsDecompX decides: the backend is what its build provides, CUDA-aware support is asked of the
    // MPI library). False for NCCL/NVSHMEM and for a single rank.
    bool mpi_takes_device_pointers() const {
        return decomp_runtime_ && decomp_runtime_->mpi_takes_device_pointers();
    }

    void sync_decomposition_domain();
    void warn_if_domain_narrower_than_two_halos(Real halo_width);
    void invalidate_halo_cache();
    void refresh_local_bonded_topology();
    void set_bonded_forces_redundant(bool redundant) {
        if (bonded_redundant_ != redundant) {
            bonded_redundant_ = redundant;
            bonded_layout_local_count_ = static_cast<size_t>(-1);  // lists must be rebuilt
        }
    }
    bool bonded_forces_redundant() const { return bonded_redundant_; }

    dsdecompx::HaloResult perform_halo_exchange(bool rebuild = false);
    bool sync_halo_particles(bool rebuild_halo);
    bool sync_distributed_particles(bool local_rebuild_hint);
    bool unify_rebuild_decision(bool local_rebuild_flag) const;
    void perform_reverse_halo_exchange();

    void migrate_particles();
    bool migrate_particles_check();
    void redistribute_particles_after_scale();

#ifdef USE_CUDA
    bool exchange_halo_on_device(class ForceCalculatorCUDA& calculator,
                                 bool rebuild_halo);
    // Whether the last device halo exchange refreshed its plan in place. Same
    // value on every rank, so callers may use it to skip a collective.
    bool last_device_halo_plan_reused() const { return last_device_halo_plan_reused_; }

    // Split-phase device halo exchange for overlapping with force work on owned
    // atoms. can_overlap_halo_on_device() is true when a non-rebuild exchange
    // would refresh the existing plan in place (the ghost layout and count are
    // known up front). begin posts the exchange on the calculator's halo stream;
    // end completes it, unpacks the ghosts and orders the force stream after the
    // unpack. Nothing may resize or exchange particles in between.
    bool can_overlap_halo_on_device(class ForceCalculatorCUDA& calculator);
    void begin_halo_exchange_on_device(class ForceCalculatorCUDA& calculator);
    void end_halo_exchange_on_device(class ForceCalculatorCUDA& calculator);
    bool migrate_particles_on_device(class ForceCalculatorCUDA& calculator);
    void reverse_halo_on_device(class ForceCalculatorCUDA& calculator);
    // Moves one application field along the halo plan of the last device halo exchange (dsDecompX forward_exchange /
    // reverse_exchange): `data` is [rows, width] row-major in device memory, rows = owned atoms then ghost atoms, float or double
    // (element_bytes 4 or 8). forward: the owners' rows are copied into the ghost rows; reverse: the ghost rows are added to the
    // rows of their owners. Queued on `stream` (a cudaStream_t). The node features of a layer-exchange MLIP use it (MLIPHaloTransport).
    void exchange_field_on_device(class ForceCalculatorCUDA& calculator, void* data, int64_t rows, int64_t width, int element_bytes,
                                  bool reverse, void* stream);
#endif

};

#endif // MD_SYSTEM_H
