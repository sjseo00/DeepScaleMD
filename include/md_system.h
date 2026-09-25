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
    struct LocalBond {
        size_t atom1, atom2;
        std::string type;
    };
    struct LocalAngle {
        size_t atom1, atom2, atom3;
        std::string type;
    };
    struct LocalDihedral {
        size_t atom1, atom2, atom3, atom4;
        std::string type;
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

    std::unique_ptr<dsdecompx::Decomp> decomp_runtime_;

    std::vector<LocalBond> local_bonds_;
    std::vector<LocalAngle> local_angles_;
    std::vector<LocalDihedral> local_dihedrals_;
    std::vector<LocalDihedral> local_impropers_;
    std::vector<LocalSpecialPair> local_special_pairs_;
    std::vector<LocalConstraint> local_constraints_;
    std::vector<int> bonded_layout_global_ids_;
    size_t bonded_layout_local_count_ = static_cast<size_t>(-1);
    void resize_particle_storage(size_t count);
    void initialize_decomposition_runtime();

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

    void sync_decomposition_domain();
    void invalidate_halo_cache();
    void refresh_local_bonded_topology();

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
    bool migrate_particles_on_device(class ForceCalculatorCUDA& calculator);
    void reverse_halo_on_device(class ForceCalculatorCUDA& calculator);
#endif

};

#endif // MD_SYSTEM_H
