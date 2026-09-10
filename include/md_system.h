#ifndef MD_SYSTEM_H
#define MD_SYSTEM_H

#include "runtime.hpp"
#include "force_manager.h"
#include "md_types.h"
#include "topology.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mpi.h>

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
    NeighborList neighbor_list;

    std::unique_ptr<Topology> topology;
    std::unique_ptr<ForceManager> force_manager;

    Real physical_cutoff;
    Real verlet_skin_ = 2.0;

    int rank;
    int size;
    int local_start;
    int local_end;
    size_t num_local_particles_;
    std::vector<std::string> coordinate_elements_;

    std::unique_ptr<dsdecompx::Decomp> decomp_runtime_;

    void* d_particles;
    bool use_gpu;
    bool cuda_direct_halo_;
    size_t ghost_count_ = 0;
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
    void allocate_device_memory();
    void copy_to_device();
    void copy_from_device();

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
    NeighborList& get_neighbor_list() { return neighbor_list; }
    Topology& get_topology() { return *topology; }
    bool has_topology() const { return static_cast<bool>(topology); }
    const std::vector<LocalBond>& get_local_bonds() const { return local_bonds_; }
    const std::vector<LocalAngle>& get_local_angles() const { return local_angles_; }
    const std::vector<LocalDihedral>& get_local_dihedrals() const { return local_dihedrals_; }
    const std::vector<LocalDihedral>& get_local_impropers() const { return local_impropers_; }
    const std::vector<LocalSpecialPair>& get_local_special_pairs() const { return local_special_pairs_; }
    const std::vector<LocalConstraint>& get_local_constraints() const { return local_constraints_; }
    ForceManager& get_force_manager() { return *force_manager; }

    size_t get_num_particles() const { return particles.num_particles; }
    size_t get_local_particles() const { return static_cast<size_t>(local_end - local_start); }
    size_t get_ghost_count() const { return ghost_count_; }
    size_t get_total_particles() const { return get_local_particles() + ghost_count_; }

    int get_rank() const { return rank; }
    int get_size() const { return size; }
    bool using_gpu() const { return use_gpu; }

    Real get_physical_cutoff() const { return physical_cutoff; }
    void set_physical_cutoff(Real cutoff) { physical_cutoff = cutoff; }
    void set_verlet_skin(Real skin) { verlet_skin_ = skin; }
    int get_local_start() const { return local_start; }

    void sync_decomposition_domain();
    void invalidate_halo_cache();
    void refresh_local_bonded_topology();

    dsdecompx::HaloResult perform_halo_exchange(bool rebuild = false);
    bool sync_halo_particles(bool rebuild_halo, NeighborList* nlist = nullptr);
    bool sync_distributed_particles(bool local_rebuild_hint,
                                    NeighborList* nlist = nullptr);
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
