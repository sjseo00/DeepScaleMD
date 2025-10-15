#ifndef MD_SYSTEM_H
#define MD_SYSTEM_H

#define OMPI_SKIP_MPICXX 1
#include "md_types.h"
#include "topology.h"
#include "force_manager.h"
#include <string>
#include <memory>

#ifdef USE_MPI
#include <mpi.h>
#endif

class MDSystem {
private:
    ParticleData particles;
    SimulationBox box;
    ForceFieldParams ff_params;
    NeighborList neighbor_list;
    std::unique_ptr<Topology> topology;
    std::unique_ptr<ForceManager> force_manager;
    
    // Physical parameters
    Real physical_cutoff;  // Physical cutoff distance (without skin)
    
    // MPI variables
    int rank, size;
    int local_start, local_end;
    
    // Domain decomposition and halo exchange
    int neighbor_ranks[26];  // Maximum 26 neighbors in 3D (including corners)
    int num_neighbors;
    Real domain_min[3], domain_max[3];  // Local domain boundaries
    std::vector<int> halo_particles;    // Indices of halo particles
    std::vector<ParticleData> neighbor_halos;  // Halo particles from neighbors
    
    // Device management
    void* d_particles;  // Device particle data
    bool use_gpu;

public:
    MDSystem();
    virtual ~MDSystem();
    
    // Initialization
    virtual void initialize();
    virtual void finalize();
    void initialize_mpi();
    void load_config(const std::string& config_file);
    void load_coordinates(const std::string& coord_file);
    void load_topology(const std::string& topo_file);
    void initialize_velocities(Real temperature);
    void setup_domain_decomposition();
    void allocate_device_memory();
    void copy_to_device();
    void copy_from_device();
    
    // Halo exchange methods
    void setup_neighbor_ranks();
    void identify_halo_particles();
    void exchange_halo_particles();
    void pack_halo_data(int neighbor_rank, std::vector<Real>& buffer);
    void unpack_halo_data(int neighbor_rank, const std::vector<Real>& buffer);
    void update_ghost_positions();
    
    // Topology-related methods
    void load_standard_xyz(const std::string& coord_file);
    void assign_masses_from_topology();
    void assign_charges_from_topology();
    void setup_force_field_params();
    void initialize_force_manager(const std::map<std::string, std::string>& config_map);
    
    // Force computation
    void compute_forces();
    
    // Getters
    ParticleData& get_particles() { return particles; }
    SimulationBox& get_simulation_box() { return box; }
    SimulationBox& get_box() { return box; }  // Alias for get_simulation_box()
    ForceFieldParams& get_ff_params() { return ff_params; }
    NeighborList& get_neighbor_list() { return neighbor_list; }
    Topology& get_topology() { return *topology; }
    ForceManager& get_force_manager() { return *force_manager; }
    
    size_t get_num_particles() const { return particles.num_particles; }
    size_t get_local_particles() const { return local_end - local_start; }
    int get_rank() const { return rank; }
    int get_size() const { return size; }
    bool using_gpu() const { return use_gpu; }
    Real get_physical_cutoff() const { return physical_cutoff; }
    void set_physical_cutoff(Real cutoff) { physical_cutoff = cutoff; }
    
    // Halo exchange getters
    const std::vector<int>& get_halo_particles() const { return halo_particles; }
    const std::vector<ParticleData>& get_neighbor_halos() const { return neighbor_halos; }
    int get_num_neighbors() const { return num_neighbors; }
    const int* get_neighbor_ranks() const { return neighbor_ranks; }
};

#endif // MD_SYSTEM_H
