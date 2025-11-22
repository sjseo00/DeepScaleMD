#ifndef MD_SYSTEM_H
#define MD_SYSTEM_H


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
    
    // Domain decomposition
    int neighbor_ranks[26];  // Maximum 26 neighbors in 3D (including corners)
    int num_neighbors;
    Real domain_min[3], domain_max[3];  // Local domain boundaries
    
    // New halo exchange data (face-focused, directional)
    struct HaloParticle {
        Real x, y, z;
        Real vx, vy, vz;
        Real fx, fy, fz;  // forces for reverse halo exchange
        Real mass, charge;
        int type;
        int global_id; // global particle id for MPI pair ownership
    };
    struct NeighborInfo { int rank; int dx; int dy; int dz; }; // direction vector (-1,0,1)
    std::vector<NeighborInfo> neighbor_info;              // up to 26 entries
    std::vector<std::vector<HaloParticle>> halo_send_buffers; // per neighbor packed particles (host)
    std::vector<std::vector<HaloParticle>> halo_recv_buffers; // per neighbor received particles (host)
    Real halo_width_;  // current halo width (cutoff + skin)
#ifdef USE_CUDA
    // Optional CUDA device staging (CUDA-aware MPI assumed; if not, host buffers are used)
    std::vector<void*> d_halo_send; // device pointers mirrored to halo_send_buffers
    std::vector<void*> d_halo_recv; // device pointers mirrored to halo_recv_buffers
#endif
    
    // MPI Cartesian topology (cached to avoid repeated creation)
#ifdef USE_MPI
    MPI_Comm cart_comm;
    int cart_dims[3];      // Cartesian grid dimensions
    int cart_coords[3];    // This rank's coordinates in grid
#endif
    
    // Reusable buffers for MPI communication (memory pool for reduced allocation overhead)
    std::vector<Real> mpi_send_buffer_flat;  // Persistent flat send buffer for Alltoallv
    std::vector<Real> mpi_recv_buffer_flat;  // Persistent flat receive buffer for Alltoallv
    std::vector<int> mpi_send_counts;        // Send counts for each neighbor
    std::vector<int> mpi_recv_counts;        // Receive counts for each neighbor
    std::vector<int> mpi_send_disps;         // Send displacements for Alltoallv
    std::vector<int> mpi_recv_disps;         // Receive displacements for Alltoallv
    
    // Device management
    void* d_particles;  // Device particle data
    bool use_gpu;
    bool cuda_direct_halo_; // use CUDA-aware MPI direct device buffers if available
    size_t ghost_count_ = 0; // number of ghost particles currently appended

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
    size_t get_ghost_count() const { return ghost_count_; }
    size_t get_total_particles() const { return get_local_particles() + ghost_count_; }  // local + ghost
    int get_rank() const { return rank; }
    int get_size() const { return size; }
    bool using_gpu() const { return use_gpu; }
    Real get_physical_cutoff() const { return physical_cutoff; }
    void set_physical_cutoff(Real cutoff) { physical_cutoff = cutoff; }
    int get_local_start() const { return local_start; }
    
    // Domain decomposition getters
    int get_num_neighbors() const { return num_neighbors; }
    const int* get_neighbor_ranks() const { return neighbor_ranks; }
    
    // Halo exchange methods (new API)
    void setup_neighbor_ranks();            // builds neighbor_info and Cartesian topology
    void update_domain_bounds();            // compute domain_min/max arrays from cart coords
    void build_halo_send_buffers();         // select boundary particles per neighbor and pack
    void perform_halo_exchange();           // non-blocking MPI exchange (CUDA-aware if enabled)
    void perform_reverse_halo_exchange();   // send back accumulated ghost forces
    size_t get_total_halo_particles() const; // convenience for diagnostics
    const std::vector<std::vector<HaloParticle>>& get_halo_recv_buffers() const { return halo_recv_buffers; }
    // Ghost integration helpers
    void assemble_ghost_particles();    // append received ghosts to particles arrays
    void remove_ghost_particles();      // truncate particles arrays back to local only

    // MPI datatype creation for HaloParticle (lazily initialized)
    static MPI_Datatype get_halo_particle_type();
    static void free_halo_particle_type();
};

#endif // MD_SYSTEM_H
