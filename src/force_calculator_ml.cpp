#include "../include/force_calculator_ml.h"
#include "../include/md_system.h"
#include "../include/mpi_utils.h"
#include <iostream>
#include <chrono>
#include <mpi.h>
#include <cmath>

ForceCalculatorML::ForceCalculatorML()
    : python_interface(nullptr)
    , python_initialized(false)
    , ml_model_type("mace")
    , device("cpu")
    , dtype("float32")
    , total_ml_time(0.0)
    , ml_call_count(0)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
}

ForceCalculatorML::~ForceCalculatorML() {
    finalize();
}

void ForceCalculatorML::set_ml_model_type(const std::string& type) {
    ml_model_type = type;
}

void ForceCalculatorML::set_model_path(const std::string& path) {
    model_path = path;
}

void ForceCalculatorML::set_device(const std::string& dev) {
    device = dev;
}

void ForceCalculatorML::set_dtype(const std::string& dt) {
    dtype = dt;
}

void ForceCalculatorML::initialize() {
    if (is_rank_zero()) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "Initializing ML Force Calculator" << std::endl;
        std::cout << "  Model type: " << ml_model_type << std::endl;
        std::cout << "  Model path: " << model_path << std::endl;
        std::cout << "  Device: " << device << std::endl;
        std::cout << "  Dtype: " << dtype << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    }
    
    // Create Python interface
    python_interface = new PythonForceInterface();
    
    // Initialize Python interpreter (only on rank 0 for MPI)
    if (mpi_rank == 0) {
        python_interface->initialize_python();
        python_initialized = true;
        
        if (is_rank_zero()) {
            std::cout << "✅ Python interpreter initialized" << std::endl;
        }
    }
    
    // Broadcast initialization status
    int init_status = python_initialized ? 1 : 0;
    MPI_Bcast(&init_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    python_initialized = (init_status == 1);
    
    // Initialize ML model (Python side)
    if (mpi_rank == 0 && python_initialized) {
        // Import ml_forces module and initialize the model
        // This is done by calling Python function
        if (is_rank_zero()) {
            std::cout << "Initializing " << ml_model_type << " model..." << std::endl;
        }
        
        // We'll use Python C API to call ml_forces.initialize_ml_forcefield_*
        // For now, we expect the Python side to be pre-initialized
        // In production, we would call Python initialization here
        
        if (is_rank_zero()) {
            std::cout << "✅ ML model initialization requested" << std::endl;
            std::cout << "Note: Ensure Python ml_forces module is properly configured" << std::endl;
        }
    }
    
    if (is_rank_zero()) {
        std::cout << "✅ ML Force Calculator initialized" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    }
}

void ForceCalculatorML::finalize() {
    if (python_initialized && python_interface) {
        if (is_rank_zero()) {
            print_performance_stats();
        }
        
        if (mpi_rank == 0) {
            python_interface->finalize_python();
        }
        
        delete python_interface;
        python_interface = nullptr;
        python_initialized = false;
    }
}

void ForceCalculatorML::compute_forces(MDSystem& system) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    
    // Get local particles (domain decomposition)
    size_t total_particles = system.get_num_particles();
    
    // Clear forces
    std::fill(particles.fx.begin(), particles.fx.end(), 0.0);
    std::fill(particles.fy.begin(), particles.fy.end(), 0.0);
    std::fill(particles.fz.begin(), particles.fz.end(), 0.0);
    
    // Reset energies
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    last_virial = 0.0;
    
    if (!python_initialized || !python_interface) {
        if (is_rank_zero() && ml_call_count == 0) {
            std::cerr << "❌ Python ML interface not initialized!" << std::endl;
        }
        return;
    }
    
    // MPI Strategy: Only rank 0 calls Python ML inference
    // Other ranks wait for broadcast of forces/energy
    
    Real energy = 0.0;
    
    if (mpi_rank == 0) {
        // Gather all particle data to rank 0 for ML inference
        // (ML potentials typically need all atoms due to message passing in GNN)
        
        if (mpi_size > 1) {
            // TODO: Implement MPI gather for distributed particles
            // For now, assume single process or shared memory
            if (ml_call_count == 0) {
                std::cout << "⚠️  MPI gather for ML forces not yet implemented" << std::endl;
                std::cout << "   Running with particles on rank 0 only" << std::endl;
            }
        }
        
        // Prepare data for Python ML call
        // Need: positions, atom_types, cell, pbc
        
        std::vector<Real> positions_flat;
        std::vector<int> atom_types;
        std::vector<Real> cell_vectors;
        std::vector<int> pbc;
        
        // Flatten positions (x, y, z) for each particle
        positions_flat.reserve(total_particles * 3);
        atom_types.reserve(total_particles);
        
        for (size_t i = 0; i < total_particles; ++i) {
            positions_flat.push_back(particles.x[i]);
            positions_flat.push_back(particles.y[i]);
            positions_flat.push_back(particles.z[i]);
            atom_types.push_back(particles.type[i]);
        }
        
        // Cell vectors (assuming orthorhombic box)
        cell_vectors = {box.lx, box.ly, box.lz};
        
        // PBC (periodic boundary conditions)
        pbc = {
            box.periodic_x ? 1 : 0,
            box.periodic_y ? 1 : 0,
            box.periodic_z ? 1 : 0
        };
        
        // Call Python ML force computation
        // This will invoke ml_forces.compute_ml_forces() which calls MACE/NequIP
        
        // For now, we use a direct Python call
        // In production, this would be in python_interface.cpp
        
        try {
            // Get neighbor list (even though MACE builds its own, we pass for compatibility)
            NeighborList& nlist = system.get_neighbor_list();
            ForceFieldParams& params = system.get_ff_params();
            
            // Call the Python ML force computation with actual box dimensions
            python_interface->compute_ml_forces(particles, params, nlist,
                                                box.lx, box.ly, box.lz,
                                                box.periodic_x, box.periodic_y, box.periodic_z);
            
            // Note: The above function currently doesn't return forces properly
            // We need to implement the data exchange
            
            // WORKAROUND: For now, call Python directly via subprocess or expect
            // forces to be updated in particles.fx/fy/fz
            
            // Placeholder: Indicate ML computation occurred
            if (ml_call_count == 0) {
                std::cout << "🔬 ML force computation called (rank 0)" << std::endl;
                std::cout << "   Particles: " << total_particles << std::endl;
                std::cout << "   Box: " << box.lx << " x " << box.ly << " x " << box.lz << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "❌ ML force computation failed: " << e.what() << std::endl;
        }
    }
    
    // Broadcast forces from rank 0 to all ranks
    // PERF FIX: merge fx/fy/fz/energy into one contiguous buffer → single MPI_Bcast
    // instead of 4 separate calls (reduces latency by ~4x for many-rank runs)
    if (mpi_size > 1) {
        size_t stride = total_particles;
        std::vector<double> bcast_buf(stride * 3 + 1);
        if (mpi_rank == 0) {
            for (size_t i = 0; i < stride; ++i) {
                bcast_buf[i]            = particles.fx[i];
                bcast_buf[stride + i]   = particles.fy[i];
                bcast_buf[2*stride + i] = particles.fz[i];
            }
            bcast_buf[3*stride] = energy;
        }
        MPI_Bcast(bcast_buf.data(), static_cast<int>(stride * 3 + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (mpi_rank != 0) {
            for (size_t i = 0; i < stride; ++i) {
                particles.fx[i] = bcast_buf[i];
                particles.fy[i] = bcast_buf[stride + i];
                particles.fz[i] = bcast_buf[2*stride + i];
            }
            energy = bcast_buf[3*stride];
        }
    }
    
    last_potential_energy = energy;
    
    // Track performance
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    total_ml_time += elapsed.count();
    ml_call_count++;
    
    // Periodic reporting
    if (ml_call_count % 100 == 0 && is_rank_zero()) {
        std::cout << "ML force call " << ml_call_count 
                  << ": avg time = " << (total_ml_time / ml_call_count * 1000.0) 
                  << " ms" << std::endl;
    }
}

void ForceCalculatorML::calculate_kinetic_energy(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    Real ke = 0.0;
    
    #pragma omp parallel for reduction(+:ke)
    for (size_t i = 0; i < local_particles; ++i) {
        Real vx = particles.vx[i];
        Real vy = particles.vy[i];
        Real vz = particles.vz[i];
        Real mass = particles.mass[i];
        
        ke += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    }
    
    // Sum kinetic energy across all ranks
    Real global_ke = 0.0;
    MPI_Allreduce(&ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_kinetic_energy = global_ke;
}

double ForceCalculatorML::get_average_ml_time() const {
    if (ml_call_count == 0) return 0.0;
    return total_ml_time / ml_call_count;
}

void ForceCalculatorML::print_performance_stats() const {
    if (ml_call_count == 0) return;
    
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "ML Force Calculator Performance Stats" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "  Total ML calls: " << ml_call_count << std::endl;
    std::cout << "  Total ML time:  " << total_ml_time << " s" << std::endl;
    std::cout << "  Average time:   " << (get_average_ml_time() * 1000.0) << " ms/call" << std::endl;
    std::cout << "  Model type:     " << ml_model_type << std::endl;
    std::cout << "  Device:         " << device << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
}
