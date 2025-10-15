#include "include/neighbor_list.h"
#include "include/md_types.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

// Simple test to verify neighbor list methods work correctly
int main() {
    // Create test system
    const size_t N = 1000;
    const Real box_size = 10.0;
    const Real cutoff = 2.5;
    
    // Initialize particles in a simple cubic lattice
    ParticleData particles;
    particles.num_particles = N;
    particles.x.resize(N);
    particles.y.resize(N);
    particles.z.resize(N);
    particles.vx.resize(N, 0.0);
    particles.vy.resize(N, 0.0);
    particles.vz.resize(N, 0.0);
    particles.mass.resize(N, 1.0);
    particles.charge.resize(N, 0.0);
    particles.type.resize(N, 0);
    
    // Create simple lattice
    int nx = static_cast<int>(std::cbrt(N));
    Real spacing = box_size / nx;
    
    for (size_t i = 0; i < N; i++) {
        int ix = i % nx;
        int iy = (i / nx) % nx;
        int iz = i / (nx * nx);
        
        particles.x[i] = ix * spacing + 0.1 * (rand() / static_cast<Real>(RAND_MAX) - 0.5);
        particles.y[i] = iy * spacing + 0.1 * (rand() / static_cast<Real>(RAND_MAX) - 0.5);
        particles.z[i] = iz * spacing + 0.1 * (rand() / static_cast<Real>(RAND_MAX) - 0.5);
    }
    
    // Setup simulation box
    SimulationBox box;
    box.lx = box_size;
    box.ly = box_size;
    box.lz = box_size;
    box.periodic_x = true;
    box.periodic_y = true;
    box.periodic_z = true;
    
    std::cout << "Testing neighbor list methods with " << N << " particles" << std::endl;
    std::cout << "Box size: " << box_size << ", Cutoff: " << cutoff << std::endl << std::endl;
    
    // Test each method
    std::vector<std::string> methods = {"cell_list", "verlet_cell_list", "morton_hilbert"};
    std::vector<NeighborListType> types = {
        NeighborListType::CELL_LIST,
        NeighborListType::VERLET_CELL_LIST,
#if defined(USE_CUDA) || defined(USE_SYCL)
        NeighborListType::MORTON_HILBERT_CSR
#endif
    };
    
    for (size_t m = 0; m < methods.size(); m++) {
#if !defined(USE_CUDA) && !defined(USE_SYCL)
        if (methods[m] == "morton_hilbert") continue;
#endif
        
        std::cout << "Testing " << methods[m] << "..." << std::endl;
        
        try {
            auto builder = create_neighbor_list_builder(types[m], cutoff, "1.0", 0);
            builder->initialize();
            
            NeighborList nlist;
            builder->build(particles, box, nlist);
            
            // Count total neighbors
            size_t total_neighbors = nlist.neighbors.size();
            Real avg_neighbors = total_neighbors / static_cast<Real>(N);
            
            std::cout << "  Total neighbors: " << total_neighbors << std::endl;
            std::cout << "  Average neighbors per particle: " << avg_neighbors << std::endl;
            std::cout << "  Max neighbors per particle: " << nlist.max_neighbors_per_atom << std::endl;
            
            // Basic sanity checks
            if (total_neighbors == 0) {
                std::cout << "  ERROR: No neighbors found!" << std::endl;
            } else if (avg_neighbors > 100) {
                std::cout << "  WARNING: Suspiciously high neighbor count - possible divergence" << std::endl;
            } else {
                std::cout << "  PASS: Reasonable neighbor count" << std::endl;
            }
            
            builder->finalize();
            
        } catch (const std::exception& e) {
            std::cout << "  ERROR: Exception caught - " << e.what() << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    return 0;
}