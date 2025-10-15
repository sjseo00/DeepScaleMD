#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

using Real = double;

// Custom atomicAdd implementation for double precision
// (needed for older CUDA architectures < 6.0)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native double atomicAdd is available
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}
#endif

__global__ void lj_force_kernel(Real* x, Real* y, Real* z,
                               Real* fx, Real* fy, Real* fz,
                               int* neighbors, int* num_neighbors,
                               int* neighbor_offset,
                               Real* epsilon, Real* sigma,
                               int num_particles, Real cutoff,
                               int max_neighbors_total,
                               Real box_x, Real box_y, Real box_z,
                               int num_local_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    Real xi = x[i], yi = y[i], zi = z[i];
    Real fxi = 0.0, fyi = 0.0, fzi = 0.0;
    
    int neighbor_start = neighbor_offset[i];
    int num_neigh = num_neighbors[i];
    
    // Defensive bounds checking
    if (num_neigh > 1000) num_neigh = 1000; // Safety limit
    
    for (int k = 0; k < num_neigh; k++) {
        int neighbor_idx = neighbor_start + k;
        // Bounds check for neighbors array access
        if (neighbor_idx >= 0 && neighbor_idx < max_neighbors_total) {
            int j = neighbors[neighbor_idx];
            if (j >= num_particles || j < 0 || i == j) continue;
        
        Real dx = xi - x[j];
        Real dy = yi - y[j];
        Real dz = zi - z[j];
        
        // Apply periodic boundary conditions
        if (dx > box_x * 0.5) dx -= box_x;
        else if (dx < -box_x * 0.5) dx += box_x;
        if (dy > box_y * 0.5) dy -= box_y;
        else if (dy < -box_y * 0.5) dy += box_y;
        if (dz > box_z * 0.5) dz -= box_z;
        else if (dz < -box_z * 0.5) dz += box_z;
        
        Real r2 = dx*dx + dy*dy + dz*dz;
        
        if (r2 < cutoff*cutoff && r2 > 0.0) {
            Real eps = sqrt(epsilon[i] * epsilon[j]);
            Real sig = 0.5 * (sigma[i] + sigma[j]);
            
            Real sig2 = sig * sig;
            Real sig6 = sig2 * sig2 * sig2;
            Real sig12 = sig6 * sig6;
            
            Real r2inv = 1.0 / r2;
            Real r6inv = r2inv * r2inv * r2inv;
            Real r12inv = r6inv * r6inv;
            
            Real force_mag = 24.0 * eps * r2inv * (2.0 * sig12 * r12inv - sig6 * r6inv);
            
            fxi += force_mag * dx;
            fyi += force_mag * dy;
            fzi += force_mag * dz;
            
            // Apply Newton's third law only for local-local pairs to avoid double counting
            // In MPI mode, only apply if j is also a local particle
            if (j < num_local_particles) {
                atomicAdd(&fx[j], -force_mag * dx);
                atomicAdd(&fy[j], -force_mag * dy);
                atomicAdd(&fz[j], -force_mag * dz);
            }
        }
        }
    }
    
    atomicAdd(&fx[i], fxi);
    atomicAdd(&fy[i], fyi);
    atomicAdd(&fz[i], fzi);
}

__global__ void coulomb_force_kernel(Real* x, Real* y, Real* z,
                                    Real* fx, Real* fy, Real* fz,
                                    Real* charge, int* neighbors, int* num_neighbors,
                                    int* neighbor_offset,
                                    int num_particles, Real cutoff,
                                    int max_neighbors_total,
                                    Real box_x, Real box_y, Real box_z,
                                    int num_local_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    Real xi = x[i], yi = y[i], zi = z[i];
    Real qi = charge[i];
    Real fxi = 0.0, fyi = 0.0, fzi = 0.0;
    
    const Real ke = 8.9875517873681764e9; // Coulomb constant
    
    int neighbor_start = neighbor_offset[i];
    int num_neigh = num_neighbors[i];
    
    // Defensive bounds checking
    if (num_neigh > 1000) num_neigh = 1000; // Safety limit
    
    for (int k = 0; k < num_neigh; k++) {
        int neighbor_idx = neighbor_start + k;
        // Bounds check for neighbors array access
        if (neighbor_idx >= 0 && neighbor_idx < max_neighbors_total) {
            int j = neighbors[neighbor_idx];
            if (j >= num_particles || j < 0 || i == j) continue;
        
        Real dx = xi - x[j];
        Real dy = yi - y[j];
        Real dz = zi - z[j];
        
        // Apply periodic boundary conditions
        if (dx > box_x * 0.5) dx -= box_x;
        else if (dx < -box_x * 0.5) dx += box_x;
        if (dy > box_y * 0.5) dy -= box_y;
        else if (dy < -box_y * 0.5) dy += box_y;
        if (dz > box_z * 0.5) dz -= box_z;
        else if (dz < -box_z * 0.5) dz += box_z;
        
        Real r2 = dx*dx + dy*dy + dz*dz;
        
        if (r2 < cutoff*cutoff && r2 > 0.0) {
            Real r = sqrt(r2);
            Real qj = charge[j];
            
            Real force_mag = ke * qi * qj / (r2 * r);
            
            fxi += force_mag * dx;
            fyi += force_mag * dy;
            fzi += force_mag * dz;
            
            // Apply Newton's third law only for local-local pairs to avoid double counting
            // In MPI mode, only apply if j is also a local particle
            if (j < num_local_particles) {
                atomicAdd(&fx[j], -force_mag * dx);
                atomicAdd(&fy[j], -force_mag * dy);
                atomicAdd(&fz[j], -force_mag * dz);
            }
        }
        }
    }
    
    atomicAdd(&fx[i], fxi);
    atomicAdd(&fy[i], fyi);
    atomicAdd(&fz[i], fzi);
}

void launch_lj_kernel(Real* d_x, Real* d_y, Real* d_z,
                                 Real* d_fx, Real* d_fy, Real* d_fz,
                                 int* d_neighbors, int* d_num_neighbors,
                                 int* d_neighbor_offset,
                                 Real* d_epsilon, Real* d_sigma,
                                 int num_particles, Real cutoff,
                                 int max_neighbors_total,
                                 cudaStream_t stream,
                                 Real box_x, Real box_y, Real box_z,
                                 int num_local_particles) {
    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    
    lj_force_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_neighbors, d_num_neighbors, d_neighbor_offset,
        d_epsilon, d_sigma, num_particles, cutoff, max_neighbors_total,
        box_x, box_y, box_z, num_local_particles
    );
}

void launch_coulomb_kernel(Real* d_x, Real* d_y, Real* d_z,
                                      Real* d_fx, Real* d_fy, Real* d_fz,
                                      Real* d_charge, int* d_neighbors, int* d_num_neighbors,
                                      int* d_neighbor_offset,
                                      int num_particles, Real cutoff,
                                      int max_neighbors_total,
                                      cudaStream_t stream,
                                      Real box_x, Real box_y, Real box_z,
                                      int num_local_particles) {
    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    
    coulomb_force_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz,
        d_charge, d_neighbors, d_num_neighbors, d_neighbor_offset,
        num_particles, cutoff, max_neighbors_total,
        box_x, box_y, box_z, num_local_particles
    );
}

#endif // USE_CUDA
