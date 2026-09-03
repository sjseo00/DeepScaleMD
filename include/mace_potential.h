/**
 * MACE Potential Calculator using LibTorch (C++ PyTorch API)
 * STAGE 6: ML Potential Integration
 * 
 * This provides a C++ interface to MACE models WITHOUT Python dependencies.
 * Models must be converted to TorchScript format (.pt files).
 */

#ifndef MACE_POTENTIAL_H
#define MACE_POTENTIAL_H

#include "md_types.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include <memory>

class MACEPotential {
public:
    MACEPotential(const std::string& model_path, const std::string& device = "cpu");
    ~MACEPotential();
    
    /**
     * Initialize the MACE model
     */
    bool initialize();
    
    /**
     * Compute energy and forces for a system
     * 
     * @param positions [N, 3] particle positions
     * @param atom_types [N] atomic numbers (e.g., 18 for Ar)
     * @param cell [3, 3] simulation cell vectors
     * @param pbc [3] periodic boundary conditions
     * @param N number of atoms
     * @param energy output: total energy (eV)
     * @param forces output: [N, 3] forces (eV/Å)
     * @return true if successful
     */
    bool compute(
        const Real* positions,
        const int* atom_types,
        const Real* cell,
        const bool* pbc,
        int N,
        Real& energy,
        Real* forces
    );
    
    /**
     * Get model information
     */
    std::string get_model_info() const;
    
    /**
     * Check if model is initialized
     */
    bool is_initialized() const { return model_loaded_; }
    
private:
    std::string model_path_;
    std::string device_str_;
    torch::Device device_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    bool use_cuda_;
    
    // Cached tensors to avoid reallocation
    torch::Tensor positions_tensor_;
    torch::Tensor atom_types_tensor_;
    torch::Tensor cell_tensor_;
    torch::Tensor pbc_tensor_;
    
    int last_n_atoms_;
    
    /**
     * Convert atom types to MACE format
     */
    torch::Tensor prepare_atom_types(const int* atom_types, int N);
    
    /**
     * Prepare input dictionary for MACE model
     */
    c10::Dict<std::string, torch::Tensor> prepare_input(
        const Real* positions,
        const int* atom_types,
        const Real* cell,
        const bool* pbc,
        int N
    );
};

#endif // MACE_POTENTIAL_H
