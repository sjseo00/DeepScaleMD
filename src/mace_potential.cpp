/**
 * MACE Potential Calculator Implementation
 * STAGE 6: ML Potential Integration
 */

#include "mace_potential.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

MACEPotential::MACEPotential(const std::string& model_path, const std::string& device)
    : model_path_(model_path)
    , device_str_(device)
    , device_(device == "cuda" ? torch::kCUDA : torch::kCPU)
    , model_loaded_(false)
    , use_cuda_(device == "cuda")
    , last_n_atoms_(0)
{
    std::cout << "[MACE] Initializing MACE potential" << std::endl;
    std::cout << "[MACE] Model path: " << model_path_ << std::endl;
    std::cout << "[MACE] Device: " << device_str_ << std::endl;
}

MACEPotential::~MACEPotential() {
    if (model_loaded_) {
        std::cout << "[MACE] Cleaning up MACE model" << std::endl;
    }
}

bool MACEPotential::initialize() {
    try {
        std::cout << "[MACE] Loading model from: " << model_path_ << std::endl;
        
        // Load TorchScript model
        model_ = torch::jit::load(model_path_, device_);
        model_.eval();  // Set to evaluation mode
        
        // Optimize for inference
        torch::jit::optimize_for_inference(model_);
        
        std::cout << "[MACE] ✅ Model loaded successfully" << std::endl;
        std::cout << "[MACE] Model optimized for inference" << std::endl;
        
        model_loaded_ = true;
        return true;
        
    } catch (const c10::Error& e) {
        std::cerr << "[MACE] ❌ Error loading model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[MACE] ❌ Error: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

torch::Tensor MACEPotential::prepare_atom_types(const int* atom_types, int N) {
    // MACE expects one-hot encoded atom types
    // For now, we'll create a simple type tensor
    // TODO: Implement proper one-hot encoding based on MACE requirements
    
    std::vector<int64_t> types_vec(N);
    for (int i = 0; i < N; i++) {
        types_vec[i] = static_cast<int64_t>(atom_types[i]);
    }
    
    auto types_tensor = torch::from_blob(
        types_vec.data(),
        {N},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone().to(device_);
    
    return types_tensor;
}

c10::Dict<std::string, torch::Tensor> MACEPotential::prepare_input(
    const Real* positions,
    const int* atom_types,
    const Real* cell,
    const bool* pbc,
    int N
) {
    // Create input dictionary for MACE
    c10::Dict<std::string, torch::Tensor> input;
    
    // Positions [N, 3]
    auto pos_tensor = torch::from_blob(
        const_cast<Real*>(positions),
        {N, 3},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(device_);
    
    // Atom types [N]
    auto types_tensor = prepare_atom_types(atom_types, N);
    
    // Cell [3, 3]
    auto cell_tensor = torch::from_blob(
        const_cast<Real*>(cell),
        {3, 3},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(device_);
    
    // PBC [3]
    std::vector<int64_t> pbc_vec = {
        pbc[0] ? 1L : 0L,
        pbc[1] ? 1L : 0L,
        pbc[2] ? 1L : 0L
    };
    auto pbc_tensor = torch::from_blob(
        pbc_vec.data(),
        {3},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone().to(device_);
    
    // Batch tensor (all atoms in same batch)
    auto batch_tensor = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    
    input.insert("positions", pos_tensor);
    input.insert("atom_types", types_tensor);
    input.insert("cell", cell_tensor);
    input.insert("pbc", pbc_tensor);
    input.insert("batch", batch_tensor);
    
    return input;
}

bool MACEPotential::compute(
    const Real* positions,
    const int* atom_types,
    const Real* cell,
    const bool* pbc,
    int N,
    Real& energy,
    Real* forces
) {
    if (!model_loaded_) {
        std::cerr << "[MACE] ❌ Model not loaded" << std::endl;
        return false;
    }
    
    try {
        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;
        
        // Prepare input
        auto input_dict = prepare_input(positions, atom_types, cell, pbc, N);
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_dict);
        
        auto output = model_.forward(inputs);
        
        // Extract output dictionary
        auto output_dict = output.toGenericDict();
        
        // Extract energy
        if (output_dict.contains("energy")) {
            auto energy_tensor = output_dict.at("energy").toTensor();
            energy = energy_tensor.item<Real>();
        } else {
            std::cerr << "[MACE] ❌ No energy in output" << std::endl;
            return false;
        }
        
        // Extract forces
        if (output_dict.contains("forces")) {
            auto forces_tensor = output_dict.at("forces").toTensor();
            
            // Copy forces to output array
            if (use_cuda_) {
                // GPU -> CPU copy
                auto forces_cpu = forces_tensor.to(torch::kCPU);
                std::memcpy(forces, forces_cpu.data_ptr<Real>(), N * 3 * sizeof(Real));
            } else {
                // Direct copy (already on CPU)
                std::memcpy(forces, forces_tensor.data_ptr<Real>(), N * 3 * sizeof(Real));
            }
        } else {
            std::cerr << "[MACE] ❌ No forces in output" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const c10::Error& e) {
        std::cerr << "[MACE] ❌ Error in compute: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[MACE] ❌ Error: " << e.what() << std::endl;
        return false;
    }
}

std::string MACEPotential::get_model_info() const {
    std::string info = "MACE Potential\n";
    info += "  Model: " + model_path_ + "\n";
    info += "  Device: " + device_str_ + "\n";
    info += "  Status: " + std::string(model_loaded_ ? "Loaded" : "Not loaded");
    return info;
}
