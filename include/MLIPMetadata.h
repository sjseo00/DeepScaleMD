#ifndef MLIP_METADATA_H
#define MLIP_METADATA_H

// metadata.json of a converted model folder (docs/mlip_integration_design.md 3.5).
// No LibTorch dependency: the file is read with a small JSON parser inside MLIPMetadata.cpp.

#include <map>
#include <string>
#include <vector>

struct MLIPMetadata {
    int format_version = 0;                   // only 1 is understood
    std::string backend;                      // "torchscript" | "aoti"
    std::string model_family;                 // e.g. "mace"
    std::string model_name;
    double cutoff = 0.0;                      // neighbor cutoff (A)
    int num_layers = 0;                       // message-passing layers; 1 for strictly local models
    double ghost_cutoff = 0.0;                // ghost depth (A), at least num_layers * cutoff
    std::vector<int> elements;                // supported atomic numbers
    std::string dtype;                        // "float32" | "float64"
    std::string energy_unit;                  // units.energy: "eV" | "kcal/mol"
    std::string length_unit;                  // units.length: "Angstrom"
    std::string head;                         // empty when the model has no head
    std::vector<std::string> required_op_libs;  // absolute .so paths to dlopen before loading the model
    std::vector<std::string> required_python_modules;  // modules to import first (operators registered from Python, e.g.
                                                       // cuequivariance_ops_torch; optional field, empty when absent)
    std::string accel;                        // "e3nn" | "cueq" | "openeq" | "flash"
    std::map<std::string, std::string> versions;  // package versions the model was converted with
    std::string target_gpu;                   // "sm_90" (or "cpu": a test package); required for aoti
    bool edge_shifts = false;                 // optional field: the model takes periodic images as per-edge shift vectors (a fifth
                                              // input) and returns the virial (a fourth output); torchscript only
};

// Parses and validates the JSON text. Throws std::runtime_error whose message starts with `source` and names
// the offending field.
MLIPMetadata parse_mlip_metadata(const std::string& json_text, const std::string& source = "metadata.json");

// `path` is a converted model folder (reads <path>/metadata.json) or a metadata.json file.
MLIPMetadata load_mlip_metadata(const std::string& path);

struct VersionReport {
    std::vector<std::string> errors;    // the model must not be run in this environment
    std::vector<std::string> warnings;  // worth telling the user
};

// Compares the versions the model was converted with (meta.versions "torch" and "cuda") with the LibTorch and CUDA runtime
// versions in use, written as the packages write them ("2.7.1", "2.7.1+cu118", "11.8"). Rules:
//   torch  major.minor differs -> error (TorchScript and AOTI files are tied to the torch release); patch differs -> warning
//   cuda   major differs -> error for aoti (compiled kernels), warning for torchscript; minor differs -> warning
// A version that is not recorded (null: the package was absent at conversion) is not compared; one that cannot be read as a
// version number gives a warning. Other packages (mace, e3nn, ...) are not compared: they are not part of the C++ process.
VersionReport check_runtime_versions(const MLIPMetadata& meta, const std::string& torch_runtime, const std::string& cuda_runtime);

#endif // MLIP_METADATA_H
