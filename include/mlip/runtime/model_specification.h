/**
 * model_specification.h
 *
 * Plain C++ data structures that represent a parsed MLIP YAML model spec.
 * These are constructed once at initialization from YAML and never modified
 * during the hot path.
 *
 * No framework headers (torch, onnx, etc.) are included here.
 */

#pragma once
#ifndef MLIP_MODEL_SPECIFICATION_H
#define MLIP_MODEL_SPECIFICATION_H

#include "../mlip_types.h"   // mlip::ScalarType

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace mlip {

// ─────────────────────────── enums ───────────────────────────────────────────

enum class BackendType {
    PyTorch,
    ONNX,
    TensorRT,
    NativePlugin,
    Unknown
};

enum class CallStyle {
    Positional,
    Dictionary
};

// Use mlip::ScalarType from mlip_types.h instead of a separate DType enum.
// DType is provided as an alias for backward compatibility.
using DType = mlip::ScalarType;

enum class NeighborListKind {
    Half,
    Full
};

enum class ForceConvention {
    Force,              // model outputs F directly
    NegativeGradient,   // F = -dE/dr
    PositiveGradient    // F = +dE/dr
};

enum class VirialConvention {
    VirialTensor,
    NegativeVirialTensor,
    StressTensor,
    NegativeStressTensor,
    PositionForceOuterProduct
};

enum class VirialOrdering {
    Matrix3x3,
    xx_yy_zz_xy_xz_yz,
    xx_yy_zz_yz_xz_xy
};

enum class SpeciesInputType {
    MdType,         // raw MD type id
    AtomicNumber,   // Z
    AtomicSymbol    // "H", "O", ...
};

enum class SpeciesOutputType {
    ModelIndex,
    AtomicNumber
};

// ─────────────────────── shape descriptor ────────────────────────────────────

/// Shape symbols that may appear in YAML.
enum class ShapeSymbol {
    Nlocal,
    Nghost,
    Nall,
    Nedges,
    Literal   // a fixed integer
};

struct ShapeDim {
    ShapeSymbol symbol = ShapeSymbol::Literal;
    int64_t     value  = 0;   // used when symbol == Literal
};

using ShapeSpec = std::vector<ShapeDim>;

// ─────────────────────── unit spec ───────────────────────────────────────────

struct UnitSystem {
    std::string length = "angstrom";  // "angstrom", "nm", "bohr"
    std::string energy = "eV";        // "eV", "kcal/mol", "kJ/mol", "hartree"
};

// ─────────────────────── neighbor spec ───────────────────────────────────────

struct NeighborSpec {
    double cutoff          = 6.0;
    double skin            = 0.5;
    NeighborListKind list  = NeighborListKind::Full;
    bool directed          = true;
    bool self_edges        = false;
    bool include_ghost     = true;
    bool require_edge_vecs = false;
    bool require_distances = false;
    bool require_cell_shifts = false;
};

// ─────────────────────── species mapping ─────────────────────────────────────

struct SpeciesMapSpec {
    SpeciesInputType  input_type  = SpeciesInputType::MdType;
    SpeciesOutputType output_type = SpeciesOutputType::ModelIndex;

    // For symbol-based: e.g. {"H": 0, "O": 1}
    std::unordered_map<std::string, int> symbol_to_index;

    // For integer-based: e.g. {0: 0, 1: 1}
    std::unordered_map<int, int>         int_to_index;
};

// ─────────────────────── input / output binding specs ────────────────────────

struct InputSpec {
    std::string source;             // e.g. "md.positions"
    std::string tensor_name;        // model input name (for dict style)
    int         argument_index = -1;// positional index
    ShapeSpec   shape;
    DType       dtype          = DType::Float32;
    bool        requires_grad  = false;
    std::vector<std::string> transforms; // registered op names in order
};

struct OutputSpec {
    std::string tensor_name;        // model output key or tuple index as string
    int         output_index   = -1;// tuple index
    std::string target;             // e.g. "md.forces"
    DType       expected_dtype = DType::Float32;
    ShapeSpec   expected_shape;
    double      sign           = 1.0;
    double      scale          = 1.0;

    // force/virial specific
    ForceConvention  force_convention  = ForceConvention::Force;
    VirialConvention virial_convention = VirialConvention::VirialTensor;
    VirialOrdering   virial_ordering   = VirialOrdering::Matrix3x3;

    bool includes_ghost = false;

    std::vector<std::string> operations; // post-processing op names in order
};

// ─────────────────────── autograd spec ───────────────────────────────────────

struct AutogradSpec {
    bool        enabled         = false;
    std::string energy_output;          // name of energy output tensor
    std::string differentiate_wrt;      // usually "positions"
    double      force_sign      = -1.0;
    bool        create_graph    = false;
    bool        retain_graph    = false;
};

// ─────────────────────── backend config ──────────────────────────────────────

struct BackendConfig {
    BackendType backend       = BackendType::PyTorch;
    std::string format;          // "torchscript", "onnx", "trt", ...
    std::string entrypoint    = "forward";
    CallStyle   call_style    = CallStyle::Positional;
    bool        share_cuda_stream    = true;
    bool        allow_cpu_fallback   = false;

    // Native plugin only
    std::string plugin_library;
    std::string plugin_factory_symbol;
};

// ─────────────────────── top-level specification ─────────────────────────────

struct MlipModelSpec {
    int         schema_version = 1;
    std::string name;
    std::string model_path;
    std::string device = "cuda";
    std::string precision = "float32";  // "float32" or "float64"

    BackendConfig  backend;
    NeighborSpec   neighbor;
    SpeciesMapSpec species_map;
    AutogradSpec   autograd;
    UnitSystem     md_units;
    UnitSystem     model_units;

    // Ordered by argument_index (positional) or tensor_name (dict)
    std::unordered_map<std::string, InputSpec>  inputs;
    std::unordered_map<std::string, OutputSpec> outputs;

    // Capabilities declared in YAML (cross-checked with runtime)
    bool cap_energy   = false;
    bool cap_forces   = false;
    bool cap_virial   = false;
    bool cap_stress   = false;
    bool cap_external_neighbor_list = true;
    bool cap_external_cuda_stream   = true;
};

} // namespace mlip
#endif // MLIP_MODEL_SPECIFICATION_H
