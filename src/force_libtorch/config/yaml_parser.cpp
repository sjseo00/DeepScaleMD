
#include "force_libtorch/config/yaml_parser.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <set>
#include <stdexcept>
#include <algorithm>

namespace mlip {
namespace {


[[noreturn]] static void err(const std::string& ctx, const std::string& msg) {
    throw std::runtime_error("[MLIP YAML] " + ctx + ": " + msg);
}

static void reject_unknown_keys(const YAML::Node& node,
                                 const std::set<std::string>& allowed,
                                 const std::string& context)
{
    if (!node.IsMap()) return;
    for (auto it = node.begin(); it != node.end(); ++it) {
        std::string k = it->first.as<std::string>();
        if (!allowed.count(k))
            err(context, "Unknown key '" + k + "'");
    }
}

static std::string req_str(const YAML::Node& n, const std::string& key,
                            const std::string& ctx)
{
    if (!n[key]) err(ctx, "Missing required key '" + key + "'");
    return n[key].as<std::string>();
}

static std::string opt_str(const YAML::Node& n, const std::string& key,
                             const std::string& def = "")
{
    if (!n[key]) return def;
    return n[key].as<std::string>();
}

static bool opt_bool(const YAML::Node& n, const std::string& key, bool def = false) {
    if (!n[key]) return def;
    return n[key].as<bool>();
}

static double opt_double(const YAML::Node& n, const std::string& key, double def = 0.0) {
    if (!n[key]) return def;
    return n[key].as<double>();
}

static int opt_int(const YAML::Node& n, const std::string& key, int def = -1) {
    if (!n[key]) return def;
    return n[key].as<int>();
}


static BackendType parse_backend_type(const std::string& s, const std::string& ctx) {
    if (s == "pytorch")       return BackendType::PyTorch;
    if (s == "onnx")          return BackendType::ONNX;
    if (s == "tensorrt")      return BackendType::TensorRT;
    if (s == "native_plugin") return BackendType::NativePlugin;
    err(ctx, "Unknown backend '" + s + "'. Expected: pytorch, onnx, tensorrt, native_plugin");
}

static CallStyle parse_call_style(const std::string& s, const std::string& ctx) {
    if (s == "positional") return CallStyle::Positional;
    if (s == "dictionary") return CallStyle::Dictionary;
    err(ctx, "Unknown call_style '" + s + "'. Expected: positional, dictionary");
}

static DType parse_dtype(const std::string& s, const std::string& ctx) {
    if (s == "float32") return ScalarType::Float32;
    if (s == "float64") return ScalarType::Float64;
    if (s == "int32")   return ScalarType::Int32;
    if (s == "int64")   return ScalarType::Int64;
    if (s == "bool")    return ScalarType::Bool;
    err(ctx, "Unknown dtype '" + s + "'. Expected: float32, float64, int32, int64, bool");
}

static NeighborListKind parse_neighbor_list(const std::string& s, const std::string& ctx) {
    if (s == "half") return NeighborListKind::Half;
    if (s == "full") return NeighborListKind::Full;
    err(ctx, "Unknown neighbor list '" + s + "'. Expected: half, full");
}

static ForceConvention parse_force_convention(const std::string& s, const std::string& ctx) {
    if (s == "force")              return ForceConvention::Force;
    if (s == "negative_gradient")  return ForceConvention::NegativeGradient;
    if (s == "positive_gradient")  return ForceConvention::PositiveGradient;
    err(ctx, "Unknown force convention '" + s + "'");
}

static VirialConvention parse_virial_convention(const std::string& s, const std::string& ctx) {
    if (s == "virial_tensor")                   return VirialConvention::VirialTensor;
    if (s == "negative_virial_tensor")          return VirialConvention::NegativeVirialTensor;
    if (s == "stress_tensor")                   return VirialConvention::StressTensor;
    if (s == "negative_stress_tensor")          return VirialConvention::NegativeStressTensor;
    if (s == "position_force_outer_product")    return VirialConvention::PositionForceOuterProduct;
    err(ctx, "Unknown virial convention '" + s + "'");
}

static VirialOrdering parse_virial_ordering(const std::string& s, const std::string& ctx) {
    if (s == "matrix_3x3")          return VirialOrdering::Matrix3x3;
    if (s == "xx_yy_zz_xy_xz_yz")  return VirialOrdering::xx_yy_zz_xy_xz_yz;
    if (s == "xx_yy_zz_yz_xz_xy")  return VirialOrdering::xx_yy_zz_yz_xz_xy;
    err(ctx, "Unknown virial ordering '" + s + "'");
}

static SpeciesInputType parse_species_input(const std::string& s, const std::string& ctx) {
    if (s == "md_type")       return SpeciesInputType::MdType;
    if (s == "atomic_number") return SpeciesInputType::AtomicNumber;
    if (s == "atomic_symbol") return SpeciesInputType::AtomicSymbol;
    err(ctx, "Unknown species input type '" + s + "'");
}


static ShapeDim parse_shape_dim(const std::string& s, const std::string& ctx) {
    ShapeDim d;
    if (s == "nlocal")    { d.symbol = ShapeSymbol::Nlocal; }
    else if (s == "nghost")  { d.symbol = ShapeSymbol::Nghost; }
    else if (s == "nall")    { d.symbol = ShapeSymbol::Nall;   }
    else if (s == "nedges")  { d.symbol = ShapeSymbol::Nedges; }
    else {
        try {
            d.symbol = ShapeSymbol::Literal;
            d.value  = std::stoll(s);
        } catch (...) {
            err(ctx, "Unknown shape symbol '" + s +
                "'. Allowed: nlocal, nghost, nall, nedges, or an integer literal");
        }
    }
    return d;
}

static ShapeSpec parse_shape(const YAML::Node& n, const std::string& ctx) {
    ShapeSpec shape;
    if (!n.IsSequence())
        err(ctx + ".shape", "Expected a sequence, e.g. [nall, 3]");
    for (auto& el : n) {
        std::string s;
        if (el.IsScalar()) {
            try { s = std::to_string(el.as<int>()); }
            catch (...) { s = el.as<std::string>(); }
        } else {
            err(ctx + ".shape", "Shape elements must be scalars");
        }
        shape.push_back(parse_shape_dim(s, ctx + ".shape"));
    }
    return shape;
}


static BackendConfig parse_backend(const YAML::Node& model_n,
                                    const YAML::Node& backend_n)
{
    BackendConfig bc;
    bc.backend    = parse_backend_type(req_str(model_n, "backend", "model"), "model.backend");
    bc.format     = opt_str(backend_n, "format");
    bc.entrypoint = opt_str(backend_n, "entrypoint", "forward");
    bc.call_style = CallStyle::Positional;
    if (backend_n["call_style"])
        bc.call_style = parse_call_style(backend_n["call_style"].as<std::string>(), "backend.call_style");
    bc.share_cuda_stream  = opt_bool(backend_n, "share_cuda_stream", true);
    bc.allow_cpu_fallback = opt_bool(backend_n, "allow_cpu_fallback", false);
    if (backend_n["plugin_library"])
        bc.plugin_library = backend_n["plugin_library"].as<std::string>();
    if (backend_n["plugin_factory_symbol"])
        bc.plugin_factory_symbol = backend_n["plugin_factory_symbol"].as<std::string>();

    reject_unknown_keys(backend_n,
        {"format","entrypoint","call_style","share_cuda_stream","allow_cpu_fallback",
         "plugin_library","plugin_factory_symbol"},
        "backend");
    return bc;
}

static NeighborSpec parse_neighbor(const YAML::Node& n) {
    NeighborSpec s;
    if (!n) return s;
    s.cutoff           = opt_double(n, "cutoff", 6.0);
    s.skin             = opt_double(n, "skin", 0.5);
    if (n["list"])
        s.list         = parse_neighbor_list(n["list"].as<std::string>(), "neighbor.list");
    s.directed         = opt_bool(n, "directed", true);
    s.self_edges       = opt_bool(n, "self_edges", false);
    s.include_ghost    = opt_bool(n, "include_ghost_atoms", true);
    s.require_edge_vecs  = opt_bool(n, "require_edge_vectors", false);
    s.require_distances  = opt_bool(n, "require_distances", false);
    s.require_cell_shifts = opt_bool(n, "require_cell_shifts", false);
    reject_unknown_keys(n, {"cutoff","skin","list","directed","self_edges",
        "include_ghost_atoms","require_edge_vectors","require_distances",
        "require_cell_shifts"}, "neighbor");
    return s;
}

static SpeciesMapSpec parse_species_map(const YAML::Node& n) {
    SpeciesMapSpec sm;
    if (!n) return sm;
    sm.input_type  = parse_species_input(opt_str(n, "input", "md_type"), "species_map.input");
    if (n["values"] && n["values"].IsMap()) {
        for (auto it = n["values"].begin(); it != n["values"].end(); ++it) {
            int idx = it->second.as<int>();
            if (sm.input_type == SpeciesInputType::AtomicSymbol) {
                sm.symbol_to_index[it->first.as<std::string>()] = idx;
            } else {
                sm.int_to_index[it->first.as<int>()] = idx;
            }
        }
    }
    reject_unknown_keys(n, {"input","output","values"}, "species_map");
    return sm;
}

static UnitSystem parse_units(const YAML::Node& n) {
    UnitSystem u;
    if (!n) return u;
    u.length = opt_str(n, "length", "angstrom");
    u.energy = opt_str(n, "energy", "eV");
    return u;
}

static InputSpec parse_input_entry(const std::string& name, const YAML::Node& n) {
    InputSpec s;
    const std::string ctx = "inputs." + name;
    s.source         = req_str(n, "source", ctx);
    s.tensor_name    = opt_str(n, "tensor_name", name);
    s.argument_index = opt_int(n, "argument_index", -1);
    if (n["shape"])
        s.shape      = parse_shape(n["shape"], ctx);
    if (n["dtype"])
        s.dtype      = parse_dtype(n["dtype"].as<std::string>(), ctx + ".dtype");
    s.requires_grad  = opt_bool(n, "requires_grad", false);
    if (n["transform"] && n["transform"].IsSequence()) {
        for (auto& t : n["transform"])
            s.transforms.push_back(t.as<std::string>());
    }
    reject_unknown_keys(n, {"source","tensor_name","argument_index","shape",
        "dtype","requires_grad","transform"}, ctx);
    return s;
}

static OutputSpec parse_output_entry(const std::string& name, const YAML::Node& n) {
    OutputSpec s;
    const std::string ctx = "outputs." + name;
    s.tensor_name    = opt_str(n, "tensor_name", name);
    s.output_index   = opt_int(n, "output_index", -1);
    s.target         = req_str(n, "target", ctx);
    if (n["dtype"])
        s.expected_dtype = parse_dtype(n["dtype"].as<std::string>(), ctx + ".dtype");
    if (n["shape"])
        s.expected_shape = parse_shape(n["shape"], ctx);
    s.sign           = opt_double(n, "sign", 1.0);
    s.scale          = opt_double(n, "scale", 1.0);
    if (n["convention"])
        s.force_convention = parse_force_convention(n["convention"].as<std::string>(), ctx + ".convention");
    if (n["virial_convention"])
        s.virial_convention = parse_virial_convention(n["virial_convention"].as<std::string>(), ctx);
    if (n["ordering"])
        s.virial_ordering = parse_virial_ordering(n["ordering"].as<std::string>(), ctx);
    s.includes_ghost = opt_bool(n, "includes_ghost_atoms", false);
    if (n["operations"] && n["operations"].IsSequence()) {
        for (auto& op : n["operations"])
            s.operations.push_back(op.as<std::string>());
    }
    if (n["operation"])
        s.operations.push_back(n["operation"].as<std::string>());

    reject_unknown_keys(n, {"tensor_name","output_index","target","dtype","shape",
        "sign","scale","convention","virial_convention","ordering",
        "includes_ghost_atoms","operations","operation","reduction"}, ctx);
    return s;
}

static AutogradSpec parse_autograd(const YAML::Node& n) {
    AutogradSpec a;
    if (!n) return a;
    a.enabled           = opt_bool(n, "enabled", false);
    a.energy_output     = opt_str(n, "energy_output");
    a.differentiate_wrt = opt_str(n, "differentiate_wrt", "positions");
    a.force_sign        = opt_double(n, "force_sign", -1.0);
    a.create_graph      = opt_bool(n, "create_graph", false);
    a.retain_graph      = opt_bool(n, "retain_graph", false);
    reject_unknown_keys(n, {"enabled","energy_output","differentiate_wrt",
        "force_sign","create_graph","retain_graph"}, "autograd");
    return a;
}

static void parse_capabilities(const YAML::Node& n, MlipModelSpec& spec) {
    if (!n) return;
    spec.cap_energy   = opt_bool(n, "energy", false);
    spec.cap_forces   = opt_bool(n, "forces", false);
    spec.cap_virial   = opt_bool(n, "virial", false);
    spec.cap_stress   = opt_bool(n, "stress", false);
    spec.cap_external_neighbor_list = opt_bool(n, "external_neighbor_list", true);
    spec.cap_external_cuda_stream   = opt_bool(n, "external_cuda_stream", true);
    reject_unknown_keys(n, {"energy","forces","virial","stress",
        "external_neighbor_list","external_cuda_stream"}, "capabilities");
}


static MlipModelSpec parse_root(const YAML::Node& root, const std::string& source_name) {
    if (!root.IsMap())
        err(source_name, "Root YAML node must be a mapping");

    if (!root["schema_version"])
        err(source_name, "Missing required key 'schema_version'");
    int sv = root["schema_version"].as<int>();
    if (sv != 1)
        err(source_name, "Unsupported schema_version " + std::to_string(sv) + ". Only version 1 is supported.");

    reject_unknown_keys(root,
        {"schema_version","model","backend","inputs","outputs",
         "neighbor","species_map","units","autograd","capabilities"},
        source_name);

    MlipModelSpec spec;
    spec.schema_version = sv;

    {
        auto& m = root["model"];
        if (!m) err(source_name, "Missing required section 'model'");
        reject_unknown_keys(m,
            {"name","backend","path","device","precision"},
            "model");
        spec.name      = opt_str(m, "name", "unnamed");
        spec.model_path = req_str(m, "path", "model");
        spec.device    = opt_str(m, "device", "cuda");
        spec.precision = opt_str(m, "precision", "float32");
        if (spec.precision != "float32" && spec.precision != "float64")
            err("model.precision", "Must be 'float32' or 'float64'");
    }

    spec.backend = parse_backend(root["model"],
                                  root["backend"] ? root["backend"] : YAML::Node(YAML::NodeType::Map));

    spec.neighbor = parse_neighbor(root["neighbor"]);

    spec.species_map = parse_species_map(root["species_map"]);

    if (root["units"]) {
        auto& u = root["units"];
        reject_unknown_keys(u, {"md","model"}, "units");
        if (u["md"])    spec.md_units    = parse_units(u["md"]);
        if (u["model"]) spec.model_units = parse_units(u["model"]);
    }

    spec.autograd = parse_autograd(root["autograd"]);

    if (root["inputs"] && root["inputs"].IsMap()) {
        for (auto it = root["inputs"].begin(); it != root["inputs"].end(); ++it) {
            std::string name = it->first.as<std::string>();
            spec.inputs[name] = parse_input_entry(name, it->second);
        }
    }

    if (root["outputs"] && root["outputs"].IsMap()) {
        for (auto it = root["outputs"].begin(); it != root["outputs"].end(); ++it) {
            std::string name = it->first.as<std::string>();
            spec.outputs[name] = parse_output_entry(name, it->second);
        }
    }

    parse_capabilities(root["capabilities"], spec);

    if (spec.cap_forces && spec.inputs.empty())
        err(source_name, "Model declares forces capability but has no inputs defined");

    return spec;
}

} // anonymous namespace


MlipModelSpec parse_model_spec(const std::string& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("[MLIP YAML] Failed to parse '" + yaml_path + "': " + e.what());
    }
    return parse_root(root, yaml_path);
}

MlipModelSpec parse_model_spec_from_string(const std::string& yaml_text) {
    YAML::Node root;
    try {
        root = YAML::Load(yaml_text);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("[MLIP YAML] Parse error: " + std::string(e.what()));
    }
    return parse_root(root, "<string>");
}

} // namespace mlip
