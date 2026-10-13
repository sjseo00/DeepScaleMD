#include "MLIPMetadata.h"

#include "MiniJson.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

// ---- Field access and validation ------------------------------------------------------------------------

class MetadataReader {
public:
    MetadataReader(const JsonValue& root, const std::string& source) : root_(root), source_(source) {}

    [[noreturn]] void fail(const std::string& what) const { throw std::runtime_error(source_ + ": " + what); }

    const JsonValue& require(const JsonValue& object, const std::string& key, JsonValue::Type type,
                             const std::string& path) const {
        const JsonValue* value = object.find(key);
        if (!value) fail("missing required field '" + path + "'");
        check_type(*value, type, path);
        return *value;
    }

    const JsonValue* optional(const JsonValue& object, const std::string& key, JsonValue::Type type,
                              const std::string& path) const {
        const JsonValue* value = object.find(key);
        if (!value || value->type == JsonValue::Type::Null) return nullptr;
        check_type(*value, type, path);
        return value;
    }

    void check_type(const JsonValue& value, JsonValue::Type type, const std::string& path) const {
        if (value.type != type) {
            fail("field '" + path + "' must be " + json_type_name(type) + ", not " + json_type_name(value.type));
        }
    }

    std::string string_field(const JsonValue& object, const std::string& key, const std::string& path) const {
        const std::string& s = require(object, key, JsonValue::Type::String, path).string;
        if (s.empty()) fail("field '" + path + "' must not be empty");
        return s;
    }

    std::string choice_field(const JsonValue& object, const std::string& key, const std::string& path,
                             const std::vector<std::string>& allowed) const {
        const std::string s = string_field(object, key, path);
        if (std::find(allowed.begin(), allowed.end(), s) == allowed.end()) {
            std::string list;
            for (const std::string& a : allowed) list += (list.empty() ? "" : ", ") + a;
            fail("field '" + path + "' is '" + s + "'; supported values are: " + list);
        }
        return s;
    }

    double positive_number(const JsonValue& object, const std::string& key, const std::string& path) const {
        const double x = require(object, key, JsonValue::Type::Number, path).number;
        if (!std::isfinite(x) || x <= 0.0) fail("field '" + path + "' must be a number greater than 0");
        return x;
    }

    int integer_in_range(const JsonValue& value, const std::string& path, int low, int high) const {
        if (value.type != JsonValue::Type::Number) {
            fail("field '" + path + "' must be an integer, not " + json_type_name(value.type));
        }
        const double x = value.number;
        if (!std::isfinite(x) || x != std::floor(x) || x < low || x > high) {
            fail("field '" + path + "' must be an integer between " + std::to_string(low) + " and " +
                 std::to_string(high));
        }
        return static_cast<int>(x);
    }

    const JsonValue& root() const { return root_; }

private:
    const JsonValue& root_;
    const std::string& source_;
};

}  // namespace

MLIPMetadata parse_mlip_metadata(const std::string& json_text, const std::string& source) {
    const JsonValue root = parse_json(json_text, source);
    const MetadataReader r(root, source);
    if (root.type != JsonValue::Type::Object) r.fail("the top level must be a JSON object");

    MLIPMetadata m;

    m.format_version = r.integer_in_range(
        r.require(root, "format_version", JsonValue::Type::Number, "format_version"), "format_version", 0, 1000000);
    if (m.format_version != 1) {
        r.fail("unsupported format_version " + std::to_string(m.format_version) + " (this build reads 1)");
    }

    m.backend = r.choice_field(root, "backend", "backend", {"torchscript", "aoti"});
    m.model_family = r.string_field(root, "model_family", "model_family");
    m.model_name = r.string_field(root, "model_name", "model_name");

    m.cutoff = r.positive_number(root, "cutoff", "cutoff");
    m.num_layers = r.integer_in_range(
        r.require(root, "num_layers", JsonValue::Type::Number, "num_layers"), "num_layers", 1, 1000);
    m.ghost_cutoff = r.positive_number(root, "ghost_cutoff", "ghost_cutoff");
    // Message passing over num_layers layers needs ghosts that deep; less gives silently wrong forces.
    const double required_depth = m.num_layers * m.cutoff;
    if (m.ghost_cutoff < required_depth * (1.0 - 1e-12)) {
        r.fail("field 'ghost_cutoff' (" + std::to_string(m.ghost_cutoff) + ") must be at least num_layers * cutoff (" +
               std::to_string(required_depth) + ")");
    }

    const JsonValue& elements = r.require(root, "elements", JsonValue::Type::Array, "elements");
    if (elements.array.empty()) r.fail("field 'elements' must not be empty");
    std::set<int> seen;
    for (size_t i = 0; i < elements.array.size(); ++i) {
        const int z = r.integer_in_range(elements.array[i], "elements[" + std::to_string(i) + "]", 1, 118);
        if (!seen.insert(z).second) r.fail("field 'elements' lists atomic number " + std::to_string(z) + " twice");
        m.elements.push_back(z);
    }

    m.dtype = r.choice_field(root, "dtype", "dtype", {"float32", "float64"});

    const JsonValue& units = r.require(root, "units", JsonValue::Type::Object, "units");
    m.energy_unit = r.choice_field(units, "energy", "units.energy", {"eV", "kcal/mol"});
    m.length_unit = r.choice_field(units, "length", "units.length", {"Angstrom"});

    if (const JsonValue* head = r.optional(root, "head", JsonValue::Type::String, "head")) m.head = head->string;

    m.accel = r.choice_field(root, "accel", "accel", {"e3nn", "cueq", "openeq"});

    if (const JsonValue* libs = r.optional(root, "required_op_libs", JsonValue::Type::Array, "required_op_libs")) {
        for (size_t i = 0; i < libs->array.size(); ++i) {
            const std::string path = "required_op_libs[" + std::to_string(i) + "]";
            r.check_type(libs->array[i], JsonValue::Type::String, path);
            const std::string& lib = libs->array[i].string;
            if (lib.empty() || lib[0] != '/') r.fail("field '" + path + "' must be an absolute path, got '" + lib + "'");
            m.required_op_libs.push_back(lib);
        }
    }
    if (m.accel != "e3nn" && m.required_op_libs.empty()) {
        r.fail("accel '" + m.accel + "' needs its kernel library: field 'required_op_libs' must not be empty");
    }

    if (const JsonValue* versions = r.optional(root, "versions", JsonValue::Type::Object, "versions")) {
        for (size_t i = 0; i < versions->keys.size(); ++i) {
            const JsonValue& v = versions->values[i];
            if (v.type == JsonValue::Type::Null) continue;  // package not installed at conversion time
            r.check_type(v, JsonValue::Type::String, "versions." + versions->keys[i]);
            m.versions[versions->keys[i]] = v.string;
        }
    }

    if (const JsonValue* gpu = r.optional(root, "target_gpu", JsonValue::Type::String, "target_gpu")) {
        const std::string& s = gpu->string;
        const bool shape_ok = s.size() >= 4 && s.compare(0, 3, "sm_") == 0 &&
                              std::all_of(s.begin() + 3, s.end(), [](char c) { return c >= '0' && c <= '9'; });
        if (!shape_ok) r.fail("field 'target_gpu' must look like 'sm_90', got '" + s + "'");
        m.target_gpu = s;
    }
    if (m.backend == "aoti" && m.target_gpu.empty()) {
        r.fail("backend 'aoti' needs field 'target_gpu' (the compiled GPU architecture, e.g. 'sm_90')");
    }

    return m;
}

MLIPMetadata load_mlip_metadata(const std::string& path) {
    std::filesystem::path file(path);
    std::error_code ec;
    if (std::filesystem::is_directory(file, ec)) file /= "metadata.json";

    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open model metadata file: " + file.string());
    std::stringstream buffer;
    buffer << in.rdbuf();
    return parse_mlip_metadata(buffer.str(), file.string());
}
