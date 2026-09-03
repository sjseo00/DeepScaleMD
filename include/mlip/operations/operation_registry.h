/**
 * operation_registry.h
 *
 * Registry for preprocessing and postprocessing GPU operations.
 * Each operation is identified by a string name that matches YAML "transform"
 * and "operations" entries.
 *
 * Operations are resolved once at initialization into PreprocessOperation /
 * PostprocessOperation objects and cached in the execution plan.
 * The hot path uses direct function calls — no string lookup.
 */

#pragma once
#ifndef MLIP_OPERATION_REGISTRY_H
#define MLIP_OPERATION_REGISTRY_H

#include "../mlip_types.h"
#include "../backend/backend_interface.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlip {

// ─────────────── context passed to every operation at execution time ──────────

struct OpContext {
    int          num_local  = 0;
    int          num_ghost  = 0;
    int          num_total  = 0;
    int64_t      num_edges  = 0;
    int          device_id  = 0;
    cudaStream_t stream     = nullptr;
};

// ─────────────── preprocessing operation ─────────────────────────────────────

struct PreprocessOperation {
    std::string name;

    // Called once at plan compile time to verify dtypes, shapes, allocate scratch.
    std::function<void(const DeviceArrayView& in, DeviceArrayView& out,
                       const OpContext& ctx)> prepare;

    // Called every MD step on the GPU; must NOT allocate.
    std::function<void(const DeviceArrayView& in, DeviceArrayView& out,
                       BackendTensorView& tv, const OpContext& ctx)> execute;

    bool   in_place           = false;
    bool   supports_cuda      = true;
    bool   graph_compatible   = true;
    mlip::ScalarType output_dtype   = mlip::ScalarType::Float32;
};

// ─────────────── postprocessing operation ────────────────────────────────────

struct PostprocessOperation {
    std::string name;

    std::function<void(const BackendTensorView& model_out,
                       DeviceArrayView& target,
                       const OpContext& ctx)> execute;

    bool   supports_cuda    = true;
    bool   graph_compatible = true;
};

// ─────────────── registries ──────────────────────────────────────────────────

class PreprocessRegistry {
public:
    static PreprocessRegistry& instance();

    void register_op(const std::string& name, PreprocessOperation op) {
        if (ops_.count(name))
            throw std::runtime_error("[PreprocessRegistry] Duplicate: " + name);
        ops_[name] = std::move(op);
    }

    const PreprocessOperation& get(const std::string& name) const {
        auto it = ops_.find(name);
        if (it == ops_.end())
            throw std::runtime_error(
                "[PreprocessRegistry] Unknown preprocessing op '" + name +
                "'. Check YAML 'transform' fields.");
        return it->second;
    }

    bool has(const std::string& n) const { return ops_.count(n) > 0; }

    std::string list() const {
        std::string s;
        for (auto& [k, _] : ops_) { if (!s.empty()) s += ", "; s += k; }
        return s;
    }

private:
    PreprocessRegistry() = default;
    std::unordered_map<std::string, PreprocessOperation> ops_;
};

class PostprocessRegistry {
public:
    static PostprocessRegistry& instance();

    void register_op(const std::string& name, PostprocessOperation op) {
        if (ops_.count(name))
            throw std::runtime_error("[PostprocessRegistry] Duplicate: " + name);
        ops_[name] = std::move(op);
    }

    const PostprocessOperation& get(const std::string& name) const {
        auto it = ops_.find(name);
        if (it == ops_.end())
            throw std::runtime_error(
                "[PostprocessRegistry] Unknown postprocessing op '" + name +
                "'. Check YAML 'operations' fields.");
        return it->second;
    }

    bool has(const std::string& n) const { return ops_.count(n) > 0; }

private:
    PostprocessRegistry() = default;
    std::unordered_map<std::string, PostprocessOperation> ops_;
};

// ─────────────── source / target registries ──────────────────────────────────

/// Resolves MD source names (e.g. "md.positions") to DeviceArrayView at runtime.
using PointerResolver   = std::function<void*(const OpContext&)>;
using ShapeResolverFn   = std::function<ShapeSpec(const OpContext&)>;

struct DataSourceDescriptor {
    mlip::ScalarType       dtype;
    int              rank;          // number of dimensions
    ShapeResolverFn  shape_resolver;
    PointerResolver  pointer_resolver;
};

class SourceRegistry {
public:
    static SourceRegistry& instance();

    void register_source(const std::string& name, DataSourceDescriptor d) {
        if (sources_.count(name))
            throw std::runtime_error("[SourceRegistry] Duplicate: " + name);
        sources_[name] = std::move(d);
    }

    const DataSourceDescriptor& get(const std::string& name) const {
        auto it = sources_.find(name);
        if (it == sources_.end())
            throw std::runtime_error(
                "[SourceRegistry] Unknown source '" + name + "'. "
                "Allowed: " + list());
        return it->second;
    }

    bool has(const std::string& n) const { return sources_.count(n) > 0; }

    std::string list() const {
        std::string s;
        for (auto& [k,_] : sources_) { if (!s.empty()) s += ", "; s += k; }
        return s;
    }

private:
    SourceRegistry() = default;
    std::unordered_map<std::string, DataSourceDescriptor> sources_;
};

class TargetRegistry {
public:
    static TargetRegistry& instance();

    void register_target(const std::string& name, PointerResolver r) {
        if (targets_.count(name))
            throw std::runtime_error("[TargetRegistry] Duplicate: " + name);
        targets_[name] = std::move(r);
    }

    PointerResolver get(const std::string& name) const {
        auto it = targets_.find(name);
        if (it == targets_.end())
            throw std::runtime_error("[TargetRegistry] Unknown target '" + name + "'");
        return it->second;
    }

    bool has(const std::string& n) const { return targets_.count(n) > 0; }

private:
    TargetRegistry() = default;
    std::unordered_map<std::string, PointerResolver> targets_;
};

// ─────────────── helper macros ───────────────────────────────────────────────

#define REGISTER_MLIP_PREPROCESS_OP(name, ...)                                 \
    static int _mlip_preop_##__COUNTER__ = [](){                                \
        mlip::PreprocessRegistry::instance().register_op(name, __VA_ARGS__);   \
        return 0;                                                                \
    }()

#define REGISTER_MLIP_POSTPROCESS_OP(name, ...)                                \
    static int _mlip_postop_##__COUNTER__ = [](){                              \
        mlip::PostprocessRegistry::instance().register_op(name, __VA_ARGS__);  \
        return 0;                                                               \
    }()

} // namespace mlip
#endif // MLIP_OPERATION_REGISTRY_H
