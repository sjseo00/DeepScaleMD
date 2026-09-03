/**
 * mlip_registry.h
 *
 * Factory/registry for IMlipPotential implementations.
 * New adapters register themselves via REGISTER_MLIP() without touching MD core.
 */

#ifndef MLIP_REGISTRY_H
#define MLIP_REGISTRY_H

#include "mlip_interface.h"
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

class MlipRegistry {
public:
    using Creator = std::function<std::unique_ptr<IMlipPotential>()>;

    // Singleton access — each process has exactly one registry.
    static MlipRegistry& instance() {
        static MlipRegistry reg;
        return reg;
    }

    void register_model(const std::string& name, Creator creator) {
        if (creators_.count(name))
            throw std::runtime_error("[MlipRegistry] Duplicate registration: " + name);
        creators_[name] = std::move(creator);
    }

    std::unique_ptr<IMlipPotential> create(const std::string& name) const {
        auto it = creators_.find(name);
        if (it == creators_.end())
            throw std::runtime_error("[MlipRegistry] Unknown MLIP adapter: '" + name +
                                     "'. Check that the adapter is compiled in.");
        return it->second();
    }

    bool has(const std::string& name) const {
        return creators_.count(name) > 0;
    }

private:
    MlipRegistry() = default;
    std::unordered_map<std::string, Creator> creators_;
};

// ---------------------------------------------------------------------------
// Registration helpers
// ---------------------------------------------------------------------------

/// Helper used by REGISTER_MLIP to run registration at static-init time.
struct MlipAutoRegister {
    MlipAutoRegister(const std::string& name, MlipRegistry::Creator creator) {
        MlipRegistry::instance().register_model(name, std::move(creator));
    }
};

/**
 * Place this macro in the .cpp file of each adapter:
 *
 *   REGISTER_MLIP("mace",   MaceAdapter)
 *   REGISTER_MLIP("nequip", NequipAdapter)
 */
#define REGISTER_MLIP(name, ClassName)                                       \
    static MlipAutoRegister _mlip_autoreg_##ClassName(                       \
        name,                                                                 \
        []() -> std::unique_ptr<IMlipPotential> {                            \
            return std::make_unique<ClassName>();                             \
        })

#endif // MLIP_REGISTRY_H
