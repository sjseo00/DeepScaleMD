#include "OpLibLoader.h"

#include <dlfcn.h>

#include <filesystem>
#include <set>
#include <stdexcept>

void load_op_libs(const std::vector<std::string>& paths) {
    static std::set<std::string> loaded;   // handles are never closed: the operators stay registered

    for (const std::string& path : paths) {
        if (loaded.count(path)) continue;

        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            throw std::runtime_error("OpLibLoader: op library not found: " + path +
                                     " (from required_op_libs in metadata.json; was the model converted on "
                                     "another machine?)");
        }
        dlerror();   // clear any stale error
        if (!dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
            const char* reason = dlerror();
            throw std::runtime_error("OpLibLoader: cannot load op library '" + path + "': " +
                                     (reason ? reason : "unknown dlopen error"));
        }
        loaded.insert(path);
    }
}
