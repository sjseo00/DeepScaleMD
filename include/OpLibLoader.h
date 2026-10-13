#ifndef OP_LIB_LOADER_H
#define OP_LIB_LOADER_H

// Custom operator libraries (cuEquivariance, OpenEquivariance kernels) must be loaded before the model: a saved
// TorchScript/AOTI file only names the operators (design 6.1). No LibTorch dependency.

#include <string>
#include <vector>

// dlopen(RTLD_NOW | RTLD_GLOBAL) every library, in order (design 6.4). A library stays loaded for the life of the
// process and loading one twice is harmless. Throws std::runtime_error naming the path (and the dlerror text) for a
// missing or unloadable library.
void load_op_libs(const std::vector<std::string>& paths);

#endif // OP_LIB_LOADER_H
