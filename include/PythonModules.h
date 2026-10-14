#pragma once

#include <string>
#include <vector>

// Puts the site-packages of the active Python environment ($VIRTUAL_ENV, $CONDA_PREFIX) at the front of sys.path of the
// embedded interpreter, so that the packages of the environment the user activated (torch, the model packages) are found without
// PYTHONPATH. Does nothing when neither variable is set or the environment has no site-packages for this Python version.
// The caller holds the GIL. Both embedded-Python users (PythonBackend and import_python_modules) call it right after starting
// the interpreter.
void add_active_python_environment();

// Imports Python modules into the interpreter embedded in this process, starting the interpreter if none runs yet.
//
// mlip_backend: libtorch uses it for a model whose operators are registered from Python (cuEquivariance: metadata
// required_python_modules): the operators exist in the dispatcher only after their module has been imported, and a
// TorchScript or AOTI model that calls them cannot be loaded before. Nothing else runs in Python: the model itself is
// still executed by LibTorch.
//
// `torch` is imported first and its __version__ is returned (for example "2.7.1+cu118"), so that the caller can compare
// it with the LibTorch the program is linked against: the embedded Python must find the same torch build, in the activated
// environment (add_active_python_environment) or through PYTHONPATH.
//
// After the call the main thread no longer holds the GIL. An operator implemented in Python takes it itself, on whatever
// thread runs it: the calling thread, or an autograd engine thread that computes the backward pass of a TorchScript model.
// (A GIL held by the main thread while it waits for the engine would deadlock that.)
//
// Throws std::runtime_error with the Python error text when an import fails, and when this build has no embedded Python
// (configure with -DENABLE_PYTHON=ON).
std::string import_python_modules(const std::vector<std::string>& modules);
