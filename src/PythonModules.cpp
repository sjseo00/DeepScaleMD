#include "PythonModules.h"

#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_PYTHON_INTERFACE
#include <Python.h>
#endif

namespace {

#ifdef ENABLE_PYTHON_INTERFACE

// Owns one reference.
struct PyRef {
    PyObject* object;
    explicit PyRef(PyObject* o = nullptr) : object(o) {}
    ~PyRef() { Py_XDECREF(object); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
};

// "ExceptionType: message" of the pending Python error, which is printed (with its traceback) and cleared.
std::string take_python_error() {
    if (!PyErr_Occurred()) return "unknown error";
    PyObject *type = nullptr, *value = nullptr, *trace = nullptr;
    PyErr_Fetch(&type, &value, &trace);
    PyErr_NormalizeException(&type, &value, &trace);
    std::string text;
    if (type) {
        PyRef name(PyObject_GetAttrString(type, "__name__"));
        if (name.object && PyUnicode_Check(name.object)) text = PyUnicode_AsUTF8(name.object);
        PyErr_Clear();
    }
    if (value) {
        PyRef message(PyObject_Str(value));
        if (message.object && PyUnicode_Check(message.object)) text += (text.empty() ? "" : ": ") + std::string(PyUnicode_AsUTF8(message.object));
        PyErr_Clear();
    }
    PyErr_Restore(type, value, trace);
    PyErr_Print();   // the traceback goes to stderr; this also clears the error
    return text.empty() ? "unknown error" : text;
}

// Holds the GIL of the calling thread for its lifetime; if this call started the interpreter, the main thread's hold on the
// GIL (which Py_Initialize leaves taken) is given up at the end, for good.
struct GilScope {
    bool started_here = false;
    PyGILState_STATE state;
    GilScope() {
        if (!Py_IsInitialized()) {
            Py_InitializeEx(0);   // 0: no Python signal handlers; MPI and the CUDA runtime keep theirs
            started_here = true;
        }
        state = PyGILState_Ensure();
    }
    ~GilScope() {
        PyGILState_Release(state);
        if (started_here) PyEval_SaveThread();
    }
};

// A module can import cleanly and still not have registered what the model needs. OpenEquivariance registers the backward
// pass of its operators (torch.library.register_autograd) only when its torch extension was built and loaded, and it keeps
// the build error to itself: the import succeeds, the energies are right and every force is silently wrong ("an autograd
// kernel was not registered ... silently incorrect behavior"). Returns what is missing, or "" when all is in place.
std::string openequivariance_problem() {
    static const char* const code =
        "import torch\n"
        "problem = ''\n"
        "try:\n"
        "    from openequivariance._torch import extlib as _extlib\n"
        "    _built = getattr(_extlib, 'BUILT_EXTENSION', True)\n"
        "    _error = getattr(_extlib, 'BUILT_EXTENSION_ERROR', None)\n"
        "except Exception:\n"
        "    _built, _error = True, None\n"
        "if not _built:\n"
        "    problem = 'its torch extension was not built or loaded (%s)' % (_error,)\n"
        "else:\n"
        "    try:\n"
        "        _ok = torch._C._dispatch_has_kernel_for_dispatch_key('libtorch_tp_jit::jit_conv_forward', 'Autograd')\n"
        "    except Exception as _e:\n"
        "        problem = 'its operator libtorch_tp_jit::jit_conv_forward does not exist (%s)' % (_e,)\n"
        "    else:\n"
        "        if not _ok:\n"
        "            problem = 'the backward pass of its operator libtorch_tp_jit::jit_conv_forward is not registered'\n";
    PyRef globals(PyDict_New());
    if (!globals.object) throw std::runtime_error("cannot create a Python dictionary: " + take_python_error());
    PyDict_SetItemString(globals.object, "__builtins__", PyEval_GetBuiltins());
    PyRef ran(PyRun_String(code, Py_file_input, globals.object, globals.object));
    if (!ran.object) throw std::runtime_error("the OpenEquivariance check failed to run: " + take_python_error());
    PyObject* problem = PyDict_GetItemString(globals.object, "problem");   // borrowed
    return (problem && PyUnicode_Check(problem)) ? PyUnicode_AsUTF8(problem) : "";
}

#endif

}  // namespace

void add_active_python_environment() {
#ifdef ENABLE_PYTHON_INTERFACE
    PyRun_SimpleString(
        "import os as _os, sys as _sys\n"
        "for _root in (_os.environ.get('VIRTUAL_ENV'), _os.environ.get('CONDA_PREFIX')):\n"
        "    if _root:\n"
        "        _site = _os.path.join(_root, 'lib', 'python%d.%d' % _sys.version_info[:2], 'site-packages')\n"
        "        if _os.path.isdir(_site) and _site not in _sys.path:\n"
        "            _sys.path.insert(0, _site)\n");
#endif
}

std::string import_python_modules(const std::vector<std::string>& modules) {
#ifndef ENABLE_PYTHON_INTERFACE
    std::string names;
    for (const std::string& name : modules) names += (names.empty() ? "" : ", ") + name;
    throw std::runtime_error(
        "this model needs the Python modules " + names + " imported before it can be loaded (they register "
        "operators of the model from Python), and this build has no embedded Python: configure dsMD with -DENABLE_PYTHON=ON "
        "(Python 3 development files and NumPy), or use the e3nn version of the model");
#else
    std::string torch_version;
    GilScope gil;
    if (!Py_IsInitialized()) throw std::runtime_error("cannot start the embedded Python interpreter");
    add_active_python_environment();

    std::vector<std::string> order = {"torch"};   // first: the version, and the libraries every other module needs
    order.insert(order.end(), modules.begin(), modules.end());
    for (const std::string& name : order) {
        PyRef module(PyImport_ImportModule(name.c_str()));
        if (!module.object) {
            throw std::runtime_error(
                "cannot import the Python module '" + name + "' in the embedded interpreter: " + take_python_error() +
                " (the embedded Python finds packages in the activated environment ($VIRTUAL_ENV, $CONDA_PREFIX) and in "
                "PYTHONPATH: activate the environment the model was converted in; its torch must be the LibTorch this program "
                "is linked with)");
        }
        if (name == "torch") {
            PyRef version(PyObject_GetAttrString(module.object, "__version__"));
            if (version.object && PyUnicode_Check(version.object)) torch_version = PyUnicode_AsUTF8(version.object);
            PyErr_Clear();
        }
        if (name == "openequivariance") {
            const std::string problem = openequivariance_problem();
            if (!problem.empty()) {
                throw std::runtime_error(
                    "openequivariance was imported, but " + problem + ". Run with it and the energies are right while every "
                    "force is wrong. Check on the compute node, in the activated environment: python -c 'import "
                    "openequivariance; from openequivariance._torch import extlib; print(extlib.BUILT_EXTENSION, "
                    "extlib.BUILT_EXTENSION_ERROR)' (the extension is compiled when it is not cached for this node: a C++ compiler "
                    "(gcc 9 or newer), ninja and nvcc must be on PATH in the job; a stale lock file in ~/.cache/torch_extensions/ "
                    "also stops the build)");
            }
        }
    }
    return torch_version;
#endif
}
