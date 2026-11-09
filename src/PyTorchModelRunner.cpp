#ifdef ENABLE_PYTHON_INTERFACE
#include <Python.h>   // first, as Python asks
#endif

#include "PyTorchModelRunner.h"

#include <ATen/DLConvertor.h>

#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_PYTHON_INTERFACE

namespace {

// Owns one reference.
struct PyRef {
    PyObject* object;
    explicit PyRef(PyObject* o = nullptr) : object(o) {}
    ~PyRef() { Py_XDECREF(object); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
    PyObject* release() {
        PyObject* o = object;
        object = nullptr;
        return o;
    }
};

// Holds the GIL of the calling thread for its lifetime.
struct GilHold {
    PyGILState_STATE state;
    GilHold() : state(PyGILState_Ensure()) {}
    ~GilHold() { PyGILState_Release(state); }
    GilHold(const GilHold&) = delete;
    GilHold& operator=(const GilHold&) = delete;
};

// "ExceptionType: message" of the pending Python error; the traceback is printed to stderr and the error cleared.
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
    PyErr_Print();
    return text.empty() ? "unknown error" : text;
}

// A capsule that torch.from_dlpack took is renamed "used_dltensor" and owned by that tensor; one that was never taken frees its
// DLPack wrapper (and with it the reference to the C++ tensor) here.
void free_unused_capsule(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
        if (managed && managed->deleter) managed->deleter(managed);
    }
}

// The tensor as a DLPack capsule (shares its memory); the caller holds the GIL.
PyObject* to_capsule(const torch::Tensor& tensor) {
    DLManagedTensor* managed = at::toDLPack(tensor);
    PyObject* capsule = PyCapsule_New(managed, "dltensor", free_unused_capsule);
    if (!capsule) {
        managed->deleter(managed);
        throw std::runtime_error("PyTorchModelRunner: cannot wrap a tensor for Python: " + take_python_error());
    }
    return capsule;
}

// The tensor of a DLPack capsule made by torch.utils.dlpack.to_dlpack; the tensor owns the memory from now on.
torch::Tensor from_capsule(PyObject* capsule, const std::string& model_folder) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        throw std::runtime_error("PyTorchModelRunner: the Python model of '" + model_folder + "' did not return DLPack capsules "
                                 "(mlip_convert/pytorch_runner.py PyTorchModel.run)");
    }
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    torch::Tensor tensor = at::fromDLPack(managed);
    PyCapsule_SetName(capsule, "used_dltensor");
    return tensor;
}

}  // namespace

PyTorchModelRunner::PyTorchModelRunner(const std::string& model_folder, const std::string& python_path, const torch::Device& device)
    : model_folder_(model_folder) {
    if (!Py_IsInitialized()) {
        throw std::runtime_error("PyTorchModelRunner: the embedded Python interpreter is not running (LibTorchBackend starts it "
                                 "before it loads a --backend pytorch model)");
    }
    GilHold gil;
    PyRef sys(PyImport_ImportModule("sys"));
    PyRef path(sys.object ? PyObject_GetAttrString(sys.object, "path") : nullptr);
    PyRef entry(PyUnicode_FromString(python_path.c_str()));
    if (!path.object || !entry.object || !PyList_Check(path.object)) {
        throw std::runtime_error("PyTorchModelRunner: cannot reach sys.path of the embedded Python: " + take_python_error());
    }
    if (PySequence_Contains(path.object, entry.object) == 0 && PyList_Insert(path.object, 0, entry.object) != 0) {
        throw std::runtime_error("PyTorchModelRunner: cannot extend sys.path: " + take_python_error());
    }

    PyRef module(PyImport_ImportModule("mlip_convert.pytorch_runner"));
    if (!module.object) {
        throw std::runtime_error("PyTorchModelRunner: cannot import mlip_convert.pytorch_runner from '" + python_path + "' (metadata "
                                 "python_path of '" + model_folder + "'): " + take_python_error());
    }
    PyRef load(PyObject_GetAttrString(module.object, "load"));
    if (!load.object) throw std::runtime_error("PyTorchModelRunner: mlip_convert.pytorch_runner has no load(): " + take_python_error());
    PyRef model(PyObject_CallFunction(load.object, "si", model_folder.c_str(), static_cast<int>(device.index())));
    if (!model.object) {
        throw std::runtime_error("PyTorchModelRunner: building the model of '" + model_folder + "' in Python failed: " +
                                 take_python_error());
    }
    PyRef run(PyObject_GetAttrString(model.object, "run"));
    if (!run.object) throw std::runtime_error("PyTorchModelRunner: the Python model has no run(): " + take_python_error());
    run_method_ = run.release();
}

PyTorchModelRunner::~PyTorchModelRunner() {
    if (run_method_ && Py_IsInitialized()) {
        GilHold gil;
        Py_DECREF(static_cast<PyObject*>(run_method_));
    }
}

std::vector<torch::Tensor> PyTorchModelRunner::run(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() != 4 && inputs.size() != 5) {
        throw std::invalid_argument("PyTorchModelRunner::run expects {pos, atomic_numbers, edge_index, n_owned} "
                                    "or {pos, atomic_numbers, edge_index, edge_shift, n_owned}");
    }
    const size_t expected_outputs = inputs.size() == 5 ? 4 : 3;
    const long long n_owned = inputs.back().item<int64_t>();   // a CPU scalar: no device synchronization

    GilHold gil;
    PyRef args(PyTuple_New(static_cast<Py_ssize_t>(inputs.size())));
    if (!args.object) throw std::runtime_error("PyTorchModelRunner: cannot build the arguments: " + take_python_error());
    for (size_t i = 0; i + 1 < inputs.size(); ++i) {
        PyTuple_SET_ITEM(args.object, static_cast<Py_ssize_t>(i), to_capsule(inputs[i].contiguous()));   // steals the reference
    }
    PyTuple_SET_ITEM(args.object, static_cast<Py_ssize_t>(inputs.size() - 1), PyLong_FromLongLong(n_owned));

    PyRef result(PyObject_CallObject(static_cast<PyObject*>(run_method_), args.object));
    if (!result.object) {
        throw std::runtime_error("PyTorchModelRunner: the Python model of '" + model_folder_ + "' failed: " + take_python_error());
    }
    if (!PyTuple_Check(result.object) || PyTuple_GET_SIZE(result.object) != static_cast<Py_ssize_t>(expected_outputs)) {
        throw std::runtime_error("PyTorchModelRunner: the Python model of '" + model_folder_ + "' must return " +
                                 (expected_outputs == 4 ? "(energy, atom_energy, forces, virial)" : "(energy, atom_energy, forces)"));
    }
    std::vector<torch::Tensor> outputs;
    for (size_t i = 0; i < expected_outputs; ++i) {
        outputs.push_back(from_capsule(PyTuple_GET_ITEM(result.object, static_cast<Py_ssize_t>(i)), model_folder_));
    }
    return outputs;
}

#else   // no embedded Python in this build

PyTorchModelRunner::PyTorchModelRunner(const std::string& model_folder, const std::string&, const torch::Device&)
    : model_folder_(model_folder) {
    throw std::runtime_error("PyTorchModelRunner: model folder '" + model_folder + "' is run by Python (metadata backend "
                             "'pytorch'), and this build has no embedded Python: configure dsMD with -DENABLE_PYTHON=ON, or "
                             "convert the model with --backend torchscript");
}

PyTorchModelRunner::~PyTorchModelRunner() = default;

std::vector<torch::Tensor> PyTorchModelRunner::run(const std::vector<torch::Tensor>&) {
    throw std::runtime_error("PyTorchModelRunner: this build has no embedded Python");
}

#endif
