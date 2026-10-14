#include "PythonBackend.h"

#include "PythonModules.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef ENABLE_PYTHON_INTERFACE
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

namespace {

#ifdef USE_CUDA
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PythonBackend: ") + what + ": " + cudaGetErrorString(err));
    }
}
#endif

#ifdef ENABLE_PYTHON_INTERFACE

// The text of `value` as the inside of a single-quoted Python string literal.
std::string python_quoted(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '\\' || c == '\'') out += '\\';
        out += (c == '\n') ? ' ' : c;
    }
    return out;
}

// Owns one reference.
struct PyRef {
    PyObject* object;
    explicit PyRef(PyObject* o = nullptr) : object(o) {}
    ~PyRef() { Py_XDECREF(object); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
};

// Prints the pending Python error (if any) and stops the run: a failed MLIP call must never be answered with "no force".
[[noreturn]] void python_failure(const std::string& what) {
    if (PyErr_Occurred()) PyErr_Print();
    throw std::runtime_error("PythonBackend: " + what);
}

#endif

}  // namespace

PythonBackend::PythonBackend(const MLIPConfig& cfg) {
#ifndef ENABLE_PYTHON_INTERFACE
    throw std::runtime_error(
        "mlip_backend 'python' needs a build with the embedded Python interface (configure with -DENABLE_PYTHON=ON; "
        "Python 3 development files and NumPy are needed), and this build has none. Use mlip_backend: libtorch, or rebuild.");
#endif
    const PythonMLIPConfig& py = cfg.python;

    if (py.model_type.empty()) {
        throw std::runtime_error(
            "mlip_backend 'python' needs the model: set mlip_config: model_type (mace, nequip, sevennet, ...) and model_path");
    }
    initialize_python(py.module_path);
    if (cfg.verbose) {
        std::cout << "Python MLIP interface initialized" << std::endl;
        std::cout << "Initializing Python MLIP (type=" << py.model_type << ")..." << std::endl;
        std::cout << "  model_path: " << py.model_path << std::endl;
        std::cout << "  model_type: " << py.model_type << std::endl;
        std::cout << "  model_size: " << py.model_size << std::endl;
        std::cout << "  device:     " << py.device << std::endl;
        std::cout << "  dtype:      " << py.dtype << std::endl;
    }
    initialize_ml_model(py, cfg.type_to_z);
}

PythonBackend::~PythonBackend() {
    finalize_python();
}

void PythonBackend::compute(const MLIPInputs& in, MLIPOutputs& out) {
    const size_t n = static_cast<size_t>(in.n_all);

#ifdef USE_CUDA
    if (!in.host_memory) {
        // The copies are ordered after the builtin force kernels on the force stream.
        h_x_.resize(n); h_y_.resize(n); h_z_.resize(n);
        h_fx_.resize(n); h_fy_.resize(n); h_fz_.resize(n);
        const size_t bytes = n * sizeof(double);
        cuda_check(cudaMemcpyAsync(h_x_.data(), in.d_x, bytes, cudaMemcpyDeviceToHost, in.stream), "copy x");
        cuda_check(cudaMemcpyAsync(h_y_.data(), in.d_y, bytes, cudaMemcpyDeviceToHost, in.stream), "copy y");
        cuda_check(cudaMemcpyAsync(h_z_.data(), in.d_z, bytes, cudaMemcpyDeviceToHost, in.stream), "copy z");
        cuda_check(cudaMemcpyAsync(h_fx_.data(), out.d_fx, bytes, cudaMemcpyDeviceToHost, in.stream), "copy fx");
        cuda_check(cudaMemcpyAsync(h_fy_.data(), out.d_fy, bytes, cudaMemcpyDeviceToHost, in.stream), "copy fy");
        cuda_check(cudaMemcpyAsync(h_fz_.data(), out.d_fz, bytes, cudaMemcpyDeviceToHost, in.stream), "copy fz");
        if (staged_type_count_ != in.n_all) {
            h_type_.resize(n);
            cuda_check(cudaMemcpyAsync(h_type_.data(), in.d_type, n * sizeof(int), cudaMemcpyDeviceToHost,
                                       in.stream), "copy type");
            staged_type_count_ = in.n_all;
        }
        cuda_check(cudaStreamSynchronize(in.stream), "synchronize before Python");

        call_python(h_x_.data(), h_y_.data(), h_z_.data(), h_type_.data(), n,
                    h_fx_.data(), h_fy_.data(), h_fz_.data(), in.cell, in.pbc);

        cuda_check(cudaMemcpyAsync(out.d_fx, h_fx_.data(), bytes, cudaMemcpyHostToDevice, in.stream), "upload fx");
        cuda_check(cudaMemcpyAsync(out.d_fy, h_fy_.data(), bytes, cudaMemcpyHostToDevice, in.stream), "upload fy");
        cuda_check(cudaMemcpyAsync(out.d_fz, h_fz_.data(), bytes, cudaMemcpyHostToDevice, in.stream), "upload fz");
        cuda_check(cudaStreamSynchronize(in.stream), "synchronize after upload");
        out.energy = last_energy_;
        return;
    }
#endif

    call_python(in.d_x, in.d_y, in.d_z, in.d_type, n, out.d_fx, out.d_fy, out.d_fz, in.cell, in.pbc);
    out.energy = last_energy_;
}

#ifdef ENABLE_PYTHON_INTERFACE

void PythonBackend::initialize_python(const std::string& module_path) {
    python_mlip_path_ = module_path;

    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            throw std::runtime_error("PythonBackend: failed to initialize the Python interpreter");
        }
        add_active_python_environment();   // before NumPy is imported: the activated environment's packages come first
        if (_import_array() < 0) {   // NumPy's C API
            python_failure("the embedded Python cannot import NumPy (the interpreter this program was built with needs "
                           "numpy; activate the virtual environment that has the packages, or put its site-packages on PYTHONPATH)");
        }
    }

    // Where ml_forces.py is looked for: the configured directory (python_mlip_path), then the usual places.
    std::error_code ec;
    const std::filesystem::path configured = std::filesystem::absolute(module_path, ec).lexically_normal();
    PyRun_SimpleString("import sys, os");
    if (!ec) {
        PyRun_SimpleString(("sys.path.insert(0, '" + python_quoted(configured.string()) + "')").c_str());
    }
    PyRun_SimpleString("sys.path.append('./mlip_python')");
    PyRun_SimpleString("sys.path.append('../mlip_python')");
    PyRun_SimpleString("sys.path.append(os.path.join(os.getcwd(), 'mlip_python'))");
    PyRun_SimpleString("sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'mlip_python'))");

    std::cout << "Python interpreter initialized for the MLIP" << std::endl;
    std::cout << "Python MLIP module path: " << (ec ? module_path : configured.string()) << std::endl;
}

void PythonBackend::initialize_ml_model(const PythonMLIPConfig& py, const std::vector<int>& type_to_z) {
    std::cout << "[ML] Initializing ML model: type=" << py.model_type
              << " device=" << py.device << std::endl;

    std::string extra_str;
    for (const auto& kv : py.extra_kwargs) {
        const std::string& val = kv.second;
        bool is_literal = (!val.empty() &&
                          (val[0] == '[' || val[0] == '{' || val[0] == '(' ||
                           val == "True" || val == "False" || val == "None" ||
                           (val[0] >= '0' && val[0] <= '9') ||
                           (val[0] == '-' && val.size() > 1)));
        if (is_literal)
            extra_str += "    " + kv.first + "=" + val + ",\n";
        else
            extra_str += "    " + kv.first + "='" + python_quoted(val) + "',\n";
    }
    if (!type_to_z.empty()) {       // mlip_type_to_z: which element every MD atom type is
        extra_str += "    type_to_z=[";
        for (size_t i = 0; i < type_to_z.size(); ++i) extra_str += (i ? ", " : "") + std::to_string(type_to_z[i]);
        extra_str += "],\n";
    }

    std::string cmd =
        "import ml_forces as _ml\n"
        "_ml_result = _ml.initialize_ml_forcefield_generic(\n"
        "    model_path='" + python_quoted(py.model_path) + "',\n"
        "    model_type='" + python_quoted(py.model_type) + "',\n"
        "    model_size='" + python_quoted(py.model_size) + "',\n"
        "    device='" + python_quoted(py.device) + "',\n"
        "    dtype='" + python_quoted(py.dtype) + "',\n"
        + extra_str +
        ")\n"
        "if not _ml_result:\n"
        "    raise RuntimeError('ML model initialization failed: " + python_quoted(py.model_type) + "')\n";

    if (PyRun_SimpleString(cmd.c_str()) != 0) {      // Python has printed the traceback
        throw std::runtime_error(
            "PythonBackend: initializing the MLIP '" + py.model_type + "' failed (see the messages above; ml_forces.py is "
            "looked for in '" + python_mlip_path_ + "', the packages of the model must be importable by the embedded Python)");
    }
    std::cout << "[ML] ML model initialized successfully (" << py.model_type << ")" << std::endl;
}

void PythonBackend::finalize_python() {
    if (Py_IsInitialized()) {
        if (py_ml_compute_func_)    { Py_DECREF((PyObject*)py_ml_compute_func_);    py_ml_compute_func_    = nullptr; }
        if (py_ml_get_energy_func_) { Py_DECREF((PyObject*)py_ml_get_energy_func_); py_ml_get_energy_func_ = nullptr; }
        if (py_ml_module_)          { Py_DECREF((PyObject*)py_ml_module_);          py_ml_module_          = nullptr; }
        if (py_cached_atom_types_)  { Py_DECREF((PyObject*)py_cached_atom_types_);  py_cached_atom_types_  = nullptr; }
        if (py_cached_pbc_)         { Py_DECREF((PyObject*)py_cached_pbc_);         py_cached_pbc_         = nullptr; }
        Py_Finalize();
    }
}

void PythonBackend::call_python(const double* x, const double* y, const double* z, const int* type,
                                size_t n_particles, double* fx, double* fy, double* fz,
                                const double* cell, const bool* pbc) {
    if (!Py_IsInitialized()) throw std::runtime_error("PythonBackend: Python is not initialized");

    if (!py_ml_module_) {
        py_ml_module_ = (void*)PyImport_ImportModule("ml_forces");
        if (!py_ml_module_) python_failure("cannot import the module ml_forces; it is looked for in " + python_mlip_path_);
    }
    if (!py_ml_compute_func_) {
        py_ml_compute_func_ = (void*)PyObject_GetAttrString((PyObject*)py_ml_module_, "compute_force");
        if (!py_ml_compute_func_ || !PyCallable_Check((PyObject*)py_ml_compute_func_)) {
            Py_XDECREF((PyObject*)py_ml_compute_func_);
            py_ml_compute_func_ = nullptr;
            python_failure("ml_forces has no function compute_force(positions, types, cell, pbc)");
        }
    }

    pos_buf_.resize(n_particles * 3);
    frc_buf_.resize(n_particles * 3);
    for (size_t i = 0; i < n_particles; ++i) {
        pos_buf_[i * 3 + 0] = x[i];
        pos_buf_[i * 3 + 1] = y[i];
        pos_buf_[i * 3 + 2] = z[i];
    }

    // Zero-copy view of pos_buf_: the Python side copies what it keeps.
    npy_intp pos_dims[2] = {(npy_intp)n_particles, 3};
    PyRef positions(PyArray_SimpleNewFromData(2, pos_dims, NPY_DOUBLE, pos_buf_.data()));
    if (!positions.object) python_failure("cannot wrap the positions as a NumPy array");

    if (!py_cached_atom_types_ || cached_n_particles_ != n_particles) {
        Py_XDECREF((PyObject*)py_cached_atom_types_);
        npy_intp type_dims[1] = {(npy_intp)n_particles};
        py_cached_atom_types_ = (void*)PyArray_SimpleNew(1, type_dims, NPY_INT32);
        if (!py_cached_atom_types_) python_failure("cannot allocate the atom type array");
        int32_t* type_data = (int32_t*)PyArray_DATA((PyArrayObject*)py_cached_atom_types_);
        for (size_t i = 0; i < n_particles; ++i) type_data[i] = static_cast<int32_t>(type[i]);
        cached_n_particles_ = n_particles;
    }

    npy_intp cell_dims[1] = {3};
    PyRef py_cell(PyArray_SimpleNew(1, cell_dims, NPY_DOUBLE));
    if (!py_cell.object) python_failure("cannot allocate the cell array");
    double* cell_data = (double*)PyArray_DATA((PyArrayObject*)py_cell.object);
    cell_data[0] = cell[0];
    cell_data[1] = cell[1];
    cell_data[2] = cell[2];

    if (!py_cached_pbc_) {
        npy_intp pbc_dims[1] = {3};
        py_cached_pbc_ = (void*)PyArray_SimpleNew(1, pbc_dims, NPY_INT32);
        if (!py_cached_pbc_) python_failure("cannot allocate the pbc array");
        int32_t* pbc_data = (int32_t*)PyArray_DATA((PyArrayObject*)py_cached_pbc_);
        pbc_data[0] = pbc[0] ? 1 : 0;
        pbc_data[1] = pbc[1] ? 1 : 0;
        pbc_data[2] = pbc[2] ? 1 : 0;
    }

    // PyTuple_Pack takes its own references.
    PyRef args(PyTuple_Pack(4, positions.object, (PyObject*)py_cached_atom_types_, py_cell.object, (PyObject*)py_cached_pbc_));
    if (!args.object) python_failure("cannot build the argument tuple");
    PyRef result(PyObject_CallObject((PyObject*)py_ml_compute_func_, args.object));
    if (!result.object) python_failure("ml_forces.compute_force raised an error (traceback above)");
    if (!PyArray_Check(result.object)) python_failure("ml_forces.compute_force did not return a NumPy array");

    PyArrayObject* forces_array = (PyArrayObject*)result.object;
    if (PyArray_NDIM(forces_array) != 2 || PyArray_DIMS(forces_array)[0] != (npy_intp)n_particles ||
        PyArray_DIMS(forces_array)[1] != 3 || PyArray_TYPE(forces_array) != NPY_DOUBLE ||
        !PyArray_IS_C_CONTIGUOUS(forces_array)) {
        throw std::runtime_error("PythonBackend: ml_forces.compute_force must return a C-contiguous float64 array of shape [" +
                                 std::to_string(n_particles) + ", 3]");
    }
    memcpy(frc_buf_.data(), PyArray_DATA(forces_array), n_particles * 3 * sizeof(double));
    for (size_t i = 0; i < n_particles; ++i) {
        fx[i] += frc_buf_[i * 3 + 0];
        fy[i] += frc_buf_[i * 3 + 1];
        fz[i] += frc_buf_[i * 3 + 2];
    }

    if (!py_ml_get_energy_func_) {
        py_ml_get_energy_func_ = (void*)PyObject_GetAttrString((PyObject*)py_ml_module_, "get_last_energy");
        if (!py_ml_get_energy_func_ || !PyCallable_Check((PyObject*)py_ml_get_energy_func_)) {
            Py_XDECREF((PyObject*)py_ml_get_energy_func_);
            py_ml_get_energy_func_ = nullptr;
            python_failure("ml_forces has no function get_last_energy()");
        }
    }
    PyRef energy(PyObject_CallObject((PyObject*)py_ml_get_energy_func_, nullptr));
    if (!energy.object) python_failure("ml_forces.get_last_energy raised an error (traceback above)");
    last_energy_ = PyFloat_AsDouble(energy.object);
    if (last_energy_ == -1.0 && PyErr_Occurred()) python_failure("ml_forces.get_last_energy did not return a number");

    static bool first_call = true;
    if (first_call) {
        std::cout << "ML forces computed successfully for " << n_particles << " particles" << std::endl;
        first_call = false;
    }
}

#else  // !ENABLE_PYTHON_INTERFACE

// The constructor refuses to run without Python, so none of these is reached.
void PythonBackend::initialize_python(const std::string&) {}

void PythonBackend::initialize_ml_model(const PythonMLIPConfig&, const std::vector<int>&) {}

void PythonBackend::finalize_python() {}

void PythonBackend::call_python(const double*, const double*, const double*, const int*, size_t,
                                double*, double*, double*, const double*, const bool*) {
    throw std::runtime_error("PythonBackend: this build has no embedded Python (-DENABLE_PYTHON=ON)");
}

#endif  // ENABLE_PYTHON_INTERFACE
