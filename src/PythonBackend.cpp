#include "PythonBackend.h"

#include <cstring>
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

}  // namespace

PythonBackend::PythonBackend(const MLIPConfig& cfg) {
    const PythonMLIPConfig& py = cfg.python;

    initialize_python(py.force_path);
    if (cfg.verbose) {
        std::cout << "Python force calculation interface initialized" << std::endl;
        if (py.model_type.empty()) {
            std::cout << "Using Python force provider without MLIP model initialization" << std::endl;
        } else {
            std::cout << "Initializing Python MLIP (type=" << py.model_type << ")..." << std::endl;
            std::cout << "  model_path: " << py.model_path << std::endl;
            std::cout << "  model_type: " << py.model_type << std::endl;
            std::cout << "  model_size: " << py.model_size << std::endl;
            std::cout << "  device:     " << py.device << std::endl;
            std::cout << "  dtype:      " << py.dtype << std::endl;
        }
    }
    if (!py.model_type.empty()) {
        initialize_ml_model(py);
    }
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

void PythonBackend::initialize_python(const std::string& force_module_path) {
    python_force_path_ = force_module_path;

    if (Py_IsInitialized()) {
        return; // Already initialized
    }

    Py_Initialize();
    if (!Py_IsInitialized()) {
        throw std::runtime_error("Failed to initialize Python interpreter");
    }

    _import_array();  // Use internal function directly

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import os");

    std::string add_path_cmd = "sys.path.append('" + python_force_path_ + "')";
    PyRun_SimpleString(add_path_cmd.c_str());

    PyRun_SimpleString("sys.path.append('./python_force')");
    PyRun_SimpleString("sys.path.append('../python_force')");
    PyRun_SimpleString("sys.path.append(os.path.join(os.getcwd(), 'python_force'))");
    PyRun_SimpleString("sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'python_force'))");

    std::cout << "Python interpreter initialized for force calculations" << std::endl;
    std::cout << "Python force module path: " << python_force_path_ << std::endl;
}

void PythonBackend::initialize_ml_model(const PythonMLIPConfig& py) {
    if (!Py_IsInitialized()) {
        std::cerr << "[ML] Python not initialized" << std::endl;
        return;
    }

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
            extra_str += "    " + kv.first + "='" + val + "',\n";
    }

    std::string cmd =
        "import ml_forces as _ml\n"
        "_ml_result = _ml.initialize_ml_forcefield_generic(\n"
        "    model_path='" + py.model_path + "',\n"
        "    model_type='" + py.model_type + "',\n"
        "    model_size='" + py.model_size + "',\n"
        "    device='" + py.device + "',\n"
        "    dtype='" + py.dtype + "',\n"
        + extra_str +
        ")\n"
        "if not _ml_result:\n"
        "    raise RuntimeError('ML model initialization failed: " + py.model_type + "')\n";

    int ret = PyRun_SimpleString(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ML] ❌ Failed to initialize ML model (" << py.model_type << ")" << std::endl;
        PyErr_Print();
    } else {
        std::cout << "[ML] ✅ ML model initialized successfully (" << py.model_type << ")" << std::endl;
    }
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
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }

    try {
        if (!py_ml_module_) {
            py_ml_module_ = (void*)PyImport_ImportModule("ml_forces");
            if (!py_ml_module_) {
                std::cerr << "Cannot import ml_forces module. Make sure it exists in: "
                          << python_force_path_ << std::endl;
                PyErr_Print();
                return;
            }
        }
        if (!py_ml_compute_func_) {
            py_ml_compute_func_ = (void*)PyObject_GetAttrString(
                (PyObject*)py_ml_module_, "compute_force");
            if (!py_ml_compute_func_ || !PyCallable_Check((PyObject*)py_ml_compute_func_)) {
                std::cerr << "Cannot find compute_force(positions, types, cell, pbc)" << std::endl;
                Py_XDECREF((PyObject*)py_ml_compute_func_);
                py_ml_compute_func_ = nullptr;
                return;
            }
        }

        pos_buf_.resize(n_particles * 3);
        frc_buf_.resize(n_particles * 3);

        for (size_t i = 0; i < n_particles; ++i) {
            pos_buf_[i * 3 + 0] = x[i];
            pos_buf_[i * 3 + 1] = y[i];
            pos_buf_[i * 3 + 2] = z[i];
        }

        npy_intp pos_dims[2] = {(npy_intp)n_particles, 3};
        PyObject* py_positions = PyArray_SimpleNewFromData(
            2, pos_dims, NPY_DOUBLE, pos_buf_.data());
        if (!py_positions) { PyErr_Print(); return; }

        if (!py_cached_atom_types_ || cached_n_particles_ != n_particles) {
            Py_XDECREF((PyObject*)py_cached_atom_types_);
            npy_intp type_dims[1] = {(npy_intp)n_particles};
            py_cached_atom_types_ = (void*)PyArray_SimpleNew(1, type_dims, NPY_INT32);
            if (!py_cached_atom_types_) { Py_DECREF(py_positions); PyErr_Print(); return; }
            int32_t* type_data = (int32_t*)PyArray_DATA((PyArrayObject*)py_cached_atom_types_);
            for (size_t i = 0; i < n_particles; ++i)
                type_data[i] = static_cast<int32_t>(type[i]);
            cached_n_particles_ = n_particles;
        }

        npy_intp cell_dims[1] = {3};
        PyObject* py_cell = PyArray_SimpleNew(1, cell_dims, NPY_DOUBLE);
        if (!py_cell) { Py_DECREF(py_positions); PyErr_Print(); return; }
        double* cell_data = (double*)PyArray_DATA((PyArrayObject*)py_cell);
        cell_data[0] = cell[0];
        cell_data[1] = cell[1];
        cell_data[2] = cell[2];

        if (!py_cached_pbc_) {
            npy_intp pbc_dims[1] = {3};
            py_cached_pbc_ = (void*)PyArray_SimpleNew(1, pbc_dims, NPY_INT32);
            if (!py_cached_pbc_) {
                Py_DECREF(py_positions); Py_DECREF(py_cell); PyErr_Print(); return;
            }
            int32_t* pbc_data = (int32_t*)PyArray_DATA((PyArrayObject*)py_cached_pbc_);
            pbc_data[0] = pbc[0] ? 1 : 0;
            pbc_data[1] = pbc[1] ? 1 : 0;
            pbc_data[2] = pbc[2] ? 1 : 0;
        }

        Py_INCREF((PyObject*)py_cached_atom_types_);
        Py_INCREF((PyObject*)py_cached_pbc_);

        PyObject* py_args = PyTuple_Pack(4,
            py_positions,
            (PyObject*)py_cached_atom_types_,
            py_cell,
            (PyObject*)py_cached_pbc_);
        PyObject* py_result = PyObject_CallObject((PyObject*)py_ml_compute_func_, py_args);
        Py_DECREF(py_args);
        Py_DECREF(py_positions);  // release zero-copy wrapper (data lives in pos_buf_)
        Py_DECREF(py_cell);

        if (py_result && PyArray_Check(py_result)) {
            PyArrayObject* forces_array = (PyArrayObject*)py_result;
            double* forces_data = (double*)PyArray_DATA(forces_array);
            npy_intp* result_dims = PyArray_DIMS(forces_array);

            if (result_dims[0] == (npy_intp)n_particles && result_dims[1] == 3) {
                memcpy(frc_buf_.data(), forces_data, n_particles * 3 * sizeof(double));

                for (size_t i = 0; i < n_particles; ++i) {
                    fx[i] += frc_buf_[i * 3 + 0];
                    fy[i] += frc_buf_[i * 3 + 1];
                    fz[i] += frc_buf_[i * 3 + 2];
                }

                if (!py_ml_get_energy_func_) {
                    py_ml_get_energy_func_ = (void*)PyObject_GetAttrString(
                        (PyObject*)py_ml_module_, "get_last_energy");
                    if (!py_ml_get_energy_func_ || !PyCallable_Check((PyObject*)py_ml_get_energy_func_)) {
                        Py_XDECREF((PyObject*)py_ml_get_energy_func_);
                        py_ml_get_energy_func_ = nullptr;
                    }
                }
                if (py_ml_get_energy_func_) {
                    PyObject* py_energy = PyObject_CallObject((PyObject*)py_ml_get_energy_func_, nullptr);
                    if (py_energy) {
                        last_energy_ = PyFloat_AsDouble(py_energy);
                        Py_DECREF(py_energy);
                    } else {
                        PyErr_Clear();
                    }
                }

                static bool first_call = true;
                if (first_call) {
                    std::cout << "✅ ML forces computed successfully for "
                              << n_particles << " particles" << std::endl;
                    first_call = false;
                }
            } else {
                std::cerr << "❌ Unexpected force array dimensions from Python" << std::endl;
            }
            Py_DECREF(py_result);
        } else {
            std::cerr << "❌ Python ML force function returned invalid result" << std::endl;
            if (PyErr_Occurred()) PyErr_Print();
            Py_XDECREF(py_result);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in ML force computation: " << e.what() << std::endl;
    }
}

#else  // !ENABLE_PYTHON_INTERFACE

void PythonBackend::initialize_python(const std::string&) {
    std::cout << "Python interface not compiled - Python support disabled" << std::endl;
}

void PythonBackend::initialize_ml_model(const PythonMLIPConfig&) {
    std::cerr << "[ML] Python interface not compiled" << std::endl;
}

void PythonBackend::finalize_python() {}

void PythonBackend::call_python(const double*, const double*, const double*, const int*, size_t,
                                double*, double*, double*, const double*, const bool*) {
    std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
}

#endif  // ENABLE_PYTHON_INTERFACE
