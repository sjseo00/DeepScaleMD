
#include "../../include/force_calculator.h"
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Conditional Python support
#ifdef ENABLE_PYTHON_INTERFACE
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#define PYTHON_ENABLED true
#else
#define PYTHON_ENABLED false
// Dummy Python types and functions for compilation
typedef void* PyObject;
typedef void* PyArrayObject;
typedef long npy_intp;
#define NPY_DOUBLE 0
#define NPY_INT32 0
inline void Py_Initialize() {}
inline void Py_Finalize() {}
inline bool Py_IsInitialized() { return false; }
inline void PyRun_SimpleString(const char*) {}
inline void PyErr_Print() {}
inline PyObject* PyImport_ImportModule(const char*) { return nullptr; }
inline PyObject* PyObject_GetAttrString(PyObject*, const char*) { return nullptr; }
inline bool PyCallable_Check(PyObject*) { return false; }
inline PyObject* PyObject_CallObject(PyObject*, PyObject*) { return nullptr; }
inline PyObject* PyTuple_New(int) { return nullptr; }
inline PyObject* PyTuple_Pack(int, ...) { return nullptr; }
inline void PyTuple_SetItem(PyObject*, int, PyObject*) {}
inline PyObject* PyDict_New() { return nullptr; }
inline void PyDict_SetItemString(PyObject*, const char*, PyObject*) {}
inline PyObject* PyDict_GetItemString(PyObject*, const char*) { return nullptr; }
inline PyObject* PyList_New(int) { return nullptr; }
inline void PyList_SetItem(PyObject*, int, PyObject*) {}
inline PyObject* PyList_GetItem(PyObject*, int) { return nullptr; }
inline bool PyArray_Check(PyObject*) { return false; }
inline PyObject* PyArray_SimpleNew(int, npy_intp*, int) { return nullptr; }
inline void* PyArray_DATA(PyArrayObject*) { return nullptr; }
inline npy_intp* PyArray_DIMS(PyArrayObject*) { return nullptr; }
inline bool PyErr_Occurred() { return false; }
inline int PyList_Size(PyObject*) { return 0; }
inline PyObject* PyFloat_FromDouble(double) { return nullptr; }
inline PyObject* PyLong_FromLong(long) { return nullptr; }
inline double PyFloat_AsDouble(PyObject*) { return 0.0; }
inline bool PyDict_Check(PyObject*) { return false; }
inline bool PyList_Check(PyObject*) { return false; }
inline void Py_DECREF(PyObject*) {}
#endif

PythonForceInterface::PythonForceInterface()
    : python_force_path("python_force"),
      py_module(nullptr), py_ml_module(nullptr),
      py_compute_func(nullptr), py_ml_compute_func(nullptr),
      py_ml_get_energy_func(nullptr),
      py_cached_atom_types(nullptr), py_cached_pbc(nullptr),
      _cached_n_particles(0) {}

PythonForceInterface::~PythonForceInterface() {
    finalize_python();
}

void PythonForceInterface::initialize_python() {
    initialize_python(python_force_path);
}

void PythonForceInterface::initialize_python(const std::string& force_module_path) {
    if (!PYTHON_ENABLED) {
        std::cout << "Python interface not compiled - Python support disabled" << std::endl;
        return;
    }
    
    python_force_path = force_module_path;
    
    if (Py_IsInitialized()) {
        return; // Already initialized
    }
    
    Py_Initialize();
    if (!Py_IsInitialized()) {
        throw std::runtime_error("Failed to initialize Python interpreter");
    }
    
#ifdef ENABLE_PYTHON_INTERFACE
    // Initialize NumPy C API
    // import_array() is a macro that returns NULL on error in C
    // We need to handle it differently in C++ (void function)
    _import_array();  // Use internal function directly
#endif
    
    // Add specified Python force directory to path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import os");
    
    // Add the specified path
    std::string add_path_cmd = "sys.path.append('" + python_force_path + "')";
    PyRun_SimpleString(add_path_cmd.c_str());
    
    // Also try some common relative paths as fallback
    PyRun_SimpleString("sys.path.append('./python_force')");
    PyRun_SimpleString("sys.path.append('../python_force')");
    PyRun_SimpleString("sys.path.append(os.path.join(os.getcwd(), 'python_force'))");
    PyRun_SimpleString("sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'python_force'))");
    
    std::cout << "Python interpreter initialized for force calculations" << std::endl;
    std::cout << "Python force module path: " << python_force_path << std::endl;
}

void PythonForceInterface::set_force_module_path(const std::string& path) {
    python_force_path = path;
}

void PythonForceInterface::initialize_ml_model(const std::string& model_path,
                                                const std::string& model_type,
                                                const std::string& model_size,
                                                const std::string& device,
                                                const std::string& dtype,
                                                const std::map<std::string, std::string>& extra_kwargs) {
    if (!PYTHON_ENABLED) {
        std::cerr << "[ML] Python interface not compiled" << std::endl;
        return;
    }
    if (!Py_IsInitialized()) {
        std::cerr << "[ML] Python not initialized" << std::endl;
        return;
    }

#ifdef ENABLE_PYTHON_INTERFACE
    // Delegate all model-type branching to Python.
    // Python's initialize_ml_forcefield_generic() handles MACE, NequIP, CHGNet,
    // SevenNet, etc. – no C++ recompile needed when adding new model types.
    std::cout << "[ML] Initializing ML model: type=" << model_type
              << " device=" << device << std::endl;

    // Build extra kwargs string (e.g. head='H2O_H2O_PD', chemical_symbols=["O","H"])
    std::string extra_str;
    for (const auto& kv : extra_kwargs) {
        const std::string& val = kv.second;
        // If value looks like a Python literal (list, dict, True/False/None, number)
        // pass it raw; otherwise wrap in quotes as a string.
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
        "    model_path='" + model_path + "',\n"
        "    model_type='" + model_type + "',\n"
        "    model_size='" + model_size + "',\n"
        "    device='" + device + "',\n"
        "    dtype='" + dtype + "',\n"
        + extra_str +
        ")\n"
        "if not _ml_result:\n"
        "    raise RuntimeError('ML model initialization failed: " + model_type + "')\n";

    int ret = PyRun_SimpleString(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ML] ❌ Failed to initialize ML model (" << model_type << ")" << std::endl;
        PyErr_Print();
    } else {
        std::cout << "[ML] ✅ ML model initialized successfully (" << model_type << ")" << std::endl;
    }
#endif
}

void PythonForceInterface::finalize_python() {
    if (!PYTHON_ENABLED) {
        return;
    }
    
    if (Py_IsInitialized()) {
#ifdef ENABLE_PYTHON_INTERFACE
        // Release cached Python objects before Py_Finalize
        if (py_ml_compute_func)  { Py_DECREF((PyObject*)py_ml_compute_func);  py_ml_compute_func  = nullptr; }
        if (py_ml_get_energy_func) { Py_DECREF((PyObject*)py_ml_get_energy_func); py_ml_get_energy_func = nullptr; }
        if (py_ml_module)        { Py_DECREF((PyObject*)py_ml_module);        py_ml_module        = nullptr; }
        if (py_compute_func)     { Py_DECREF((PyObject*)py_compute_func);     py_compute_func     = nullptr; }
        if (py_module)           { Py_DECREF((PyObject*)py_module);           py_module           = nullptr; }
        if (py_cached_atom_types){ Py_DECREF((PyObject*)py_cached_atom_types); py_cached_atom_types = nullptr; }
        if (py_cached_pbc)       { Py_DECREF((PyObject*)py_cached_pbc);       py_cached_pbc       = nullptr; }
#endif
        Py_Finalize();
    }
}

void PythonForceInterface::compute_bond_forces(ParticleData& particles, const ForceFieldParams& params) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }
    
    try {
        // Import bond forces module
        PyObject* module = PyImport_ImportModule("bond_forces");
        if (!module) {
            PyErr_Print();
            return;
        }
        
        // Get the function
        PyObject* func = PyObject_GetAttrString(module, "compute_bond_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_bond_forces function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // Prepare arguments (simplified - would need proper bond data structure)
        // This is a placeholder implementation
        
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in bond force computation: " << e.what() << std::endl;
    }
}

void PythonForceInterface::compute_angle_forces(ParticleData& particles, const ForceFieldParams& params) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }
    
    // Similar implementation to bond forces
    // Placeholder for now
}

void PythonForceInterface::compute_dihedral_forces(ParticleData& particles, const ForceFieldParams& params) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }
    
    // Similar implementation to bond forces
    // Placeholder for now
}

void PythonForceInterface::compute_lj_forces(ParticleData& particles, const ForceFieldParams& params,
                                           const NeighborList& nlist) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }
    
    try {
        PyObject* module = PyImport_ImportModule("lj_forces");
        if (!module) {
            std::cerr << "Cannot import lj_forces module. Make sure it exists in: " << python_force_path << std::endl;
            PyErr_Print();
            return;
        }
        
        PyObject* func = PyObject_GetAttrString(module, "compute_lj_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_lj_forces function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // This would need proper implementation with particle data conversion
        // Placeholder for now
        
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in LJ force computation: " << e.what() << std::endl;
    }
}

void PythonForceInterface::compute_electrostatic_forces(ParticleData& particles, const ForceFieldParams& params,
                                                      const NeighborList& nlist) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }
    
    // Similar implementation to LJ forces
    // Placeholder for now
}

void PythonForceInterface::compute_force(ParticleData& particles, const ForceFieldParams& params,
                                           const NeighborList& nlist,
                                           double box_x, double box_y, double box_z,
                                           bool pbc_x, bool pbc_y, bool pbc_z) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized" << std::endl;
        return;
    }
    
    try {
#ifdef ENABLE_PYTHON_INTERFACE
        const size_t n_particles = particles.x.size();

        // ============================================================
        // [STEP 0] Cache the single user-force ABI.
        //          (PyImport_ImportModule is expensive; do it once)
        // ============================================================
        if (!py_ml_module) {
            py_ml_module = (void*)PyImport_ImportModule("ml_forces");
            if (!py_ml_module) {
                std::cerr << "Cannot import ml_forces module. Make sure it exists in: "
                          << python_force_path << std::endl;
                PyErr_Print();
                return;
            }
        }
        if (!py_ml_compute_func) {
            py_ml_compute_func = (void*)PyObject_GetAttrString(
                (PyObject*)py_ml_module, "compute_force");
            if (!py_ml_compute_func || !PyCallable_Check((PyObject*)py_ml_compute_func)) {
                std::cerr << "Cannot find compute_force(positions, types, cell, pbc)" << std::endl;
                Py_XDECREF((PyObject*)py_ml_compute_func);
                py_ml_compute_func = nullptr;
                return;
            }
        }

        // ============================================================
        // [STEP 1] D2H sync: already handled by download_positions_for_ml()
        // which calls cudaStreamSynchronize() before and after the memcpy.
        // An extra cudaDeviceSynchronize() here is redundant and adds ~1 ms
        // overhead per step. Removed for performance.
        // ============================================================

        // ============================================================
        // [STEP 2] Ensure interleaved position / force buffers are sized
        //          (resize() is a no-op when n_particles is unchanged)
        // ============================================================
        _pos_buf.resize(n_particles * 3);
        _frc_buf.resize(n_particles * 3);

        // Fill interleaved position buffer (SoA → AoS)
        for (size_t i = 0; i < n_particles; ++i) {
            _pos_buf[i * 3 + 0] = static_cast<double>(particles.x[i]);
            _pos_buf[i * 3 + 1] = static_cast<double>(particles.y[i]);
            _pos_buf[i * 3 + 2] = static_cast<double>(particles.z[i]);
        }

        // ============================================================
        // [STEP 3] Wrap position buffer with NumPy – zero extra malloc
        //          PyArray_SimpleNewFromData does NOT copy: NumPy points
        //          directly into _pos_buf.data().
        //          We pass NPY_DOUBLE; ownership stays with _pos_buf.
        // ============================================================
        npy_intp pos_dims[2] = {(npy_intp)n_particles, 3};
        PyObject* py_positions = PyArray_SimpleNewFromData(
            2, pos_dims, NPY_DOUBLE, _pos_buf.data());
        if (!py_positions) { PyErr_Print(); return; }

        // ============================================================
        // [STEP 4] Atom-types array: created once and reused every step
        //          (types never change during a simulation)
        // ============================================================
        if (!py_cached_atom_types || _cached_n_particles != n_particles) {
            Py_XDECREF((PyObject*)py_cached_atom_types);
            npy_intp type_dims[1] = {(npy_intp)n_particles};
            py_cached_atom_types = (void*)PyArray_SimpleNew(1, type_dims, NPY_INT32);
            if (!py_cached_atom_types) { Py_DECREF(py_positions); PyErr_Print(); return; }
            int32_t* type_data = (int32_t*)PyArray_DATA((PyArrayObject*)py_cached_atom_types);
            for (size_t i = 0; i < n_particles; ++i)
                type_data[i] = static_cast<int32_t>(particles.type[i]);
            _cached_n_particles = n_particles;
        }

        // ============================================================
        // [STEP 5] Cell array – created fresh each step (box can change)
        // ============================================================
        npy_intp cell_dims[1] = {3};
        PyObject* py_cell = PyArray_SimpleNew(1, cell_dims, NPY_DOUBLE);
        if (!py_cell) { Py_DECREF(py_positions); PyErr_Print(); return; }
        double* cell_data = (double*)PyArray_DATA((PyArrayObject*)py_cell);
        cell_data[0] = box_x;
        cell_data[1] = box_y;
        cell_data[2] = box_z;

        // ============================================================
        // [STEP 6] PBC array: created once (PBC flags are simulation-wide
        //          constants that never change during a run)
        // ============================================================
        if (!py_cached_pbc) {
            npy_intp pbc_dims[1] = {3};
            py_cached_pbc = (void*)PyArray_SimpleNew(1, pbc_dims, NPY_INT32);
            if (!py_cached_pbc) {
                Py_DECREF(py_positions); Py_DECREF(py_cell); PyErr_Print(); return;
            }
            int32_t* pbc_data = (int32_t*)PyArray_DATA((PyArrayObject*)py_cached_pbc);
            pbc_data[0] = pbc_x ? 1 : 0;
            pbc_data[1] = pbc_y ? 1 : 0;
            pbc_data[2] = pbc_z ? 1 : 0;
        }

        // ============================================================
        // [STEP 7] Call Python/Cython: compute_force(positions, types, cell, pbc)
        // ============================================================
        // Increment ref-counts for cached objects before packing into tuple
        Py_INCREF((PyObject*)py_cached_atom_types);
        Py_INCREF((PyObject*)py_cached_pbc);

        PyObject* py_args = PyTuple_Pack(4,
            py_positions,
            (PyObject*)py_cached_atom_types,
            py_cell,
            (PyObject*)py_cached_pbc);
        PyObject* py_result = PyObject_CallObject((PyObject*)py_ml_compute_func, py_args);
        Py_DECREF(py_args);
        Py_DECREF(py_positions);  // release zero-copy wrapper (data lives in _pos_buf)
        Py_DECREF(py_cell);

        // ============================================================
        // [STEP 8] Unpack forces from result and write into particles SoA
        // ============================================================
        if (py_result && PyArray_Check(py_result)) {
            PyArrayObject* forces_array = (PyArrayObject*)py_result;
            double* forces_data = (double*)PyArray_DATA(forces_array);
            npy_intp* result_dims = PyArray_DIMS(forces_array);

            if (result_dims[0] == (npy_intp)n_particles && result_dims[1] == 3) {
                // Use _frc_buf as staging (avoids repeated virtual calls)
                memcpy(_frc_buf.data(), forces_data, n_particles * 3 * sizeof(double));

                for (size_t i = 0; i < n_particles; ++i) {
                    // Add to existing built-in/LibTorch force components.
                    particles.fx[i] += static_cast<Real>(_frc_buf[i * 3 + 0]);
                    particles.fy[i] += static_cast<Real>(_frc_buf[i * 3 + 1]);
                    particles.fz[i] += static_cast<Real>(_frc_buf[i * 3 + 2]);
                }

                // ============================================================
                // [STEP 8b] Fetch potential energy for this step.  The additive
                // force ABI above returns forces only, so the caller must pull
                // energy separately via ml_forces.get_last_energy() (kcal/mol,
                // already reflects whatever the compute_force() call just did).
                // ============================================================
                if (!py_ml_get_energy_func) {
                    py_ml_get_energy_func = (void*)PyObject_GetAttrString(
                        (PyObject*)py_ml_module, "get_last_energy");
                    if (!py_ml_get_energy_func || !PyCallable_Check((PyObject*)py_ml_get_energy_func)) {
                        Py_XDECREF((PyObject*)py_ml_get_energy_func);
                        py_ml_get_energy_func = nullptr;
                    }
                }
                if (py_ml_get_energy_func) {
                    PyObject* py_energy = PyObject_CallObject((PyObject*)py_ml_get_energy_func, nullptr);
                    if (py_energy) {
                        last_computed_energy_ = PyFloat_AsDouble(py_energy);
                        Py_DECREF(py_energy);
                    } else {
                        PyErr_Clear();
                    }
                }

                // ============================================================
                // [STEP 9] H2D sync: forces are now in CPU host vectors.
                //          upload_forces_from_ml() in md_simulation.cpp does the
                //          actual cudaMemcpy; _sync_after() in the Python adapter
                //          already called torch.cuda.synchronize() to ensure all
                //          PyTorch GPU ops are complete.  An extra device-wide
                //          cudaDeviceSynchronize() here is redundant – removed
                //          for performance (~1 ms saved per step).
                // ============================================================

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

#endif // ENABLE_PYTHON_INTERFACE

    } catch (const std::exception& e) {
        std::cerr << "Error in ML force computation: " << e.what() << std::endl;
    }
}

void PythonForceInterface::compute_ml_forces(ParticleData& particles,
                                             const ForceFieldParams& params,
                                             const NeighborList& nlist,
                                             double box_x, double box_y, double box_z,
                                             bool pbc_x, bool pbc_y, bool pbc_z) {
    compute_force(particles, params, nlist, box_x, box_y, box_z, pbc_x, pbc_y, pbc_z);
}

