#define OMPI_SKIP_MPICXX 1
#include "../include/force_calculator.h"
#include <iostream>
#include <stdexcept>

// Conditional Python support
#ifdef ENABLE_PYTHON_INTERFACE
#include <Python.h>
#define PYTHON_ENABLED true
#else
#define PYTHON_ENABLED false
// Dummy Python types and functions for compilation
typedef void* PyObject;
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
inline void PyTuple_SetItem(PyObject*, int, PyObject*) {}
inline PyObject* PyDict_New() { return nullptr; }
inline void PyDict_SetItemString(PyObject*, const char*, PyObject*) {}
inline PyObject* PyDict_GetItemString(PyObject*, const char*) { return nullptr; }
inline PyObject* PyList_New(int) { return nullptr; }
inline void PyList_SetItem(PyObject*, int, PyObject*) {}
inline PyObject* PyList_GetItem(PyObject*, int) { return nullptr; }
inline int PyList_Size(PyObject*) { return 0; }
inline PyObject* PyFloat_FromDouble(double) { return nullptr; }
inline PyObject* PyLong_FromLong(long) { return nullptr; }
inline double PyFloat_AsDouble(PyObject*) { return 0.0; }
inline bool PyDict_Check(PyObject*) { return false; }
inline bool PyList_Check(PyObject*) { return false; }
inline void Py_DECREF(PyObject*) {}
#endif

PythonForceInterface::PythonForceInterface() : python_force_path("python_force") {}

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

void PythonForceInterface::finalize_python() {
    if (!PYTHON_ENABLED) {
        return;
    }
    
    if (Py_IsInitialized()) {
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
        
        // Prepare arguments - convert particle data and bond topology to Python
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle types
        PyObject* types = PyList_New(particles.num_particles);
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
        }
        PyDict_SetItemString(py_particles, "type", types);
        
        // Create force field parameters dictionary
        PyObject* py_ff_params = PyDict_New();
        // Add bond parameters if available
        if (!params.bond_k.empty()) {
            PyObject* bond_k = PyList_New(params.bond_k.size());
            PyObject* bond_r0 = PyList_New(params.bond_r0.size());
            for (size_t i = 0; i < params.bond_k.size(); i++) {
                PyList_SetItem(bond_k, i, PyFloat_FromDouble(params.bond_k[i]));
                PyList_SetItem(bond_r0, i, PyFloat_FromDouble(params.bond_r0[i]));
            }
            PyDict_SetItemString(py_ff_params, "bond_k", bond_k);
            PyDict_SetItemString(py_ff_params, "bond_r0", bond_r0);
        }
        
        // Call Python function
        PyObject* args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python bond force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid bond force data returned from Python function" << std::endl;
                }
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_bond_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        
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
    
    try {
        // Import angle forces module
        PyObject* module = PyImport_ImportModule("angle_forces");
        if (!module) {
            std::cerr << "Cannot import angle_forces module. Make sure it exists in: " << python_force_path << std::endl;
            PyErr_Print();
            return;
        }
        
        // Get the function
        PyObject* func = PyObject_GetAttrString(module, "compute_angle_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_angle_forces function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // Prepare arguments - convert particle data and angle topology to Python
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle types
        PyObject* types = PyList_New(particles.num_particles);
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
        }
        PyDict_SetItemString(py_particles, "type", types);
        
        // Create force field parameters dictionary
        PyObject* py_ff_params = PyDict_New();
        // Add angle parameters if available
        if (!params.angle_k.empty()) {
            PyObject* angle_k = PyList_New(params.angle_k.size());
            PyObject* angle_theta0 = PyList_New(params.angle_theta0.size());
            for (size_t i = 0; i < params.angle_k.size(); i++) {
                PyList_SetItem(angle_k, i, PyFloat_FromDouble(params.angle_k[i]));
                PyList_SetItem(angle_theta0, i, PyFloat_FromDouble(params.angle_theta0[i]));
            }
            PyDict_SetItemString(py_ff_params, "angle_k", angle_k);
            PyDict_SetItemString(py_ff_params, "angle_theta0", angle_theta0);
        }
        
        // Call Python function
        PyObject* args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python angle force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid angle force data returned from Python function" << std::endl;
                }
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_angle_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in angle force computation: " << e.what() << std::endl;
    }
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
    
    try {
        // Import dihedral forces module
        PyObject* module = PyImport_ImportModule("dihedral_forces");
        if (!module) {
            std::cerr << "Cannot import dihedral_forces module. Make sure it exists in: " << python_force_path << std::endl;
            PyErr_Print();
            return;
        }
        
        // Get the function
        PyObject* func = PyObject_GetAttrString(module, "compute_dihedral_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_dihedral_forces function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // Prepare arguments - convert particle data and dihedral topology to Python
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle types
        PyObject* types = PyList_New(particles.num_particles);
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
        }
        PyDict_SetItemString(py_particles, "type", types);
        
        // Create force field parameters dictionary
        PyObject* py_ff_params = PyDict_New();
        // Add dihedral parameters if available
        if (!params.dihedral_k.empty()) {
            PyObject* dihedral_k = PyList_New(params.dihedral_k.size());
            PyObject* dihedral_phi0 = PyList_New(params.dihedral_phi0.size());
            PyObject* dihedral_n = PyList_New(params.dihedral_n.size());
            for (size_t i = 0; i < params.dihedral_k.size(); i++) {
                PyList_SetItem(dihedral_k, i, PyFloat_FromDouble(params.dihedral_k[i]));
                PyList_SetItem(dihedral_phi0, i, PyFloat_FromDouble(params.dihedral_phi0[i]));
                PyList_SetItem(dihedral_n, i, PyLong_FromLong(params.dihedral_n[i]));
            }
            PyDict_SetItemString(py_ff_params, "dihedral_k", dihedral_k);
            PyDict_SetItemString(py_ff_params, "dihedral_phi0", dihedral_phi0);
            PyDict_SetItemString(py_ff_params, "dihedral_n", dihedral_n);
        }
        
        // Call Python function
        PyObject* args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python dihedral force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid dihedral force data returned from Python function" << std::endl;
                }
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_dihedral_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in dihedral force computation: " << e.what() << std::endl;
    }
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
        
        // Convert particle data and neighbor list to Python format
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle types
        PyObject* types = PyList_New(particles.num_particles);
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
        }
        PyDict_SetItemString(py_particles, "type", types);
        
        // Create force field parameters dictionary
        PyObject* py_ff_params = PyDict_New();
        // Add LJ parameters if available
        if (!params.lj_epsilon.empty()) {
            PyObject* epsilon = PyList_New(params.lj_epsilon.size());
            PyObject* sigma = PyList_New(params.lj_sigma.size());
            for (size_t i = 0; i < params.lj_epsilon.size(); i++) {
                PyList_SetItem(epsilon, i, PyFloat_FromDouble(params.lj_epsilon[i]));
                PyList_SetItem(sigma, i, PyFloat_FromDouble(params.lj_sigma[i]));
            }
            PyDict_SetItemString(py_ff_params, "epsilon", epsilon);
            PyDict_SetItemString(py_ff_params, "sigma", sigma);
        }
        
        // Convert neighbor list to Python format
        PyObject* py_nlist = PyDict_New();
        PyObject* neighbors = PyList_New(nlist.neighbors.size());
        PyObject* neighbor_offset = PyList_New(nlist.neighbor_offset.size());
        PyObject* num_neighbors = PyList_New(nlist.num_neighbors.size());
        
        for (size_t i = 0; i < nlist.neighbors.size(); i++) {
            PyList_SetItem(neighbors, i, PyLong_FromLong(nlist.neighbors[i]));
        }
        for (size_t i = 0; i < nlist.neighbor_offset.size(); i++) {
            PyList_SetItem(neighbor_offset, i, PyLong_FromLong(nlist.neighbor_offset[i]));
        }
        for (size_t i = 0; i < nlist.num_neighbors.size(); i++) {
            PyList_SetItem(num_neighbors, i, PyLong_FromLong(nlist.num_neighbors[i]));
        }
        
        PyDict_SetItemString(py_nlist, "neighbors", neighbors);
        PyDict_SetItemString(py_nlist, "neighbor_offset", neighbor_offset);
        PyDict_SetItemString(py_nlist, "num_neighbors", num_neighbors);
        PyDict_SetItemString(py_nlist, "cutoff", PyFloat_FromDouble(nlist.cutoff));
        
        // Call Python function
        PyObject* args = PyTuple_New(3);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        PyTuple_SetItem(args, 2, py_nlist);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python LJ force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid LJ force data returned from Python function" << std::endl;
                }
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_lj_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        
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
    
    try {
        // Import electrostatic forces module
        PyObject* module = PyImport_ImportModule("electrostatic_forces");
        if (!module) {
            std::cerr << "Cannot import electrostatic_forces module. Make sure it exists in: " << python_force_path << std::endl;
            PyErr_Print();
            return;
        }
        
        PyObject* func = PyObject_GetAttrString(module, "compute_electrostatic_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_electrostatic_forces function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // Convert particle data to Python format
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle types and charges
        PyObject* types = PyList_New(particles.num_particles);
        PyObject* charges = PyList_New(particles.num_particles);
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
            PyList_SetItem(charges, i, PyFloat_FromDouble(particles.charge[i]));
        }
        PyDict_SetItemString(py_particles, "type", types);
        PyDict_SetItemString(py_particles, "charge", charges);
        
        // Create force field parameters dictionary
        PyObject* py_ff_params = PyDict_New();
        // Add electrostatic parameters
        PyDict_SetItemString(py_ff_params, "coulomb_constant", PyFloat_FromDouble(params.coulomb_constant));
        
        // Convert neighbor list to Python format
        PyObject* py_nlist = PyDict_New();
        PyObject* neighbors = PyList_New(nlist.neighbors.size());
        PyObject* neighbor_offset = PyList_New(nlist.neighbor_offset.size());
        PyObject* num_neighbors = PyList_New(nlist.num_neighbors.size());
        
        for (size_t i = 0; i < nlist.neighbors.size(); i++) {
            PyList_SetItem(neighbors, i, PyLong_FromLong(nlist.neighbors[i]));
        }
        for (size_t i = 0; i < nlist.neighbor_offset.size(); i++) {
            PyList_SetItem(neighbor_offset, i, PyLong_FromLong(nlist.neighbor_offset[i]));
        }
        for (size_t i = 0; i < nlist.num_neighbors.size(); i++) {
            PyList_SetItem(num_neighbors, i, PyLong_FromLong(nlist.num_neighbors[i]));
        }
        
        PyDict_SetItemString(py_nlist, "neighbors", neighbors);
        PyDict_SetItemString(py_nlist, "neighbor_offset", neighbor_offset);
        PyDict_SetItemString(py_nlist, "num_neighbors", num_neighbors);
        PyDict_SetItemString(py_nlist, "cutoff", PyFloat_FromDouble(nlist.cutoff));
        
        // Call Python function
        PyObject* args = PyTuple_New(3);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        PyTuple_SetItem(args, 2, py_nlist);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python electrostatic force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid electrostatic force data returned from Python function" << std::endl;
                }
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_electrostatic_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in electrostatic force computation: " << e.what() << std::endl;
    }
}

void PythonForceInterface::compute_ml_forces(ParticleData& particles, const ForceFieldParams& params,
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
        PyObject* module = PyImport_ImportModule("ml_forces");
        if (!module) {
            std::cerr << "Cannot import ml_forces module. Make sure it exists in: " << python_force_path << std::endl;
            PyErr_Print();
            return;
        }
        
        PyObject* func = PyObject_GetAttrString(module, "compute_ml_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_ml_forces function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // Convert particle data to Python format for ML processing
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle velocities for ML features
        PyObject* vel_x = PyList_New(particles.num_particles);
        PyObject* vel_y = PyList_New(particles.num_particles);
        PyObject* vel_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(vel_x, i, PyFloat_FromDouble(particles.vx[i]));
            PyList_SetItem(vel_y, i, PyFloat_FromDouble(particles.vy[i]));
            PyList_SetItem(vel_z, i, PyFloat_FromDouble(particles.vz[i]));
        }
        
        PyDict_SetItemString(py_particles, "vx", vel_x);
        PyDict_SetItemString(py_particles, "vy", vel_y);
        PyDict_SetItemString(py_particles, "vz", vel_z);
        
        // Add particle types and masses
        PyObject* types = PyList_New(particles.num_particles);
        PyObject* masses = PyList_New(particles.num_particles);
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
            PyList_SetItem(masses, i, PyFloat_FromDouble(particles.mass[i]));
        }
        PyDict_SetItemString(py_particles, "type", types);
        PyDict_SetItemString(py_particles, "mass", masses);
        
        // Create force field parameters dictionary for ML
        PyObject* py_ff_params = PyDict_New();
        // Add any ML-specific parameters
        PyDict_SetItemString(py_ff_params, "ml_model_path", PyLong_FromLong(0)); // Placeholder
        
        // Convert neighbor list to Python format
        PyObject* py_nlist = PyDict_New();
        PyObject* neighbors = PyList_New(nlist.neighbors.size());
        PyObject* neighbor_offset = PyList_New(nlist.neighbor_offset.size());
        PyObject* num_neighbors = PyList_New(nlist.num_neighbors.size());
        
        for (size_t i = 0; i < nlist.neighbors.size(); i++) {
            PyList_SetItem(neighbors, i, PyLong_FromLong(nlist.neighbors[i]));
        }
        for (size_t i = 0; i < nlist.neighbor_offset.size(); i++) {
            PyList_SetItem(neighbor_offset, i, PyLong_FromLong(nlist.neighbor_offset[i]));
        }
        for (size_t i = 0; i < nlist.num_neighbors.size(); i++) {
            PyList_SetItem(num_neighbors, i, PyLong_FromLong(nlist.num_neighbors[i]));
        }
        
        PyDict_SetItemString(py_nlist, "neighbors", neighbors);
        PyDict_SetItemString(py_nlist, "neighbor_offset", neighbor_offset);
        PyDict_SetItemString(py_nlist, "num_neighbors", num_neighbors);
        PyDict_SetItemString(py_nlist, "cutoff", PyFloat_FromDouble(nlist.cutoff));
        
        // Call Python ML function
        PyObject* args = PyTuple_New(3);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        PyTuple_SetItem(args, 2, py_nlist);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from ML result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python ML force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid ML force data returned from Python function" << std::endl;
                }
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_ml_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ML force computation: " << e.what() << std::endl;
    }
}

// NEW UNIFIED COMPUTE_FORCES METHOD
void PythonForceInterface::compute_forces(ParticleData& particles, const ForceFieldParams& params, const NeighborList& nlist) {
    if (!PYTHON_ENABLED) {
        std::cerr << "Python interface not compiled - cannot compute Python forces" << std::endl;
        return;
    }
    
    if (!Py_IsInitialized()) {
        std::cerr << "Python not initialized. Please call initialize_python() first." << std::endl;
        return;
    }
    
    try {
        PyObject* module = PyImport_ImportModule("force_calculator");
        if (!module) {
            std::cerr << "Cannot import force_calculator module. Make sure it exists in: " << python_force_path << std::endl;
            PyErr_Print();
            return;
        }
        
        PyObject* func = PyObject_GetAttrString(module, "compute_all_forces");
        if (!func || !PyCallable_Check(func)) {
            std::cerr << "Cannot find compute_all_forces function in force_calculator module" << std::endl;
            std::cerr << "Please implement compute_all_forces(particles, ff_params, neighbor_list) function" << std::endl;
            Py_DECREF(module);
            return;
        }
        
        // Create Python arguments
        // Convert particles data to Python format
        PyObject* py_particles = PyDict_New();
        
        // Add particle positions
        PyObject* pos_x = PyList_New(particles.num_particles);
        PyObject* pos_y = PyList_New(particles.num_particles);
        PyObject* pos_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(pos_x, i, PyFloat_FromDouble(particles.x[i]));
            PyList_SetItem(pos_y, i, PyFloat_FromDouble(particles.y[i]));
            PyList_SetItem(pos_z, i, PyFloat_FromDouble(particles.z[i]));
        }
        
        PyDict_SetItemString(py_particles, "x", pos_x);
        PyDict_SetItemString(py_particles, "y", pos_y);
        PyDict_SetItemString(py_particles, "z", pos_z);
        
        // Add particle velocities
        PyObject* vel_x = PyList_New(particles.num_particles);
        PyObject* vel_y = PyList_New(particles.num_particles);
        PyObject* vel_z = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(vel_x, i, PyFloat_FromDouble(particles.vx[i]));
            PyList_SetItem(vel_y, i, PyFloat_FromDouble(particles.vy[i]));
            PyList_SetItem(vel_z, i, PyFloat_FromDouble(particles.vz[i]));
        }
        
        PyDict_SetItemString(py_particles, "vx", vel_x);
        PyDict_SetItemString(py_particles, "vy", vel_y);
        PyDict_SetItemString(py_particles, "vz", vel_z);
        
        // Add particle masses and types
        PyObject* masses = PyList_New(particles.num_particles);
        PyObject* types = PyList_New(particles.num_particles);
        
        for (size_t i = 0; i < particles.num_particles; i++) {
            PyList_SetItem(masses, i, PyFloat_FromDouble(particles.mass[i]));
            PyList_SetItem(types, i, PyLong_FromLong(particles.type[i]));
        }
        
        PyDict_SetItemString(py_particles, "mass", masses);
        PyDict_SetItemString(py_particles, "type", types);
        
        // Create proper neighbor list data for Python
        PyObject* py_nlist = PyDict_New();
        PyObject* neighbors = PyList_New(nlist.neighbors.size());
        PyObject* neighbor_offset = PyList_New(nlist.neighbor_offset.size());
        PyObject* num_neighbors = PyList_New(nlist.num_neighbors.size());
        
        for (size_t i = 0; i < nlist.neighbors.size(); i++) {
            PyList_SetItem(neighbors, i, PyLong_FromLong(nlist.neighbors[i]));
        }
        for (size_t i = 0; i < nlist.neighbor_offset.size(); i++) {
            PyList_SetItem(neighbor_offset, i, PyLong_FromLong(nlist.neighbor_offset[i]));
        }
        for (size_t i = 0; i < nlist.num_neighbors.size(); i++) {
            PyList_SetItem(num_neighbors, i, PyLong_FromLong(nlist.num_neighbors[i]));
        }
        
        PyDict_SetItemString(py_nlist, "neighbors", neighbors);
        PyDict_SetItemString(py_nlist, "neighbor_offset", neighbor_offset);
        PyDict_SetItemString(py_nlist, "num_neighbors", num_neighbors);
        PyDict_SetItemString(py_nlist, "cutoff", PyFloat_FromDouble(nlist.cutoff));
        PyDict_SetItemString(py_nlist, "list_type", PyLong_FromLong(0)); // Placeholder for list type
        
        // Create comprehensive force field parameters dictionary
        PyObject* py_ff_params = PyDict_New();
        
        // Add bond parameters if available
        if (!params.bond_k.empty()) {
            PyObject* bond_k = PyList_New(params.bond_k.size());
            PyObject* bond_r0 = PyList_New(params.bond_r0.size());
            for (size_t i = 0; i < params.bond_k.size(); i++) {
                PyList_SetItem(bond_k, i, PyFloat_FromDouble(params.bond_k[i]));
                PyList_SetItem(bond_r0, i, PyFloat_FromDouble(params.bond_r0[i]));
            }
            PyDict_SetItemString(py_ff_params, "bond_k", bond_k);
            PyDict_SetItemString(py_ff_params, "bond_r0", bond_r0);
        }
        
        // Add angle parameters if available
        if (!params.angle_k.empty()) {
            PyObject* angle_k = PyList_New(params.angle_k.size());
            PyObject* angle_theta0 = PyList_New(params.angle_theta0.size());
            for (size_t i = 0; i < params.angle_k.size(); i++) {
                PyList_SetItem(angle_k, i, PyFloat_FromDouble(params.angle_k[i]));
                PyList_SetItem(angle_theta0, i, PyFloat_FromDouble(params.angle_theta0[i]));
            }
            PyDict_SetItemString(py_ff_params, "angle_k", angle_k);
            PyDict_SetItemString(py_ff_params, "angle_theta0", angle_theta0);
        }
        
        // Add dihedral parameters if available
        if (!params.dihedral_k.empty()) {
            PyObject* dihedral_k = PyList_New(params.dihedral_k.size());
            PyObject* dihedral_phi0 = PyList_New(params.dihedral_phi0.size());
            for (size_t i = 0; i < params.dihedral_k.size(); i++) {
                PyList_SetItem(dihedral_k, i, PyFloat_FromDouble(params.dihedral_k[i]));
                PyList_SetItem(dihedral_phi0, i, PyFloat_FromDouble(params.dihedral_phi0[i]));
            }
            PyDict_SetItemString(py_ff_params, "dihedral_k", dihedral_k);
            PyDict_SetItemString(py_ff_params, "dihedral_phi0", dihedral_phi0);
        }
        
        // Add basic parameters that are commonly available
        PyDict_SetItemString(py_ff_params, "cutoff", PyFloat_FromDouble(nlist.cutoff));
        PyDict_SetItemString(py_ff_params, "num_atom_types", PyLong_FromLong(10)); // Default placeholder
        
        // Call Python function
        PyObject* args = PyTuple_New(3);
        PyTuple_SetItem(args, 0, py_particles);
        PyTuple_SetItem(args, 1, py_ff_params);
        PyTuple_SetItem(args, 2, py_nlist);
        
        PyObject* result = PyObject_CallObject(func, args);
        
        if (result) {
            // Extract forces from result
            if (PyDict_Check(result)) {
                PyObject* fx_list = PyDict_GetItemString(result, "fx");
                PyObject* fy_list = PyDict_GetItemString(result, "fy");
                PyObject* fz_list = PyDict_GetItemString(result, "fz");
                
                if (fx_list && fy_list && fz_list && 
                    PyList_Check(fx_list) && PyList_Check(fy_list) && PyList_Check(fz_list)) {
                    
                    size_t num_forces = std::min({
                        (size_t)PyList_Size(fx_list),
                        (size_t)PyList_Size(fy_list), 
                        (size_t)PyList_Size(fz_list),
                        particles.num_particles
                    });
                    
                    for (size_t i = 0; i < num_forces; i++) {
                        particles.fx[i] += PyFloat_AsDouble(PyList_GetItem(fx_list, i));
                        particles.fy[i] += PyFloat_AsDouble(PyList_GetItem(fy_list, i));
                        particles.fz[i] += PyFloat_AsDouble(PyList_GetItem(fz_list, i));
                    }
                    
                    std::cout << "Python unified force calculation completed for " << num_forces << " particles" << std::endl;
                } else {
                    std::cerr << "Invalid force data returned from Python function" << std::endl;
                    std::cerr << "Expected dictionary with 'fx', 'fy', 'fz' keys containing lists" << std::endl;
                }
            } else {
                std::cerr << "Python function should return a dictionary with 'fx', 'fy', 'fz' keys" << std::endl;
            }
            Py_DECREF(result);
        } else {
            std::cerr << "Error calling Python compute_all_forces function" << std::endl;
            PyErr_Print();
        }
        
        Py_DECREF(args);
        Py_DECREF(func);
        Py_DECREF(module);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in unified Python force computation: " << e.what() << std::endl;
    }
}
