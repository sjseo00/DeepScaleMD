"""
ml_forces.py  –  the module the C++ PythonBackend imports  (mlip_backend: python)
=================================================================================

PythonBackend (src/PythonBackend.cpp) imports this module by name and calls three functions:

    initialize_ml_forcefield_generic(model_path, model_type, model_size, device, dtype, **kwargs) -> bool
    compute_force(positions, atom_types, cell, pbc)  -> forces [N, 3] float64, kcal/mol/A (added to the MD forces)
    get_last_energy()                                -> potential energy, kcal/mol

Everything about the individual MLIPs (MACE, NequIP, SevenNet, ...) lives in mlip_interface.py. The keyword arguments of
initialize_ml_forcefield_generic() are the keys of the input file's `mlip_config:` section, plus `type_to_z` (the input
file's `mlip_type_to_z`).
"""
import numpy as np

import mlip_interface as _mlip     # an ImportError here is reported by PythonBackend with the module path searched


def initialize_ml_forcefield_generic(model_path=None, model_type="mace", model_size="small", device="cpu",
                                     dtype="float32", **kwargs) -> bool:
    return _mlip.initialize(model_type=model_type, model_path=model_path, model_size=model_size, device=device,
                            dtype=dtype, **kwargs)


def compute_force(positions, atom_types, cell, pbc):
    """positions [N, 3] float64 (A), atom_types [N] int32, cell [3] float64 (box edges, A), pbc [3] int32."""
    return _mlip.compute_forces(np.asarray(positions, dtype=np.float64), np.asarray(atom_types),
                                np.asarray(cell, dtype=np.float64), np.asarray(pbc, dtype=np.int32))


def get_last_energy() -> float:
    return _mlip.get_last_energy()
