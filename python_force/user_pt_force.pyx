# cython: language_level=3
"""Cython bridge for a user TorchScript ``.pt`` force model.

The C++ layer hands NumPy views over its particle buffers to ``compute_force``.
This extension accepts those views directly (no list/dict marshaling) and keeps
the user model loaded for the lifetime of the simulation.
"""
import numpy as np
cimport numpy as cnp


cdef class UserPTForce:
    cdef object model
    cdef object device
    cdef object dtype

    def __init__(self, str model_path, str device="cpu", str dtype="float32"):
        import torch
        self.device = device
        self.dtype = dtype
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def compute_force(self, cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] positions,
                      cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] types,
                      object cell, object pbc):
        """Return ``[N,3]`` float64 forces for the current particle buffers.

        The TorchScript module may return either forces directly, or an energy
        scalar/tensor; in the latter case forces are ``-dE/dR``.  Models with a
        different signature should be wrapped/exported to ``(positions, types)``.
        """
        import torch
        torch_dtype = torch.float64 if self.dtype == "float64" else torch.float32
        pos = torch.as_tensor(positions, dtype=torch_dtype, device=self.device)
        species = torch.as_tensor(types, dtype=torch.long, device=self.device)
        pos.requires_grad_(True)
        result = self.model(pos, species)
        if isinstance(result, dict):
            result = result.get("forces", result.get("force", result.get("energy")))
        if isinstance(result, (tuple, list)):
            result = result[-1]
        if result.ndim == 2 and result.shape[-1] == 3:
            force = result
        else:
            force = -torch.autograd.grad(result.sum(), pos)[0]
        return np.ascontiguousarray(force.detach().cpu().numpy(), dtype=np.float64)
