"""TensorNet adapter (matgl 2.0.6, PyTorch Geometric backend).

What the installed matgl does, which this adapter relies on:
  * a TensorNet (matgl.models.TensorNet) works on a PyG Data graph with pos, edge_index, node_type (the index of the element in
    model.element_types) and the periodic image offsets of the edges. Its layers read only edge_index, the edge vectors and
    lengths, the radial basis expansion and node_type from that graph object, which TorchScript cannot hold. This adapter
    runs the same computation (TensorEmbedding.forward, TensorNetInteraction.forward, the output norm, linear and readout)
    on plain tensors, with the model's own submodules.
  * bond_vec = pos[edge_index[1]] - pos[edge_index[0]], the standard edge list of this converter (symmetric, both
    directions), and the image offsets are zero because the ghost atoms are real positions.
  * the model must be extensive (is_intensive=False): its readout then gives one energy per atom (a gated MLP on the node
    features). An intensive model (a mean over the atoms before the output layers) has no per-atom energy.
  * a trained potential is a matgl.apps.Potential: total energy = data_std * model output + data_mean + the sum of the
    element references (AtomRef) of the atoms. The adapter applies data_std and the element references per atom;
    data_mean is a constant per structure and cannot be split over atoms: only 0 is accepted.
  * cutoff = model.cutoff, num_layers = nblocks, so the ghost depth is nblocks * cutoff.
A model is read from a local matgl model folder (model.pt, state.pt, model.json, as Potential.save writes it) with
matgl.load_model; nothing is downloaded: the published models (TensorNet-MatPES, ...) are downloaded by matgl's own
load_model(<name>), which this converter does not call: download the folder yourself and pass its path.
Not supported: intensive models, a model with state attributes (include_state), repulsion terms (calc_repuls), the DGL backend.
"""
from typing import Dict, Optional

import torch
from torch import Tensor

from matgl.utils.maths import (decompose_tensor, new_radial_tensor, scatter_add, tensor_norm, vector_to_skewtensor,
                               vector_to_symtensor)

from .base import MLIPAdapter


def cosine_cutoff(r: Tensor, cutoff: float) -> Tensor:
    """matgl.utils.cutoff.cosine_cutoff with the constant pi written out (TorchScript cannot read the module-level float)."""
    return torch.where(r <= cutoff, 0.5 * (torch.cos(3.141592653589793 * r / cutoff) + 1), torch.zeros_like(r))


def load_tensornet_model(path):
    """The Potential (or bare TensorNet) of a local matgl model folder."""
    from pathlib import Path
    folder = Path(path)
    if not folder.is_dir() or not (folder / "model.pt").is_file():
        raise ValueError(f"TensorNet model folder not found: {path} (a folder with model.pt, state.pt and model.json, as matgl "
                         f"saves it; this converter downloads nothing)")
    import matgl
    matgl.set_backend("PYG")
    loaded = matgl.load_model(str(folder))
    loaded = loaded.eval()
    for parameter in loaded.parameters():
        parameter.requires_grad_(False)
    return loaded


class _Embedding(torch.nn.Module):
    """matgl TensorEmbedding.forward on plain tensors (its submodules are the model's own)."""

    def __init__(self, source):
        super().__init__()
        self.emb = source.emb
        self.emb2 = source.emb2
        self.distance_proj1 = source.distance_proj1
        self.distance_proj2 = source.distance_proj2
        self.distance_proj3 = source.distance_proj3
        self.linears_tensor = source.linears_tensor
        self.linears_scalar = source.linears_scalar
        self.init_norm = source.init_norm
        self.act = source.act
        self.units = int(source.units)
        self.cutoff = float(source.cutoff)

    def forward(self, node_type: Tensor, edge_index: Tensor, edge_attr: Tensor, bond_dist: Tensor, bond_vec: Tensor) -> Tensor:
        x = self.emb(node_type)
        C = cosine_cutoff(bond_dist, self.cutoff)
        W1 = self.distance_proj1(edge_attr) * C.view(-1, 1)
        W2 = self.distance_proj2(edge_attr) * C.view(-1, 1)
        W3 = self.distance_proj3(edge_attr) * C.view(-1, 1)
        edge_vec = bond_vec / torch.norm(bond_vec, dim=1, keepdim=True).clamp(min=1e-6)
        Iij, Aij, Sij = new_radial_tensor(
            torch.eye(3, 3, device=edge_vec.device, dtype=edge_vec.dtype).unsqueeze(0).unsqueeze(0),
            vector_to_skewtensor(edge_vec).unsqueeze(-3),
            vector_to_symtensor(edge_vec).unsqueeze(-3),
            W1, W2, W3)
        zij = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        Zij = self.emb2(zij)
        n_nodes = x.size(0)
        scalars = scatter_add(Zij[..., None, None] * Iij, edge_index[1], dim_size=n_nodes)
        skew_matrices = scatter_add(Zij[..., None, None] * Aij, edge_index[1], dim_size=n_nodes)
        traceless_tensors = scatter_add(Zij[..., None, None] * Sij, edge_index[1], dim_size=n_nodes)

        norm = tensor_norm(scalars + skew_matrices + traceless_tensors)
        norm = self.init_norm(norm)
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_matrices = self.linears_tensor[1](skew_matrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(norm.shape[0], self.units, 3)
        scalars, skew_matrices, traceless_tensors = new_radial_tensor(
            scalars, skew_matrices, traceless_tensors, norm[..., 0], norm[..., 1], norm[..., 2])
        return scalars + skew_matrices + traceless_tensors


class _Interaction(torch.nn.Module):
    """matgl TensorNetInteraction.forward on plain tensors."""

    def __init__(self, source):
        super().__init__()
        self.linears_scalar = source.linears_scalar
        self.linears_tensor = source.linears_tensor
        self.act = source.act
        self.units = int(source.units)
        self.cutoff = float(source.cutoff)
        self.o3 = source.equivariance_invariance_group == "O(3)"

    def forward(self, edge_index: Tensor, bond_dist: Tensor, edge_attr: Tensor, X: Tensor) -> Tensor:
        C = cosine_cutoff(bond_dist, self.cutoff)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(edge_attr.shape[0], self.units, 3)

        X = X / (tensor_norm(X) + 1)[..., None, None]
        scalars, skew_matrices, traceless_tensors = decompose_tensor(X)
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_matrices = self.linears_tensor[1](skew_matrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = scalars + skew_matrices + traceless_tensors

        dst = edge_index[1]
        Im, Am, Sm = new_radial_tensor(scalars[dst], skew_matrices[dst], traceless_tensors[dst],
                                       edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2])
        n_nodes = X.size(0)
        msg = (scatter_add(Im, edge_index[0], dim_size=n_nodes) + scatter_add(Am, edge_index[0], dim_size=n_nodes)
               + scatter_add(Sm, edge_index[0], dim_size=n_nodes))

        if self.o3:
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            scalars, skew_matrices, traceless_tensors = decompose_tensor(A + B)
        else:
            B = torch.matmul(Y, msg)
            scalars, skew_matrices, traceless_tensors = decompose_tensor(2 * B)

        normp1 = (tensor_norm(scalars + skew_matrices + traceless_tensors) + 1)[..., None, None]
        scalars = scalars / normp1
        skew_matrices = skew_matrices / normp1
        traceless_tensors = traceless_tensors / normp1
        scalars = self.linears_tensor[3](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_matrices = self.linears_tensor[4](skew_matrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[5](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = scalars + skew_matrices + traceless_tensors
        return X + dX + torch.matmul(dX, dX)


class TensorNetAdapter(MLIPAdapter):
    model_family = "tensornet"

    def __init__(self, loaded: torch.nn.Module, model_name: str = "tensornet"):
        from ase.data import atomic_numbers
        from matgl.models import TensorNet
        potential = loaded if hasattr(loaded, "model") else None
        model = loaded.model if potential is not None else loaded
        if not isinstance(model, TensorNet):
            raise ValueError(f"TensorNet: expected a matgl TensorNet, got {type(model).__name__}")
        if model.is_intensive:
            raise ValueError("TensorNet: an intensive model (mean over the atoms before the output layers) has no per-atom "
                             "energy: only extensive models (is_intensive=False) are supported")
        if model.include_state:
            raise ValueError("TensorNet: models with state attributes are not supported")
        data_std = 1.0
        offsets = None
        if potential is not None:
            if getattr(potential, "calc_repuls", False):
                raise ValueError("TensorNet: a potential with the ZBL repulsion term is not supported")
            data_mean = float(torch.as_tensor(potential.data_mean).reshape(-1)[0])
            if abs(data_mean) > 0.0:
                raise ValueError(f"TensorNet: data_mean = {data_mean} is a constant per structure and cannot be split over the "
                                 f"atoms: only a potential with data_mean 0 is supported")
            data_std = float(torch.as_tensor(potential.data_std).reshape(-1)[0])
            if potential.element_refs is not None:
                refs = potential.element_refs.property_offset.detach().reshape(-1)
                if refs.shape[0] != len(model.element_types):
                    raise ValueError("TensorNet: the element references do not have one value per element type")
                offsets = refs
        z = []
        for name in model.element_types:
            if name not in atomic_numbers:
                raise ValueError(f"TensorNet element '{name}' is not a chemical symbol")
            z.append(int(atomic_numbers[name]))
        super().__init__(z)
        self.model_name = model_name
        self.cutoff = float(model.cutoff)
        self.num_layers = int(model.num_layers)
        self.bond_expansion = model.bond_expansion
        self.embedding = _Embedding(model.tensor_embedding)
        self.interactions = torch.nn.ModuleList([_Interaction(layer) for layer in model.layers])
        self.out_norm = model.out_norm
        self.linear = model.linear
        self.readout = model.final_layer.gated
        self.data_std = data_std
        self.register_buffer("element_offset", offsets.clone() if offsets is not None else torch.zeros(len(z)))

    def package_versions(self) -> Dict[str, Optional[str]]:
        import importlib.metadata
        return {"matgl": importlib.metadata.version("matgl")}

    @property
    def reference_elements(self):
        return [8, 1] if 1 in self.elements and 8 in self.elements else self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        bond_vec = pos[edge_index[1]] - pos[edge_index[0]]
        bond_dist = torch.norm(bond_vec, dim=1)
        edge_attr = self.bond_expansion(bond_dist)
        X = self.embedding(species, edge_index, edge_attr, bond_dist, bond_vec)
        for layer in self.interactions:
            X = layer(edge_index, bond_dist, edge_attr, X)
        scalars, skew_matrices, traceless_tensors = decompose_tensor(X)
        x = torch.cat((tensor_norm(scalars), tensor_norm(skew_matrices), tensor_norm(traceless_tensors)), dim=-1)
        x = self.linear(self.out_norm(x))
        atomic = self.readout(x).reshape(-1)
        return self.data_std * atomic + self.element_offset.to(atomic.dtype)[species]
