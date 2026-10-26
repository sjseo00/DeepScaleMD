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
from typing import Dict, Optional, Tuple

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

    def forward(self, node_type: Tensor, edge_index: Tensor, edge_attr: Tensor, bond_dist: Tensor, bond_vec: Tensor,
                n_nodes: int) -> Tensor:
        # edge_index[1] is the atom an edge feeds (the matgl convention); n_nodes rows are computed, the first n_nodes atoms
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
    """matgl TensorNetInteraction.forward on plain tensors. forward_layered is the same layer for a rank with ghost atoms: the node
    tensors it works on are those of the owned atoms, and the pieces the messages need from the senders (the three irreducible parts
    of every atom's tensor after the first linear layers) are sent to the ghost copies with one halo exchange."""

    def __init__(self, source):
        super().__init__()
        self.linears_scalar = source.linears_scalar
        self.linears_tensor = source.linears_tensor
        self.act = source.act
        self.units = int(source.units)
        self.cutoff = float(source.cutoff)
        self.o3 = source.equivariance_invariance_group == "O(3)"
        for index in range(3):
            if self.linears_tensor[index].bias is not None:
                raise ValueError("TensorNet: the linear layers in front of the messages have a bias, which the layer exchange "
                                 "cannot send in its compact form")

    def _edge_weights(self, bond_dist: Tensor, edge_attr: Tensor) -> Tensor:
        C = cosine_cutoff(bond_dist, self.cutoff)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        return (edge_attr * C.view(-1, 1)).reshape(edge_attr.shape[0], self.units, 3)

    def _node_parts(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        X = X / (tensor_norm(X) + 1)[..., None, None]
        scalars, skew_matrices, traceless_tensors = decompose_tensor(X)
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_matrices = self.linears_tensor[1](skew_matrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return X, scalars + skew_matrices + traceless_tensors, scalars, skew_matrices, traceless_tensors

    def _finish(self, X: Tensor, Y: Tensor, msg: Tensor) -> Tensor:
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

    def forward(self, edge_index: Tensor, bond_dist: Tensor, edge_attr: Tensor, X: Tensor) -> Tensor:
        weights = self._edge_weights(bond_dist, edge_attr)
        X, Y, scalars, skew_matrices, traceless_tensors = self._node_parts(X)
        dst = edge_index[1]
        Im, Am, Sm = new_radial_tensor(scalars[dst], skew_matrices[dst], traceless_tensors[dst],
                                       weights[..., 0], weights[..., 1], weights[..., 2])
        n_nodes = X.size(0)
        msg = (scatter_add(Im, edge_index[0], dim_size=n_nodes) + scatter_add(Am, edge_index[0], dim_size=n_nodes)
               + scatter_add(Sm, edge_index[0], dim_size=n_nodes))
        return self._finish(X, Y, msg)

    def forward_layered(self, edge_index: Tensor, bond_dist: Tensor, edge_attr: Tensor, X: Tensor, n_ghost: int) -> Tensor:
        """X [n_owned, units, 3, 3] of the owned atoms; returns the same for the next layer. edge_index has only edges whose
        receiver (edge_index[0]) is an owned atom."""
        weights = self._edge_weights(bond_dist, edge_attr)
        X, Y, scalars, skew_matrices, traceless_tensors = self._node_parts(X)
        n_owned = X.size(0)
        # The three parts are one scalar (times the unit matrix), a skew matrix (3 numbers) and a symmetric traceless matrix
        # (5 numbers) per channel, because the linear layers act on the channels only: 9 numbers per channel go to the ghosts.
        packed = torch.stack([scalars[..., 0, 0],
                              skew_matrices[..., 0, 1], skew_matrices[..., 0, 2], skew_matrices[..., 1, 2],
                              traceless_tensors[..., 0, 0], traceless_tensors[..., 1, 1], traceless_tensors[..., 0, 1],
                              traceless_tensors[..., 0, 2], traceless_tensors[..., 1, 2]], dim=-1)
        packed = torch.ops.dsmd.halo_exchange(packed.reshape(n_owned, self.units * 9), n_ghost)
        packed = packed.reshape(n_owned + n_ghost, self.units, 9)
        eye = torch.eye(3, 3, device=packed.device, dtype=packed.dtype)
        zero = torch.zeros_like(packed[..., 0])
        scalars = packed[..., 0][..., None, None] * eye
        skew_matrices = torch.stack([torch.stack([zero, packed[..., 1], packed[..., 2]], dim=-1),
                                     torch.stack([-packed[..., 1], zero, packed[..., 3]], dim=-1),
                                     torch.stack([-packed[..., 2], -packed[..., 3], zero], dim=-1)], dim=-2)
        traceless_tensors = torch.stack([torch.stack([packed[..., 4], packed[..., 6], packed[..., 7]], dim=-1),
                                         torch.stack([packed[..., 6], packed[..., 5], packed[..., 8]], dim=-1),
                                         torch.stack([packed[..., 7], packed[..., 8], -packed[..., 4] - packed[..., 5]],
                                                     dim=-1)], dim=-2)
        dst = edge_index[1]
        Im, Am, Sm = new_radial_tensor(scalars[dst], skew_matrices[dst], traceless_tensors[dst],
                                       weights[..., 0], weights[..., 1], weights[..., 2])
        msg = (scatter_add(Im, edge_index[0], dim_size=n_owned) + scatter_add(Am, edge_index[0], dim_size=n_owned)
               + scatter_add(Sm, edge_index[0], dim_size=n_owned))
        return self._finish(X, Y, msg)


class TensorNetAdapter(MLIPAdapter):
    model_family = "tensornet"
    supports_shifts = True
    supports_layer_exchange = True
    # Both the embedding and the interaction layers sum onto one atom per edge; the embedding is written here with the
    # receiver in edge_index[0] (matgl has it in edge_index[1]: the edge list is symmetric, so it is the same sum), as the
    # interaction layers have it.
    edge_receiver_row = 0

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
        # The embedding sums over the edges of every atom too: the features of the atoms one cutoff further out go into it, so
        # the energy of an atom depends on atoms (nblocks + 1) cutoffs away: the embedding counts as a layer here.
        self.num_layers = int(model.num_layers) + 1
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
        return self.forward_with_shifts(pos, species, edge_index, torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device))

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        bond_vec = pos[edge_index[1]] - pos[edge_index[0]] + edge_shift
        bond_dist = torch.norm(bond_vec, dim=1)
        edge_attr = self.bond_expansion(bond_dist)
        # the embedding with the receiver in row 0: the edge seen from its other end (swapped atoms, opposite vector)
        X = self.embedding(species, edge_index.flip(0), edge_attr, bond_dist, -bond_vec, pos.shape[0])
        for layer in self.interactions:
            X = layer(edge_index, bond_dist, edge_attr, X)
        return self._readout(X, species)

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """The embedding and the interaction layers of matgl's TensorNet for a rank with ghost atoms: only the owned atoms are
        computed (the only receivers of edges), the pieces of their tensors that the next layer's messages need are sent to the ghost
        copies after the embedding and after every layer but the last. Per atom energies [N], zero for the ghost atoms."""
        n_ghost = pos.shape[0] - n_owned
        bond_vec = pos[edge_index[1]] - pos[edge_index[0]] + edge_shift
        bond_dist = torch.norm(bond_vec, dim=1)
        edge_attr = self.bond_expansion(bond_dist)
        X = self.embedding(species, edge_index.flip(0), edge_attr, bond_dist, -bond_vec, n_owned)
        for layer in self.interactions:
            X = layer.forward_layered(edge_index, bond_dist, edge_attr, X, n_ghost)
        energy = self._readout(X, species[:n_owned])
        return torch.cat([energy, torch.zeros(n_ghost, dtype=energy.dtype, device=energy.device)])

    def _readout(self, X: Tensor, species: Tensor) -> Tensor:
        scalars, skew_matrices, traceless_tensors = decompose_tensor(X)
        x = torch.cat((tensor_norm(scalars), tensor_norm(skew_matrices), tensor_norm(traceless_tensors)), dim=-1)
        x = self.linear(self.out_norm(x))
        atomic = self.readout(x).reshape(-1)
        return self.data_std * atomic + self.element_offset.to(atomic.dtype)[species]
