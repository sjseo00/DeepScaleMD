"""MatterSim adapter (mattersim 1.2.3, the M3GNet-based MatterSim-v1 models).

MatterSim-v1 is an M3GNet (mattersim.forcefield.m3gnet.m3gnet.M3Gnet): a graph network with three-body terms. Its input is
built from a periodic ase.Atoms by mattersim.datasets.utils.convertor.GraphConvertor and handed to the model as a dict. This
adapter builds the same tensors from the standard input with torch operations and runs the layers of M3Gnet.forward itself.
What it relies on, from the installed mattersim:
  * edge_index [2, E]: edge_index[0] is the center (sent atom), edge_index[1] the neighbor, every pair within the cutoff (5 A)
    in both directions: the standard edge list of this converter; the edge vector is pos[center] - pos[neighbor] (the model's
    periodic image offsets are zero because the ghost atoms are real positions);
  * three_body_indices [T, 2]: every ordered pair of different edges with the same center, both within the three-body cutoff
    (4 A); the left edge of a triple is also the edge its sum is scattered to (the model's three_body_edge_map);
  * the atom embedding takes the one-hot of the atomic number Z (max_z + 1 classes, Z = 1..94), and the energy of every atom is
    the output of the final gated MLP rescaled by the AtomScaling normalizer (scale and shift per element). The model's energy
    is the sum of these per-atom energies, which is what is returned (before that sum).
  * cutoff = 5 A, num_layers = the number of graph_conv blocks (4), so the ghost depth is 20 A for the defaults.
The weights of the published MatterSim-v1 checkpoints (1M, 5M) are downloaded by mattersim from the project's repository when
they are not given as a file; this converter does not download anything: pass --model <checkpoint file>.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from torch_runstats.scatter import scatter      # a dependency of mattersim; this module is imported only for MatterSim

from .base import MLIPAdapter
from .triplets import ordered_edge_pairs


def _polynomial(r: Tensor, cutoff: float) -> Tensor:
    """mattersim's polynomial cutoff function (message_passing.polynomial)."""
    ratio = torch.div(r, cutoff)
    result = 1 - 6 * torch.pow(ratio, 5) + 15 * torch.pow(ratio, 4) - 10 * torch.pow(ratio, 3)
    return torch.clamp(result, min=0.0)


class _ThreeBody(torch.nn.Module):
    """mattersim's ThreeDInteraction with its forward written for TorchScript: the original has a `three_body_edge_map: Tensor
    = None` default that TorchScript cannot save. The submodules are the model's own."""

    def __init__(self, source):
        super().__init__()
        self.atom_mlp = source.atom_mlp
        self.edge_gate_mlp = source.edge_gate_mlp
        self.threebody_cutoff = float(source.threebody_cutoff)

    def forward(self, edge_attr: Tensor, three_basis: Tensor, atom_attr: Tensor, edge_index: Tensor, three_body_index: Tensor,
                edge_length: Tensor, left_edge: Tensor, n_edges: int) -> Tensor:
        atom_mask = (self.atom_mlp(atom_attr)[edge_index[0][three_body_index[:, 1]]]
                     * _polynomial(edge_length[three_body_index[:, 0]], self.threebody_cutoff)
                     * _polynomial(edge_length[three_body_index[:, 1]], self.threebody_cutoff))
        three_basis = three_basis * atom_mask
        # the triples of one directed edge (their left edge) are summed into that edge
        e_ij_tuda = scatter(three_basis, left_edge, dim=0, reduce="sum", dim_size=n_edges)
        return edge_attr + self.edge_gate_mlp(e_ij_tuda)


class _Block(torch.nn.Module):
    """mattersim's MainBlock (one message passing round) with the same change."""

    def __init__(self, source):
        super().__init__()
        self.gated_mlp_atom = source.gated_mlp_atom
        self.edge_layer_atom = source.edge_layer_atom
        self.gated_mlp_edge = source.gated_mlp_edge
        self.edge_layer_edge = source.edge_layer_edge
        self.three_body = _ThreeBody(source.three_body)

    def forward(self, atom_attr: Tensor, edge_attr: Tensor, edge_attr_zero: Tensor, edge_index: Tensor, three_basis: Tensor,
                three_body_index: Tensor, edge_length: Tensor, left_edge: Tensor, n_atoms: int,
                n_edges: int) -> Tuple[Tensor, Tensor]:
        edge_attr = self.three_body(edge_attr, three_basis, atom_attr, edge_index, three_body_index, edge_length, left_edge, n_edges)
        feat = torch.concat([atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr], dim=1)
        edge_attr = edge_attr + self.gated_mlp_edge(feat) * self.edge_layer_edge(edge_attr_zero)
        feat = torch.concat([atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr], dim=1)
        atom_attr_prime = self.gated_mlp_atom(feat) * self.edge_layer_atom(edge_attr_zero)
        atom_attr = atom_attr + scatter(atom_attr_prime, edge_index[0], dim=0, dim_size=n_atoms)
        return atom_attr, edge_attr


def load_mattersim_model(path) -> torch.nn.Module:
    """The M3Gnet of a MatterSim checkpoint file (the file mattersim's Potential.from_checkpoint reads; it is not downloaded
    here)."""
    from pathlib import Path
    if not path or not Path(path).is_file():
        raise ValueError(f"MatterSim checkpoint file not found: {path} (this converter does not download weights: pass --model "
                         f"<checkpoint .pth file>)")
    from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "model_args" not in checkpoint:
        raise ValueError(f"{path} is not a MatterSim checkpoint (expected a dict with 'model' and 'model_args')")
    args = dict(checkpoint["model_args"])
    args["device"] = "cpu"
    model = M3Gnet(**args)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class MatterSimAdapter(MLIPAdapter):
    model_family = "mattersim"

    def __init__(self, model: torch.nn.Module, model_name: str = "mattersim"):
        from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
        if not isinstance(model, M3Gnet):
            raise ValueError(f"MatterSim: expected a mattersim M3Gnet model, got {type(model).__name__}")
        super().__init__(list(range(1, int(model.max_z) + 1)))
        self.model_name = model_name
        self.cutoff = float(model.model_args["cutoff"])
        self.threebody_cutoff = float(model.model_args["threebody_cutoff"])
        self.num_layers = int(model.model_args["num_blocks"])
        self.max_z = int(model.max_z)
        self.rbf = model.rbf
        self.sbf = model.sbf
        self.edge_encoder = model.edge_encoder
        self.graph_conv = torch.nn.ModuleList([_Block(block) for block in model.graph_conv])
        self.final = model.final
        self.atom_embedding = model.atom_embedding
        self.normalizer = model.normalizer

    def package_versions(self) -> Dict[str, Optional[str]]:
        import importlib.metadata
        return {"mattersim": importlib.metadata.version("mattersim")}

    @property
    def reference_elements(self):
        return [8, 1] if 1 in self.elements and 8 in self.elements else self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        z = species + 1
        center = edge_index[0]

        edge_vector = pos[center] - pos[edge_index[1]]
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        left, right = ordered_edge_pairs(center, edge_length, self.threebody_cutoff, n_atoms)
        three_body_indices = torch.stack([left, right], dim=1)
        vij = edge_vector[left]
        vik = edge_vector[right]
        rij = edge_length[left]
        rik = edge_length[right]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)           # keeps acos finite
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)

        atom_attr = self.atom_embedding(torch.nn.functional.one_hot(z, num_classes=self.max_z + 1).to(pos.dtype))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        for conv in self.graph_conv:
            atom_attr, edge_attr = conv(atom_attr, edge_attr, edge_attr_zero, edge_index, three_basis, three_body_indices,
                                        edge_length, left, n_atoms, n_edges)
        energies_i = self.final(atom_attr).view(-1)
        return self.normalizer(energies_i, z)
