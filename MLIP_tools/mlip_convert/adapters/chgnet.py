"""CHGNet adapter (chgnet 0.4.2).

CHGNet is a three-body model: besides the atom graph (every pair within the atom cutoff) it has a bond graph of angles
(every ordered pair of bonds at an atom, both shorter than the bond cutoff). Its own input is built by a graph converter
from a periodic pymatgen structure and handed to the model as a BatchedGraph. This adapter builds the same tensors from the
standard input (positions of owned + ghost atoms, a symmetric directed edge list) with torch operations, and runs the layers
of CHGNet._compute itself. What it relies on, from the installed chgnet:
  * the atom graph holds directed edges [center, neighbor]; the bond vector is center - neighbor and is normalized for the
    angles; an undirected bond (the pair of a directed edge and its reverse) carries the radial basis of its length, the
    directed edges point to it through directed2undirected;
  * the bond graph has one row for every ordered pair (left bond, right bond) of different directed edges with the same
    center, both shorter than the bond cutoff: [center atom, undirected index of the left bond, undirected index of the right bond],
    and the angle basis comes from the unit vectors of the two directed bonds;
  * _compute: atom embedding of Z - 1, bond/angle embeddings, n_conv - 1 rounds of (AtomConv, BondConv, AngleUpdate) and a last
    AtomConv, a layer norm, and the readout MLP on every atom (mlp_first); the pooled energy is the sum of the site energies,
    divided by the number of atoms when the model is intensive (its prediction is eV/atom). The composition model (AtomRef)
    adds one number per element. So the per-atom energy here is mlp(site features) + the AtomRef weight of its element, and the
    total energy is their sum over the atoms (what chgnet reports as e * natoms).
  * cutoff = atom_graph_cutoff (6 A: the default, and what the pretrained 0.3.0 weights in the package use; the bond graph
    cutoff is 3 A), num_layers = n_conv = 4, so the ghost depth is n_conv * cutoff = 24 A.
Not supported: a model that pools before the MLP (mlp_first False), a model with a graph attention readout.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .base import MLIPAdapter
from .triplets import ordered_edge_pairs


def load_chgnet_model(path=None) -> torch.nn.Module:
    """A CHGNet model: from a file (CHGNet.from_file) or, without a path, the pretrained weights that ship inside the chgnet
    package (CHGNet.load(): read from the installed package, nothing is downloaded)."""
    from chgnet.model import CHGNet
    if path:
        from pathlib import Path
        if not Path(path).is_file():
            raise ValueError(f"CHGNet model file not found: {path}")
        model = CHGNet.from_file(str(path))
    else:
        model = CHGNet.load()
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class _RadialBessel(torch.nn.Module):
    """chgnet.model.basis.RadialBessel with a TorchScript-compatible forward (its own has keyword-only arguments)."""

    def __init__(self, source):
        super().__init__()
        self.inv_cutoff = float(source.inv_cutoff)
        self.norm_const = float(source.norm_const)
        self.frequencies = torch.nn.Parameter(source.frequencies.detach().clone(), requires_grad=False)
        smooth = source.smooth_cutoff
        self.has_smooth = smooth is not None
        self.cutoff = float(smooth.cutoff) if smooth is not None else 1.0
        self.p = int(smooth.p) if smooth is not None else 0
        self.a = float(smooth.a) if smooth is not None else 0.0
        self.b = float(smooth.b) if smooth is not None else 0.0
        self.c = float(smooth.c) if smooth is not None else 0.0

    def forward(self, length: Tensor) -> Tensor:
        dist = length[:, None]
        out = self.norm_const * torch.sin(self.frequencies * (dist * self.inv_cutoff)) / dist
        if self.has_smooth:
            r_scaled = dist / self.cutoff
            if self.p != 0:
                envelope = 1 + self.a * r_scaled ** self.p + self.b * r_scaled ** (self.p + 1) + self.c * r_scaled ** (self.p + 2)
                smooth = torch.where(r_scaled < 1, envelope, torch.zeros_like(r_scaled))
            else:
                smooth = torch.ones_like(r_scaled)
            out = smooth * out
        return out


class _AngleFourier(torch.nn.Module):
    """chgnet's AngleEncoder: the cosine of the angle between two unit vectors, expanded in a Fourier series."""

    def __init__(self, source):
        super().__init__()
        fourier = source.fourier_expansion
        self.order = int(fourier.order)
        self.frequencies = torch.nn.Parameter(fourier.frequencies.detach().clone(), requires_grad=False)

    def forward(self, bond_i: Tensor, bond_j: Tensor) -> Tensor:
        cosine = torch.sum(bond_i * bond_j, dim=1) * (1 - 1e-6)     # 1 - 1e-6: acos stays finite
        angle = torch.acos(cosine)
        result = angle.new_zeros(angle.shape[0], 1 + 2 * self.order)
        result[:, 0] = 1.0 / (2.0 ** 0.5)
        phase = torch.outer(angle, self.frequencies.to(angle.dtype))
        result[:, 1: self.order + 1] = torch.sin(phase)
        result[:, self.order + 1:] = torch.cos(phase)
        return result / (3.141592653589793 ** 0.5)


def _aggregate(data: Tensor, owners: Tensor, average: bool = True, num_owner: Optional[int] = None) -> Tensor:
    """chgnet.model.functions.aggregate with ordinary (not keyword-only) arguments, which TorchScript can compile. The same
    computation: sum (or average) the rows of `data` by owner, with `num_owner` rows in the result when it is given."""
    bin_count = torch.bincount(owners)
    bin_count = torch.where(bin_count != 0, bin_count, torch.ones_like(bin_count))
    if num_owner is not None:
        if bin_count.shape[0] != num_owner:
            difference = num_owner - bin_count.shape[0]
            bin_count = torch.cat([bin_count, torch.ones(difference, dtype=bin_count.dtype, device=bin_count.device)])
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, owners, data)
    if average:
        output = (output.T / bin_count).T
    return output


def _use_scriptable_aggregate() -> None:
    """The conv layers of chgnet call `aggregate` through their module's globals; point them at the scriptable twin."""
    from chgnet.model import functions, layers
    functions.aggregate = _aggregate
    layers.aggregate = _aggregate


class _Round(torch.nn.Module):
    """One interaction block of CHGNet._compute: an atom conv, then (when the block has them) a bond conv and an angle update.
    A module of its own because TorchScript can index a ModuleList only with literals, and iterates over it with enumerate."""

    def __init__(self, atom_conv, bond_conv, angle_update):
        super().__init__()
        self.atom_conv = atom_conv
        self.bond_conv = bond_conv
        self.angle_update = angle_update

    def forward(self, atom_feas: Tensor, bond_feas: Tensor, angle_feas: Tensor, bond_weights_ag: Tensor,
                bond_weights_bg: Tensor, atom_graph: Tensor, directed2undirected: Tensor, bond_graph: Tensor,
                has_angles: bool) -> Tuple[Tensor, Tensor, Tensor]:
        atom_feas = self.atom_conv(atom_feas=atom_feas, bond_feas=bond_feas, bond_weights=bond_weights_ag,
                                   atom_graph=atom_graph, directed2undirected=directed2undirected)
        if has_angles:
            if self.bond_conv is not None:
                bond_feas = self.bond_conv(atom_feas=atom_feas, bond_feas=bond_feas, bond_weights=bond_weights_bg,
                                           angle_feas=angle_feas, bond_graph=bond_graph)
                if self.angle_update is not None:
                    angle_feas = self.angle_update(atom_feas=atom_feas, bond_feas=bond_feas, angle_feas=angle_feas,
                                                   bond_graph=bond_graph)
        return atom_feas, bond_feas, angle_feas


def _float_dropout(module: torch.nn.Module) -> None:
    """TorchScript wants a float p; nn.Dropout(0) was built with an int."""
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = float(child.p)


class CHGNetAdapter(MLIPAdapter):
    model_family = "chgnet"
    supports_shifts = True
    # Pruning only, no layer exchange: the bond and angle states live on the edges and each bond belongs to both its atoms, so what
    # a rank would have to send between the layers are states of edges, not node features (docs/mlip_converter_models.md).
    # The atoms are updated by sums over the edges centered on them (edge_index[0]); an edge whose center is farther than
    # (num_layers - 1) cutoffs from the owned atoms does not reach them (checked: tests/mlip/test_layer_exchange_more_models.py).
    edge_receiver_row = 0

    def __init__(self, model: torch.nn.Module, model_name: str = "chgnet"):
        from chgnet.model import CHGNet
        if not isinstance(model, CHGNet):
            raise ValueError(f"CHGNet: expected a chgnet CHGNet model, got {type(model).__name__}")
        if not model.mlp_first:
            raise ValueError("CHGNet: models that pool before the readout MLP (mlp_first=False) are not supported")
        if model.composition_model is not None and not model.composition_model.fitted:
            raise ValueError("CHGNet: the composition model is not fitted")
        _use_scriptable_aggregate()
        n_elements = model.atom_embedding.embedding.num_embeddings
        super().__init__(list(range(1, n_elements + 1)))
        self.model_name = model_name
        self.cutoff = float(model.graph_converter.atom_graph_cutoff)
        self.bond_cutoff = float(model.graph_converter.bond_graph_cutoff)
        self.num_layers = int(model.n_conv)

        self.atom_embedding = model.atom_embedding
        self.bond_embedding = model.bond_embedding
        self.angle_embedding = model.angle_embedding
        self.bond_weights_ag = model.bond_weights_ag
        self.bond_weights_bg = model.bond_weights_bg
        self.rbf_ag = _RadialBessel(model.bond_basis_expansion.rbf_expansion_ag)
        self.rbf_bg = _RadialBessel(model.bond_basis_expansion.rbf_expansion_bg)
        self.angle_basis = _AngleFourier(model.angle_basis_expansion)
        # CHGNet._compute: n_conv - 1 blocks (atom conv, bond conv, angle update), then a last atom conv
        atom_layers = list(model.atom_conv_layers)
        rounds = []
        for k, atom_layer in enumerate(atom_layers[:-1]):
            bond_layer = model.bond_conv_layers[k] if k < len(model.bond_conv_layers) else None
            angle_layer = model.angle_layers[k] if k < len(model.angle_layers) else None
            rounds.append(_Round(atom_layer, bond_layer, angle_layer))
        self.rounds = torch.nn.ModuleList(rounds)
        self.last_atom_conv = atom_layers[-1]
        self.readout_norm = model.readout_norm if model.readout_norm is not None else torch.nn.Identity()
        self.mlp = model.mlp
        _float_dropout(self)
        weights = (model.composition_model.fc.weight.detach().reshape(-1).clone()
                   if model.composition_model is not None else torch.zeros(n_elements))
        self.register_buffer("element_energy", weights)

    def package_versions(self) -> Dict[str, Optional[str]]:
        import chgnet
        return {"chgnet": chgnet.__version__}

    @property
    def reference_elements(self):
        return [8, 1] if 1 in self.elements and 8 in self.elements else self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward_with_shifts(pos, species, edge_index, torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device))

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        z = species + 1
        center = edge_index[0]
        neighbor = edge_index[1]
        n_edges = center.shape[0]

        vec = pos[center] - pos[neighbor] - edge_shift         # chgnet: center - neighbor; the shift is added to pos[neighbor] - pos[center]
        length = torch.norm(vec, dim=1)
        unit = vec / length[:, None]

        # undirected bonds: the pair of a directed edge and its reverse (the same two atoms, the opposite shift): the key of an edge
        # is its two atoms in index order and the shift as seen from the first of them (a bond of an atom with its own image has
        # the same atom twice: the shift with a positive first nonzero component)
        negative = (edge_shift[:, 0] < 0) | ((edge_shift[:, 0] == 0) & (edge_shift[:, 1] < 0)) | (
            (edge_shift[:, 0] == 0) & (edge_shift[:, 1] == 0) & (edge_shift[:, 2] < 0))
        swap = (center > neighbor) | ((center == neighbor) & negative)
        first = torch.where(swap, neighbor, center)
        second = torch.where(swap, center, neighbor)
        key_shift = torch.where(swap[:, None], -edge_shift, edge_shift)
        key = torch.cat([first[:, None].to(pos.dtype), second[:, None].to(pos.dtype), key_shift], dim=1)
        unique_keys, directed2undirected = torch.unique(key, sorted=True, return_inverse=True, dim=0)
        n_undirected = unique_keys.shape[0]
        # one directed edge speaks for each undirected bond (its length is taken by an index, so the gradient reaches the
        # positions once; a scatter of the lengths would hand the gradient to both directed edges)
        edge_number = torch.arange(n_edges, device=pos.device)
        undirected2directed = torch.full((n_undirected,), n_edges, dtype=edge_number.dtype, device=pos.device)
        undirected2directed = undirected2directed.scatter_reduce(0, directed2undirected, edge_number, "amin", include_self=True)
        undirected_length = length[undirected2directed]

        # bond graph: all ordered pairs of different directed edges at one center, both inside the bond cutoff
        left_edge, right_edge = ordered_edge_pairs(center, length, self.bond_cutoff, n_atoms)
        bond_graph = torch.stack([center[left_edge], directed2undirected[left_edge], directed2undirected[right_edge]], dim=1)

        # features
        atom_graph = torch.stack([center, neighbor], dim=1)
        atom_feas = self.atom_embedding(z - 1)
        bond_basis_ag = self.rbf_ag(undirected_length)
        bond_basis_bg = self.rbf_bg(undirected_length)
        bond_feas = self.bond_embedding(bond_basis_ag)
        bond_weights_ag = self.bond_weights_ag(bond_basis_ag)
        bond_weights_bg = self.bond_weights_bg(bond_basis_bg)
        angle_feas = self.angle_embedding(self.angle_basis(unit[left_edge], unit[right_edge]))

        has_angles = bond_graph.shape[0] > 0
        for block in self.rounds:
            atom_feas, bond_feas, angle_feas = block(atom_feas, bond_feas, angle_feas, bond_weights_ag, bond_weights_bg,
                                                     atom_graph, directed2undirected, bond_graph, has_angles)
        atom_feas = self.last_atom_conv(atom_feas=atom_feas, bond_feas=bond_feas, bond_weights=bond_weights_ag,
                                        atom_graph=atom_graph, directed2undirected=directed2undirected)
        atom_feas = self.readout_norm(atom_feas)
        site_energy = self.mlp(atom_feas).reshape(-1)
        return site_energy + self.element_energy.to(site_energy.dtype)[z - 1]
