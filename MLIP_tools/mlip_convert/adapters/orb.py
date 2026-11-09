"""ORB adapter (orb-models 0.5.5, orb-v3 conservative models; as read from its source).

What orb-models does, which this adapter relies on:
  * The architecture is built by a function of orb_models.forcefield.pretrained (ORB_PRETRAINED_MODELS, e.g.
    "orb-v3-conservative-inf-omat") and the checkpoint is its state dict. The name is taken from the checkpoint's file name, or from
    --head. Only ConservativeForcefieldRegressor models are supported: their forces are the gradient of the energy.
  * MoleculeGNS.forward: edge vectors pos[receivers] - pos[senders] + shift, edge features from a radial basis and spherical harmonics
    (times a cutoff envelope), atom embedding by Z, an encoder, num_message_passing_steps attention interaction layers in which the
    edges keep their own latent state and a node is updated from the sums over the edges it receives and the edges it sends, both.
    The standard edge list is that graph with senders = edge_index[0], receivers = edge_index[1].
  * The energy head does not sum per-atom energies: it averages the node features of all atoms of the system first and applies its MLP
    to that mean (predict_atom_avg), E = N * (std * mlp(mean) + mean_shift) + sum of linear references per element; the ZBL pair
    repulsion is added as the MEAN over the atoms of the per-atom sums over sent edges (ZBLBasis, node_aggregation "mean"), with the
    atomic numbers taken as argmax(one_hot) + 1 (both as in orb-models). The energy therefore reads the whole system at once: this
    adapter sets whole_system, the owned atoms of a call are the whole system, and dsMD never splits the model over ranks.
  * Per atom (owned): std * mlp(mean) + mean_shift + reference(Z) + V_i / N; the images get zero.
  * omol models take the total charge and spin through a conditioner: --charge and --spin (both, as orb-models asks).
  * The graph of orb-models keeps at most max_num_neighbors neighbors per atom; the standard edge list has every pair within the
    cutoff. Only the "inf" models (120 neighbors, more than an atom has within 6 A at ordinary densities) give the same graph; models
    with a smaller limit (the "20" models) are refused.
Not supported (refused): direct-force models (orb-v2, orb-v3 direct, the D3/D4 models), neighbor limits below 120.
orb-models' modules do not script (TorchScript): --backend pytorch only.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from torch import Tensor

from .base import MLIPAdapter

MIN_NEIGHBORS = 120


def _architecture_name(path, architecture: Optional[str]) -> str:
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
    if architecture:
        if architecture not in ORB_PRETRAINED_MODELS:
            raise ValueError(f"--head '{architecture}' is not an ORB model name; known: {', '.join(ORB_PRETRAINED_MODELS)}")
        return architecture
    name = Path(path).name
    found = [key for key in ORB_PRETRAINED_MODELS if key in name]
    if not found:
        raise ValueError(f"cannot tell the ORB model from the file name '{name}': name it with --head (e.g. orb-v3-conservative-inf-omat)")
    return max(found, key=len)


class ORBAdapter(MLIPAdapter):
    model_family = "orb"
    supports_shifts = True
    whole_system = True
    scriptable = False

    def __init__(self, model: torch.nn.Module, model_name: str, charge: Optional[float] = None, spin: Optional[float] = None,
                 versions: Optional[Dict[str, Optional[str]]] = None):
        from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
        if not isinstance(model, ConservativeForcefieldRegressor):
            raise ValueError(f"ORB: only conservative models are supported (forces = -dE/dx); this is a {type(model).__name__} that "
                             "predicts forces directly")
        config = model.system_config
        if config is None or config.max_num_neighbors < MIN_NEIGHBORS:
            raise ValueError(f"ORB: this model keeps at most {getattr(config, 'max_num_neighbors', '?')} neighbors per atom, so its graph "
                             f"is not all pairs within the cutoff; use an 'inf' model ({MIN_NEIGHBORS} neighbors)")
        gns = model.model
        if gns.expects_atom_type_embedding or not gns.use_embedding:
            raise ValueError("ORB: only models with an atom-type embedding table are supported")
        super().__init__(list(range(1, 118)))
        self.model_name = model_name
        self.cutoff = float(config.radius)
        self.num_layers = len(gns.gnn_stacks)
        self.gns = gns
        head = model.heads["energy"]
        self.energy_mlp = head.mlp
        self.atom_avg = bool(head.atom_avg)
        self.energy_std = float(torch.sqrt(head.normalizer.bn.running_var).reshape(-1)[0])
        self.energy_mean = float(head.normalizer.bn.running_mean.reshape(-1)[0])
        self.register_buffer("reference", head.reference.linear.weight.detach().reshape(-1)[1:118].clone())   # per species (Z = i + 1)
        self.register_buffer("atomic_numbers", torch.tensor(self.elements, dtype=torch.int64))

        self.pair_repulsion = bool(model.pair_repulsion)
        zbl = model.pair_repulsion_fn if self.pair_repulsion else None
        if zbl is not None and zbl.node_aggregation not in ("mean", "sum"):
            raise ValueError(f"ORB: ZBL aggregation '{zbl.node_aggregation}' is not supported")
        self.zbl_mean = zbl is None or zbl.node_aggregation == "mean"
        for name in ("c", "d", "covalent_radii", "a_exp", "a_prefactor"):
            self.register_buffer("zbl_" + name, getattr(zbl, name).detach().clone() if zbl is not None else torch.zeros(1))
        self.zbl_p = float(zbl.p) if zbl is not None else 6.0

        self.has_condition = gns.conditioner is not None
        if self.has_condition:
            if charge is None or spin is None:
                raise ValueError("this ORB model takes the total charge and spin of the system: give both --charge and --spin")
            system = SimpleNamespace(system_features={"total_charge": torch.tensor([float(charge)]), "total_spin": torch.tensor([float(spin)])},
                                     n_node=torch.tensor([1]), n_edge=torch.tensor([1]))
            with torch.no_grad():
                nodes, edges = gns.conditioner(system)
            self.register_buffer("cond_nodes", nodes.reshape(-1).clone() if nodes is not None else torch.zeros(0))
            self.register_buffer("cond_edges", edges.reshape(-1).clone() if edges is not None else torch.zeros(0))
        else:
            if charge is not None or spin is not None:
                raise ValueError("this ORB model has no charge or spin input: do not pass --charge or --spin")
            self.register_buffer("cond_nodes", torch.zeros(0))
            self.register_buffer("cond_edges", torch.zeros(0))
        self._versions = versions or {}
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path, architecture: Optional[str] = None, charge: Optional[int] = None,
                        spin: Optional[int] = None) -> "ORBAdapter":
        if not Path(path).is_file():
            raise ValueError(f"ORB checkpoint not found: {path}")
        from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
        name = _architecture_name(path, architecture)
        model = ORB_PRETRAINED_MODELS[name](weights_path=str(Path(path).resolve()), device="cpu", precision="float32-highest", compile=False)
        try:
            from importlib.metadata import version
            orb_version = version("orb-models")
        except Exception:       # noqa: BLE001
            orb_version = None
        return cls(model.eval(), Path(path).stem, charge=charge, spin=spin, versions={"orb-models": orb_version})

    def package_versions(self) -> Dict[str, Optional[str]]:
        return dict(self._versions)

    @property
    def reference_elements(self):
        return [8, 1]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        no_shift = torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device)
        return self.forward_whole_system(pos, species, edge_index, no_shift, pos.shape[0])

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        return self.forward_whole_system(pos, species, edge_index, edge_shift, pos.shape[0])

    def forward_whole_system(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        from orb_models.forcefield.nn_util import get_cutoff
        gns = self.gns
        z = self.atomic_numbers[species]
        senders = edge_index[0]
        receivers = edge_index[1]
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        vectors = pos[receivers] - pos[senders] + edge_shift
        lengths = vectors.norm(dim=1)

        # MoleculeGNS.featurize_edges / featurize_nodes
        angular = gns.angular_transform(vectors)
        rbfs = gns.rbf_transform(lengths)
        if gns.outer_product_with_cutoff:
            edge_features = get_cutoff(lengths) * (rbfs[:, :, None] * angular[:, None, :]).view(n_edges, gns.edge_embed_size)
        else:
            edge_features = torch.cat([rbfs, angular], dim=1)
        nodes, edges = gns._encoder(gns.atom_emb(z), edge_features)

        cutoff = get_cutoff(lengths)
        cond_nodes = self.cond_nodes.to(pos.dtype).expand(n_atoms, -1) if self.cond_nodes.numel() else None
        cond_edges = self.cond_edges.to(pos.dtype).expand(n_edges, -1) if self.cond_edges.numel() else None
        for gnn in gns.gnn_stacks:
            nodes, edges = gnn(nodes, edges, senders, receivers, cutoff, cond_nodes=cond_nodes, cond_edges=cond_edges)

        # EnergyHead on the whole system (the owned atoms), split evenly over the atoms
        owned = nodes[:n_owned]
        pooled = owned.mean(dim=0, keepdim=True) if self.atom_avg else owned.sum(dim=0, keepdim=True)
        value = self.energy_mlp(pooled).reshape(()) * self.energy_std + self.energy_mean
        per_atom = value if self.atom_avg else value / n_owned
        energy = per_atom + self.reference.to(pos.dtype)[species[:n_owned]]

        if self.pair_repulsion:
            # ZBLBasis.forward, with orb-models' atomic numbers argmax(one_hot) + 1
            z_u = (z[senders] + 1).to(pos.dtype)
            z_v = (z[receivers] + 1).to(pos.dtype)
            a = self.zbl_a_prefactor * 0.529 / (torch.pow(z_u, self.zbl_a_exp) + torch.pow(z_v, self.zbl_a_exp))
            r_over_a = lengths / a
            phi = torch.sum(self.zbl_c * torch.exp(-self.zbl_d * r_over_a.unsqueeze(0)), dim=0)
            v_raw = 14.3996 * z_u * z_v / lengths * phi
            r_max = self.zbl_covalent_radii[z[senders] + 1] + self.zbl_covalent_radii[z[receivers] + 1]
            ratio = lengths / r_max
            p = self.zbl_p
            envelope = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(ratio, p) + p * (p + 2.0) * torch.pow(ratio, p + 1.0)
                        - (p * (p + 1.0) / 2.0) * torch.pow(ratio, p + 2.0)) * (lengths < r_max).to(pos.dtype)
            v_node = torch.zeros(n_atoms, dtype=pos.dtype, device=pos.device).index_add_(0, senders, 0.5 * v_raw * envelope)
            energy = energy + (v_node[:n_owned] / n_owned if self.zbl_mean else v_node[:n_owned])
        return torch.cat([energy, torch.zeros(n_atoms - n_owned, dtype=energy.dtype, device=energy.device)])
