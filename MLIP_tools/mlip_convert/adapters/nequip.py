"""NequIP adapter (design 3.4, 11장 확인 결과 nequip 0.19.1).

What the installed nequip does, which this adapter relies on:
  * a model is a nequip.nn.GraphModel. Called as model(data) with the dict
      pos [N, 3], atom_types [N] int64 (index into model.type_names), edge_index [2, E] int64
    (cell [1, 3, 3] and edge_cell_shift [E, 3] are optional: without them the edge vectors are plain position differences,
    which is what ghost atoms need). edge_index[0] is the center (dst), edge_index[1] the neighbor (src), exactly the
    standard convention of this converter, so edge_index is passed as it is.
  * out["atomic_energy"] [N, 1] holds the per-atom energies including the per-type scale and shift; out["total_energy"] is
    their sum over all atoms (ghosts included), so it is not used: MDWrapper sums the owned atoms and differentiates.
  * cutoff = metadata["r_max"], num_layers = the number of ConvNetLayer modules (message-passing layers), so
    ghost_cutoff = num_layers * r_max. Strictly local models (Allegro) count as one layer.
  * type_names are the model's own atom type names; here they must be chemical symbols (or pass the atomic numbers).
  * TorchScript: nequip compiles with e3nn.util.jit.script (nequip/model/inference_models/torchscript.py), as MACE does.
  * Saved models: nequip.model.saved_models.load_utils.load_saved_model reads .ckpt and .nequip.zip files. (nequip itself
    deprecates TorchScript for PyTorch >= 2.10.)
"""
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from .base import MLIPAdapter


def ensure_global_state() -> None:
    """nequip refuses to build or load models before its global state is initialized."""
    from nequip.utils.global_state import global_state_initialized, set_global_state
    if not global_state_initialized():
        set_global_state(allow_tf32=False)


def load_nequip_model(path, dtype: torch.dtype = torch.float64, accel: str = "e3nn") -> torch.nn.Module:
    """A nequip GraphModel from a .ckpt or .nequip.zip file (nequip's own loader), or from a plain torch.save of the model.
    accel "openeq" / "cueq" applies nequip's own `enable_OpenEquivariance` / `enable_CuEquivariance` model modifier (found on
    the model: NequIP has both; Allegro has other ones, 11.3); that builds the kernels, which need a CUDA device and the
    kernel package."""
    from_nequip_net = str(path).startswith("nequip.net:")      # an id nequip downloads itself, only when it is given
    if not from_nequip_net and not Path(path).is_file():
        raise ValueError(f"NequIP model file not found: {path}")
    if accel not in ("e3nn", "cueq", "openeq"):
        raise ValueError(f"NequIP supports --accel e3nn, cueq and openeq, not '{accel}'")
    ensure_global_state()
    if from_nequip_net or str(path).endswith((".ckpt", ".nequip.zip")):
        from nequip.model.saved_models.load_utils import load_saved_model
        model = load_saved_model(str(path))
    else:
        model = torch.load(str(path), map_location="cpu", weights_only=False)
    if accel != "e3nn":
        from nequip.model.modify_utils import modify
        names = {"openeq": ("enable_OpenEquivariance", "OpenEquivariance", "openequivariance"),
                 "cueq": ("enable_CuEquivariance", "cuEquivariance", "cuequivariance cuequivariance-torch")}
        modifier, label, packages = names[accel]
        try:
            model = modify(model, [{"modifier": modifier}])
        except ImportError as error:       # nequip imports the kernel package only when it builds the kernels
            raise ValueError(f"--accel {accel} needs the Python package(s) for {label} (pip install {packages}): "
                             f"{error}") from error
        except RuntimeError as error:      # e.g. "not a registered model modifier"
            raise ValueError(f"this model cannot use {label}: {error}") from error
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _atomic_numbers_from_type_names(type_names: Sequence[str]):
    from ase.data import atomic_numbers
    z = []
    for name in type_names:
        if name not in atomic_numbers:
            raise ValueError(f"NequIP type name '{name}' is not a chemical symbol; pass the atomic number of every type "
                             f"with --elements (in the order of the model's type_names: {list(type_names)})")
        z.append(int(atomic_numbers[name]))
    return z


PAIR_POTENTIALS = ("ZBL", "LennardJones", "SimpleLennardJones")     # nequip.nn.pair_potential: edge energies summed onto the center atom


def _is_module(module: torch.nn.Module, name: str) -> bool:
    """Whether the module is an instance of the nequip class `name`. A model read from a .nequip.zip package has its own copy of the
    nequip classes (nequip loads it with torch.package), so isinstance with the installed class is False for it."""
    return any(cls.__name__ == name for cls in type(module).__mro__)


class _NequIPLayer(torch.nn.Module):
    """One ConvNetLayer of a NequIP model (nequip's own modules, the same weights) for a rank that owns n_owned atoms and has ghost
    atoms after them: InteractionBlock.forward as nequip runs it for LAMMPS ML-IAP, with the ghost exchange written as the op
    dsmd::halo_exchange. Features after the first layer exist for the owned atoms only; the senders of the messages (ghost rows) get
    them from the exchange, which comes after the first linear layer and the normalization, as in nequip. The first layer starts
    from the atom types, which every rank has for its ghost atoms, so it computes all rows and does not exchange."""

    def __init__(self, layer: torch.nn.Module, first: bool):
        super().__init__()
        block = layer.conv
        self.sc = block.sc
        self.linear_1 = block.linear_1
        # the division by the average number of neighbors: a module (per type) in nequip 0.16 and later, a plain factor before (the
        # NequIP-OAM packages were made with 0.15 or earlier, and a package carries the code of its own nequip)
        self.norm = getattr(block, "avg_num_neighbors_norm", None)
        self.alpha: Optional[float] = None
        if self.norm is None:
            if not hasattr(block, "scatter_norm_factor"):
                raise ValueError("layer exchange: this NequIP interaction block has neither avg_num_neighbors_norm nor scatter_norm_factor")
            self.alpha = block.scatter_norm_factor
        self.edge_mlp = block.edge_mlp
        self.tp_scatter = block.tp_scatter
        self.linear_2 = block.linear_2
        self.nonlin = layer.equivariant_nonlin
        self.resnet = bool(layer.resnet)
        self.first = first

    def forward(self, x: Tensor, node_attrs: Tensor, atom_types: Tensor, edge_attrs: Tensor, edge_embedding: Tensor,
                edge_index: Tensor, n_owned: int, n_ghost: int) -> Tensor:
        x_owned = x[:n_owned]
        if self.sc is not None:
            sc = self.sc(x_owned, node_attrs[:n_owned])
        if self.first:
            h = self.linear_1(x)
        else:
            h = self.linear_1(x_owned)
        if self.norm is not None:
            normed: Dict[str, Tensor] = {"node_features": h, "atom_types": atom_types[:h.shape[0]]}
            h = self.norm(normed)["node_features"]
        if self.alpha is not None:
            h = self.alpha * h
        if not self.first:
            h = torch.ops.dsmd.halo_exchange(h, n_ghost)
        out = self.tp_scatter(h, edge_attrs, self.edge_mlp(edge_embedding), edge_index[0], edge_index[1])[:n_owned]
        out = self.linear_2(out)
        if self.sc is not None:
            out = out + sc
        out = self.nonlin(out)
        if self.resnet:
            out = x_owned + out
        return out


class NequIPAdapter(MLIPAdapter):
    model_family = "nequip"
    e3nn_compile = True
    supports_shifts = True
    # NequIP (message passing): every ConvNetLayer sums its messages onto the center atom, edge_index[0] (nequip's edge_dst). An
    # Allegro model, read by this class too, is strictly local and has no ConvNetLayer: __init__ switches both off for it.
    supports_layer_exchange = True
    edge_receiver_row = 0

    def __init__(self, model: torch.nn.Module, elements: Optional[Sequence[int]] = None, model_name: str = "nequip",
                 accel: str = "e3nn"):
        type_names = list(model.type_names)
        if elements is None:
            elements = _atomic_numbers_from_type_names(type_names)
        elif len(elements) != len(type_names):
            raise ValueError(f"--elements has {len(elements)} entries but the model has {len(type_names)} atom types "
                             f"{type_names}")
        super().__init__([int(z) for z in elements])
        self.accel = accel
        self.model = model
        self.model_name = model_name
        self.type_names_py = type_names
        # nequip builds a model for one dtype (model_dtype) and its modules keep dtype-dependent state: it is not converted.
        self.fixed_dtype_name = str(model.model_dtype).replace("torch.", "")

        layers = sum(_is_module(m, "ConvNetLayer") for m in model.modules())
        self.num_layers = max(layers, 1)
        self.cutoff = float(model.metadata["r_max"])
        if layers == 0:
            # a strictly local model (Allegro): ghost atoms one cutoff deep already, every edge as it was
            self.supports_layer_exchange = False
            self.edge_receiver_row = None

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        import nequip
        return {"nequip": nequip.__version__, "e3nn": e3nn.__version__}

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        data: Dict[str, Tensor] = {"pos": pos, "atom_types": species, "edge_index": edge_index}
        out = self.model(data)
        return out["atomic_energy"].squeeze(-1).to(pos.dtype)

    def enable_layer_exchange(self) -> None:
        """Switches layer exchange on and sorts the modules of the model into what runs before the layers (all atoms), the layers
        (owned atoms, ghost atoms only as senders) and the readout (owned atoms). They share the weights of the model. Needs the
        layout NequIPGNNModel builds: type_embed, spharm, ..., layerN_convnet ..., per_atom_energy_readout, per_type_energy_scale_shift,
        total_energy_sum."""
        sequence = next((m for m in self.model.modules() if _is_module(m, "SequentialGraphNetwork")), None)
        if sequence is None:
            raise ValueError("layer exchange: no SequentialGraphNetwork in this NequIP model")
        names = list(sequence._modules.keys())
        convs = [i for i, name in enumerate(names) if _is_module(sequence._modules[name], "ConvNetLayer")]
        if not convs or convs != list(range(convs[0], convs[0] + len(convs))):
            raise ValueError(f"layer exchange: the convolution layers of this NequIP model are not one block of modules: {names}")
        # after the layers: the readout, the scale and shift, then pair potentials (the ZBL of the NequIP-OAM models: edge energies summed
        # onto the center atom) and the sum over the atoms
        after = names[convs[-1] + 1:]
        pair_names = after[2:-1]
        pair_classes = [type(sequence._modules[name]).__name__ for name in pair_names]
        if (after[:2] != ["per_atom_energy_readout", "per_type_energy_scale_shift"] or after[-1:] != ["total_energy_sum"]
                or any(not any(_is_module(sequence._modules[name], cls) for cls in PAIR_POTENTIALS) for name in pair_names)
                or any(getattr(sequence._modules[name], "per_atom_energy_field", "atomic_energy") != "atomic_energy" for name in pair_names)):
            raise ValueError(f"layer exchange: after its convolution layers this NequIP model has {after} ({pair_classes} between the "
                             f"scale/shift and the sum), not per_atom_energy_readout, per_type_energy_scale_shift, then pair potentials "
                             f"({', '.join(PAIR_POTENTIALS)}) and total_energy_sum: another extra term is not supported")
        from nequip.data import AtomicDataDict
        for literal, key in (("node_features", AtomicDataDict.NODE_FEATURES_KEY), ("node_attrs", AtomicDataDict.NODE_ATTRS_KEY),
                             ("edge_attrs", AtomicDataDict.EDGE_ATTRS_KEY), ("edge_embedding", AtomicDataDict.EDGE_EMBEDDING_KEY),
                             ("atom_types", AtomicDataDict.ATOM_TYPE_KEY), ("atomic_energy", AtomicDataDict.PER_ATOM_ENERGY_KEY)):
            if literal != key:
                raise ValueError(f"layer exchange: this nequip names its data field {key}, not {literal}")
        pre = [sequence._modules[name] for name in names[:convs[0]]]
        layers = [_NequIPLayer(sequence._modules[names[i]], first=(k == 0)) for k, i in enumerate(convs)]
        super().enable_layer_exchange()
        self.pre_layers = torch.nn.ModuleList(pre)
        self.conv_layers = torch.nn.ModuleList(layers)
        self.energy_readout = sequence._modules["per_atom_energy_readout"]
        self.energy_scale_shift = sequence._modules["per_type_energy_scale_shift"]
        self.pair_modules = torch.nn.ModuleList([sequence._modules[name] for name in pair_names])

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """The modules of the model with the node features of the owned atoms sent to their ghost copies before every layer but
        the first; only the owned atoms are computed after the first layer. Per atom energies [N], zero for the ghost atoms."""
        n_ghost = pos.shape[0] - n_owned
        data: Dict[str, Tensor] = {
            "pos": pos,
            "atom_types": species,
            "edge_index": edge_index,
            "cell": torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0),
            "edge_cell_shift": edge_shift,
        }
        for module in self.pre_layers:
            data = module(data)
        x = data["node_features"]
        for layer in self.conv_layers:
            x = layer(x, data["node_attrs"], species, data["edge_attrs"], data["edge_embedding"], edge_index, n_owned, n_ghost)
        owned: Dict[str, Tensor] = {"node_features": x, "atom_types": species[:n_owned]}
        owned = self.energy_readout(owned)
        owned = self.energy_scale_shift(owned)
        energy = owned["atomic_energy"].squeeze(-1)
        if len(self.pair_modules) > 0:
            # a pair potential sums the energies of the edges onto their center atom, whose edges all exist on this rank for the owned
            # atoms; it runs over every row of the (ghost) edge list and the owned rows are taken
            data["atomic_energy"] = torch.zeros((pos.shape[0], 1), dtype=energy.dtype, device=energy.device)
            for pair in self.pair_modules:
                data = pair(data)
            energy = energy + data["atomic_energy"].squeeze(-1)[:n_owned]
        energy = energy.to(pos.dtype)
        return torch.cat([energy, torch.zeros(n_ghost, dtype=energy.dtype, device=energy.device)])

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        # nequip: edge vector = pos[j] - pos[i] + edge_cell_shift @ cell (with_edge_vectors_). With the unit cell the
        # "cell shift" is the Cartesian shift itself.
        data: Dict[str, Tensor] = {
            "pos": pos,
            "atom_types": species,
            "edge_index": edge_index,
            "cell": torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0),
            "edge_cell_shift": edge_shift,
        }
        out = self.model(data)
        return out["atomic_energy"].squeeze(-1).to(pos.dtype)
