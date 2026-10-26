"""BAM adapter (BAM-torch 0.2.1, the RACE model: https://github.com/myung-group/BAM-torch).

A BAM checkpoint is one .pkl file (torch.save of a dict). Its 'input.json' is the training configuration, 'params' the weights,
'uniq_element' maps an atomic number to the model's species index, 'enr_avg_per_element' is the energy reference of every
species and 'valid_scale_shift' the per-species energy correction fitted on the validation set. The model is built the way BAM's
own trainer builds it (BaseTrainer.set_model, called with the checkpoint's configuration) and the weights are loaded into it.

What the installed bam_torch does, which this adapter relies on:
  * RACE.forward(data, backprop, compute_displacement) takes a dict
      positions [N, 3]       requires grad
      species [N]            model species index; node_attrs [N, num_species] is the one-hot of it, in the model's dtype
      edge_index [2, E]      the vector of edge e is positions[edge_index[1]] - positions[edge_index[0]] + shift, as in MACE
      edges [E, 3], num_edges, cell [1, 3, 3]   BAM turns edges (fractional) into shifts with the cell: with cell = identity,
                             `edges` is the Cartesian shift itself (zeros when the periodic images are atoms)
      batch [N], ptr [2]     one graph
    and returns node_energy [N] (the sum over the layers of the readouts, an energy in eV before the reference is added).
  * the total energy BAM's calculators report is the sum over atoms of  node_energy + enr_avg_per_element[species] +
    valid_scale_shift[species]. The two per-species numbers are added here, to every atom.
  * forces are taken by MDWrapper from the owned-atom sum: the model's own force output is switched off (regress_forces).
  * cutoff = the model's cutoff, num_layers = nlayers, so the ghost depth is nlayers x cutoff (BAM-MP-core: 5 x 6 = 30 A).
Kernels (--accel): e3nn (the default), openeq (OpenEquivariance: BAM's "fast" interaction block only, weights as e3nn) and cueq
(cuEquivariance: its own weight layout, so only a checkpoint trained with it loads). openeq and cueq are built the way BAM builds
them (cueq_config / oeq_config) and need a GPU for their fast kernels; the layered path was written against the e3nn modules and has
only been run with e3nn so far.
Not supported: models other than "race" (the multihead "race_unified", BAM's own "mace"); OpenEquivariance's fused convolution
(oeq_conv_fusion); a checkpoint whose valid_scale_shift is one number for the whole structure instead of one per
species: it is a constant that does not change forces, and is left out (the energy is then off by that constant).
"""
import copy
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

# BAM's own tools set this for e3nn constants and checkpoints; it only changes how files we are given are loaded.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from torch import Tensor

from .base import MLIPAdapter

SUPPORTED_MODELS = ("race",)


def _e3nn_generated_code_is_scripted() -> bool:
    try:
        import e3nn
    except ImportError:
        return True
    return bool(e3nn.get_optimization_defaults()["jit_script_fx"])


def read_bam_checkpoint(path) -> Dict:
    """The checkpoint dict of a BAM .pkl file (a pickle: only open files you trust)."""
    if not Path(path).is_file():
        raise ValueError(f"BAM model file not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    missing = [key for key in ("input.json", "params", "uniq_element", "enr_avg_per_element", "valid_scale_shift")
               if not isinstance(checkpoint, dict) or key not in checkpoint]
    if missing:
        raise ValueError(f"{path} is not a BAM checkpoint (.pkl written by bam_torch): it lacks {', '.join(missing)}")
    return checkpoint


def _backend_of(model: torch.nn.Module) -> str:
    """The equivariant kernels a built BAM model really uses: bam_torch falls back to e3nn without an error when the package of the
    requested kernels cannot be imported."""
    roots = {type(module).__module__.split(".")[0] for module in model.modules()}
    if "openequivariance" in roots:
        return "openeq"
    if "cuequivariance_torch" in roots:
        return "cueq"
    return "e3nn"


def build_bam_model(config: Dict, params: Dict[str, Tensor], accel: str = "e3nn", dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """The BAM model of a training configuration with the tensor products of `accel` (e3nn, cueq = cuEquivariance, openeq =
    OpenEquivariance), and the weights loaded into it. The weights of e3nn and OpenEquivariance models are the same tensors (BAM's
    own make_pt_oeq.py loads one into the other); a cuEquivariance model has its own layout, so only a checkpoint trained with
    cuEquivariance can be loaded into one."""
    try:
        from bam_torch.training.base_trainer import BaseTrainer
    except ImportError as error:
        raise ValueError(f"converting a BAM model needs the bam_torch package (git clone https://github.com/myung-group/BAM-torch, "
                         f"pip install -e .): {error}") from error
    name = str(config.get("model", "")).lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"the BAM model type '{name}' is not supported (supported: {', '.join(SUPPORTED_MODELS)})")
    config = copy.deepcopy(config)
    # BAM picks cuEquivariance whenever it is installed, then OpenEquivariance when asked: the kernels are chosen here
    config["cueq_config"] = accel == "cueq"
    config["oeq_config"] = accel == "openeq"
    if accel == "openeq" and str(config.get("interaction_block", "slow")) != "fast":
        raise ValueError(f"--accel openeq: only BAM's 'fast' interaction block can use OpenEquivariance (the 'slow' one runs on e3nn), "
                         f"this checkpoint has interaction_block: {config.get('interaction_block', 'slow')}. The two block kinds have "
                         f"different weights, so a slow model cannot be turned into a fast one: use --accel e3nn")
    previous = torch.get_default_dtype()
    if accel != "e3nn" and dtype is not None:
        torch.set_default_dtype(dtype)      # the kernels are built for the default dtype
    try:
        # set_model only reads json_data and appends to msg: it is called with a stand-in for the trainer, so the training
        # setup (data sets, optimizer, process groups) is not built
        model = BaseTrainer.set_model(SimpleNamespace(json_data=config, msg=""))
    finally:
        torch.set_default_dtype(previous)
    if accel != "e3nn" and _backend_of(model) != accel:
        package = {"cueq": "cuequivariance and cuequivariance-torch (pip install cuequivariance cuequivariance-torch)",
                   "openeq": "openequivariance (pip install openequivariance)"}[accel]
        raise ValueError(f"--accel {accel}: bam_torch built the model with {_backend_of(model)} kernels, not {accel}: it needs the Python "
                         f"package(s) {package} importable in this environment")
    params = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in params.items()}
    try:
        result = model.load_state_dict(params, strict=False)     # strict=False: the buffers of e3nn's tensor products (Wigner 3j) are not weights
    except RuntimeError as error:
        if accel == "cueq":
            raise ValueError(f"--accel cueq: the weights of this checkpoint do not fit a cuEquivariance model (they were not trained with "
                             f"cuEquivariance; use --accel e3nn or openeq): {error}") from error
        raise
    names = {parameter_name for parameter_name, _ in model.named_parameters()}
    missing = [key for key in result.missing_keys if key in names]
    if missing or result.unexpected_keys:
        raise ValueError(f"the checkpoint does not fit the {accel} model: missing weights {missing}, unexpected {list(result.unexpected_keys)}")
    return model


def load_bam_model(path, accel: str = "e3nn", dtype: Optional[torch.dtype] = None):
    """(model in eval mode without gradients, checkpoint dict). accel: e3nn, cueq or openeq."""
    if accel not in ("e3nn", "cueq", "openeq"):
        raise ValueError(f"BAM supports --accel e3nn, cueq and openeq, not '{accel}'")
    checkpoint = read_bam_checkpoint(path)
    model = build_bam_model(checkpoint["input.json"], checkpoint["params"], accel=accel, dtype=dtype)
    model.regress_forces = "false"      # no force output of its own: forces come from the wrapper (neither "auto" nor "direct")
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if accel != "e3nn":
        from .mace import _make_patched_forwards_scriptable    # bam_torch's wrappers patch a forward like mace's: make it scriptable
        _make_patched_forwards_scriptable(model)
    return model, checkpoint


class _RaceLayer(torch.nn.Module):
    """One layer of the RACE model (the interaction block, the product block and the readout of bam_torch's RACE.forward, the same
    modules and weights), for a rank that owns n_owned atoms and has ghost atoms after them. The block of bam_torch works on every
    row it is given: its skip connection and the normalization of the messages would be computed for ghost rows too, and the
    messages summed into rows no edge feeds (only owned atoms receive edges). Here the ghost rows are used where they are needed, as
    senders of messages (linear_up and the gather), and everything else, the skip connection, the sum onto the receivers, the
    product block and the readout, is done on the n_owned rows. The two block kinds of bam_torch differ in the tensor product
    ("slow": the full tensor product, concatenated and weighted; "fast": the weighted channel-wise one). Returns the features of the
    owned atoms after the layer and their energy of the layer."""
    __constants__ = ["slow", "per_layer_x", "rms_norm", "layer_norm"]

    def __init__(self, block: torch.nn.Module, product: torch.nn.Module, readout: torch.nn.Module, slow: bool,
                 linear_x: Optional[torch.nn.Module], per_layer_x: bool, rms_norm: bool):
        super().__init__()
        self.block = block
        self.product = product
        self.readout = readout
        self.slow = slow
        self.per_layer_x = per_layer_x
        self.rms_norm = rms_norm
        self.layer_norm = bool(block.l_separated_layer_norm)
        if per_layer_x:
            self.linear_x = linear_x

    def forward(self, skip_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor,
                edge_cutoff: Optional[Tensor], x_shared: Optional[Tensor], n_owned: int) -> Tuple[Tensor, Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        entering = node_feats[:n_owned]
        skip = self.block.skip_tp_node(entering, skip_attrs[:n_owned])
        if not torch.jit.is_scripting() and self.layer_norm:       # as bam_torch: its layer norm is not part of the scripted model
            node_feats = self.block.separated_layer_norm(node_feats)
        up = self.block.linear_up(node_feats)
        if self.slow:
            mix = self.block.conv_tp_weights(edge_feats)
            if edge_cutoff is not None:
                mix = mix * edge_cutoff
            mji = self.block.conv_tp(up[sender], edge_attrs)
            mji = self.block.concatenate_irreps_tensor(tensors=[up[sender], mji], axis=-1)
            mji = self.block.tensor_regroup_by_irreps(mji)
            mji = self.block.tensor_irreps_array_product(mix, mji)
        else:
            radial = self.block.conv_tp_weights(edge_feats)
            if edge_cutoff is not None:
                radial = radial * edge_cutoff
            mji = self.block.conv_tp(up[sender], edge_attrs, radial)
        message = mji.new_zeros([n_owned] + [size for size in mji.shape[1:]]).index_add_(0, receiver, mji)
        message = message * self.block.inv_sqrt_avg_num_neighbors
        message = self.block.linear_down(message) * self.block.inv_sqrt_avg_num_neighbors
        if self.per_layer_x:
            x = self.linear_x(entering)
            if self.rms_norm:
                x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        else:
            assert x_shared is not None
            x = x_shared
        out = self.product(x_node_feats=x, node_feats=message, sc=skip)
        energies = self.readout(out)
        return out, energies[:, 0]


class _RaceLayered(torch.nn.Module):
    """RACE.forward of bam_torch for a rank with ghost atoms, with the node features of the owned atoms sent to their ghost copies
    before every layer but the first (torch.ops.dsmd.halo_exchange). The edge list has only edges whose receiver is an owned atom.
    Returns the energy [n_owned] of the owned atoms (the model's own, before the per-species reference energy is added)."""
    __constants__ = ["cutoff", "has_species_embedding", "has_skip_projection", "per_layer_x"]

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        if getattr(model, "dens", False):
            raise ValueError("layer exchange: this BAM model has the DeNS heads (dens), which forward_layered does not run")
        self.cutoff = float(model.cutoff)
        self.node_embedding = model.node_embedding
        self.radial_embedding = model.radial_embedding
        self.spherical_harmonics = model.spherical_harmonics
        self.act_fn = model.act_fn
        self.has_species_embedding = model.species_embedding_dim is not None
        if self.has_species_embedding:
            self.species_embedding = model.species_embedding
            self.layer_scales_e = model.layer_scales_e
        self.has_skip_projection = model.skip_species_dim is not None
        if self.has_skip_projection:
            self.skip_species_embedding = model.skip_species_embedding
        self.per_layer_x = bool(model.x_feats_per_layer)
        if not self.per_layer_x:
            self.linear_x = model.linear_x
        layers = []
        for index in range(int(model.nlayers)):
            block = model.interactions[index]
            kind = type(block).__name__
            if kind not in ("ConcatenateRaceInteractionBlock", "RaceInteractionBlock"):
                raise ValueError(f"layer exchange: the BAM interaction block {kind} is not one forward_layered knows")
            if getattr(block, "_use_oeq_conv_fusion", False):
                raise ValueError("layer exchange: the OpenEquivariance fused convolution of BAM is not supported")
            layers.append(_RaceLayer(block, model.products[index], model.readouts[index], kind == "ConcatenateRaceInteractionBlock",
                                     model.linear_x[index] if self.per_layer_x else None, self.per_layer_x,
                                     bool(model.x_feats_rms_norm)))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, pos: Tensor, species: Tensor, node_attrs: Tensor, edge_index: Tensor, edge_shift: Tensor,
                n_owned: int) -> Tensor:
        n_ghost = pos.shape[0] - n_owned
        vectors = (pos[edge_index[1]] - pos[edge_index[0]] + edge_shift) / self.cutoff
        lengths = torch.norm(vectors, dim=1)
        nonzero = torch.nonzero(lengths != 0).squeeze(-1)       # bam_torch drops zero-length edges
        vectors = vectors[nonzero]
        lengths = lengths[nonzero]
        edge_index = edge_index[:, nonzero]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_cutoff: Optional[Tensor] = None
        if self.has_species_embedding:
            edge_cutoff = self.radial_embedding.cutoff_fn(lengths.unsqueeze(1))
        edge_feats = self.radial_embedding(lengths.unsqueeze(1), node_attrs, edge_index, species)
        if self.has_species_embedding:
            species_emb = self.species_embedding(node_attrs)
            edge_feats = torch.cat([edge_feats, species_emb[edge_index[0]], species_emb[edge_index[1]]], dim=-1)
        node_feats = self.node_embedding(node_attrs)
        skip_attrs = node_attrs
        if self.has_skip_projection:
            skip_attrs = self.skip_species_embedding(node_attrs)
        x_shared: Optional[Tensor] = None
        if not self.per_layer_x:
            x_shared = self.linear_x(node_feats[:n_owned])
        energies: List[Tensor] = []
        layer = 0
        for block in self.layers:
            if layer > 0:
                node_feats = torch.ops.dsmd.halo_exchange(node_feats, n_ghost)
            node_feats, energy = block(skip_attrs, node_feats, edge_attrs, edge_feats, edge_index, edge_cutoff, x_shared, n_owned)
            energies.append(energy)
            layer += 1
        node_energy = self.act_fn(torch.stack(energies, dim=-1))
        if self.has_species_embedding:
            node_energy = node_energy * self.layer_scales_e
        return torch.sum(node_energy, dim=-1)

class BAMAdapter(MLIPAdapter):
    model_family = "bam"
    e3nn_compile = True   # export_torchscript runs e3nn.util.jit.compile before scripting
    supports_shifts = True
    supports_layer_exchange = True
    edge_receiver_row = 1       # bam_torch: edge_index[0] = sender, edge_index[1] = receiver (the messages are summed onto it)

    def __init__(self, model: torch.nn.Module, checkpoint: Dict, model_name: str = "bam", accel: str = "e3nn"):
        uniq = {int(z): int(index) for z, index in checkpoint["uniq_element"].items()}
        elements = [z for z, index in sorted(uniq.items(), key=lambda item: item[1])]
        if sorted(uniq.values()) != list(range(len(uniq))):
            raise ValueError(f"the species indices of uniq_element are not 0..{len(uniq) - 1}: {sorted(uniq.values())}")
        super().__init__(elements)
        self.accel = accel
        self.model = model
        self.model_name = model_name
        self.cutoff = float(model.cutoff)
        self.num_layers = int(model.nlayers)
        num_species = int(model.num_species)
        if len(elements) > num_species:
            raise ValueError(f"the checkpoint has {len(elements)} elements but the model {num_species} species")

        # energy added to every atom of a species: the element reference + the validation-set correction
        reference = checkpoint["enr_avg_per_element"]
        correction = checkpoint["valid_scale_shift"]
        atom_ref = torch.zeros(len(elements), dtype=torch.float64)
        for index in range(len(elements)):
            atom_ref[index] = float(reference[index])
        if isinstance(correction, dict):
            for index in range(len(elements)):
                atom_ref[index] += float(correction[index]) if index in correction else 0.0
        else:
            warnings.warn("this BAM checkpoint has one valid_scale_shift for the whole structure, not one per species: it is a "
                          "constant energy offset that does not change the forces, and it is left out", stacklevel=2)
        self.register_buffer("atom_ref", atom_ref)
        self.register_buffer("species_one_hot", torch.eye(num_species))

    def enable_layer_exchange(self) -> None:
        """Switches layer exchange on and builds the layers that forward_layered runs (they share the weights of the model; they
        are not built otherwise, so a model converted without layer exchange is scripted exactly as before)."""
        layered = _RaceLayered(self.model)
        super().enable_layer_exchange()
        self.layered = layered

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        try:
            from importlib.metadata import version
            bam_version = version("bam-torch")
        except Exception:      # noqa: BLE001  (a source checkout without installed metadata)
            bam_version = None
        return {"bam_torch": bam_version, "e3nn": e3nn.__version__}

    @property
    def reference_elements(self):
        if 1 in self.elements and 8 in self.elements:
            return [8, 1]
        return self.elements[:2]

    def _atom_energy(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        data: Dict[str, Tensor] = {
            "positions": pos,
            "species": species,
            "node_attrs": self.species_one_hot.to(pos.dtype)[species],
            "edge_index": edge_index,
            "edges": edge_shift,
            "num_edges": torch.tensor([n_edges], dtype=torch.long),     # a host tensor: the model turns it into a list
            "cell": torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0),
            "batch": torch.zeros(n_atoms, dtype=torch.long, device=pos.device),
            "ptr": torch.arange(2, dtype=torch.long, device=pos.device) * n_atoms,
        }
        out = self.model(data, False, False)
        node_energy = out["node_energy"]
        assert node_energy is not None
        return node_energy.to(pos.dtype) + self.atom_ref.to(pos.dtype)[species]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        return self._atom_energy(pos, species, edge_index, torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device))

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        return self._atom_energy(pos, species, edge_index, edge_shift)

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """RACE.forward of bam_torch with the node features of the owned atoms sent to their ghost copies before every layer but the
        first (see _RaceLayered). Only the owned atoms are receivers of edges, so only their rows are computed after the first
        messages. Per atom: the model's energy + the reference energy of the species, as in forward_with_shifts. Returns [N], zero
        for the ghost atoms."""
        n_ghost = pos.shape[0] - n_owned
        node_attrs = self.species_one_hot.to(pos.dtype)[species]
        energy = self.layered(pos, species, node_attrs, edge_index, edge_shift, n_owned).to(pos.dtype)
        energy = energy + self.atom_ref.to(pos.dtype)[species[:n_owned]]
        return torch.cat([energy, torch.zeros(n_ghost, dtype=pos.dtype, device=pos.device)])
