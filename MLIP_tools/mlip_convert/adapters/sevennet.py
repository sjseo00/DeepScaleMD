"""SevenNet adapter (design 3.4, as checked in section 11: sevenn 0.13.0).

What the installed sevenn does, which this adapter relies on:
  * a model is an sevenn.nn.sequential.AtomGraphSequential (SevenNetCheckpoint(...).build_model()). Deployment for LAMMPS
    (sevenn/scripts/deploy.py) deletes the force_output module, sets eval_type_map = False and set_is_batch_data(False), and
    calls the model with the dict
      x [N] int64 (index in model.type_map, i.e. the model's species order), edge_index [2, E] int64,
      edge_vec [E, 3] = pos[edge_index[1]] - pos[edge_index[0]], num_atoms [1], nlocal [1]
    The edge list of this converter is symmetric (both directions), so the standard edge_index is passed as it is.
  * out["atomic_energy"] [N, 1] holds the per-atom energies including the per-species scale and shift, in eV;
    out["inferred_total_energy"] sums over all atoms (ghosts included) and is not used: MDWrapper sums the owned atoms.
  * cutoff = model.cutoff, num_layers = the number of *_convolution modules (the pretrained 7net models have 5, so the ghost
    depth is 25 A), type_map = {atomic number: species index}.
  * multi-modal models (SevenNet-MF) need one modal chosen (--head): prepare_modal_deploy(modal) prepends the modal input.
  * the one-hot embedding module casts to float32 whatever the dtype (OnehotEmbedding.forward: embd.float()), so a model
    converted to float64 fails. It is replaced by one that follows the model's dtype; for float32 it is identical.
  * pretrained checkpoints (7net-0, 7net-MF-0, ...) ship inside the pip package; a name that is not a file is NOT resolved
    here, because sevenn would download it. Pass the checkpoint file.
"""
from pathlib import Path
from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode
from torch import Tensor

from .base import MLIPAdapter


def _make_onehot_follow_dtype(base: torch.nn.Module) -> torch.nn.Module:
    from sevenn.nn.node_embedding import OnehotEmbedding

    @compile_mode("script")
    class OnehotFollowsDtype(OnehotEmbedding):
        def __init__(self, source: OnehotEmbedding):
            super().__init__(source.num_classes, source.key_x, source.key_output, source.key_save,
                             source.key_additional_output)
            # a buffer, so that model.to(dtype) moves it: its dtype is the model's dtype
            self.register_buffer("dtype_carrier", torch.zeros(1))

        def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
            inp = data[self.key_x]
            embd = torch.nn.functional.one_hot(inp, self.num_classes).to(self.dtype_carrier.dtype)
            data[self.key_output] = embd
            if self.key_additional_output is not None:
                data[self.key_additional_output] = embd
            if self.key_save is not None:
                data[self.key_save] = inp
            return data

    return OnehotFollowsDtype(base)


def _require_accelerated(model: torch.nn.Module, accel: str) -> None:
    """sevenn falls back to e3nn with only a warning when the kernel package is missing (checkpoint.py `_solve_accelerator`);
    a model that was asked to be accelerated and is not must not be converted as if it were."""
    names = {type(m).__name__ for m in model.modules()}
    paths = {type(m).__module__.split(".")[0] for m in model.modules()}
    if accel == "cueq" and "cuequivariance_torch" not in paths:
        raise ValueError("--accel cueq: sevenn fell back to e3nn: the packages cuequivariance and cuequivariance-torch are "
                         "missing (pip install cuequivariance cuequivariance-torch)")
    if accel == "flash" and "IrrepsScatterGatherFusedConvolution" not in names:
        raise ValueError("--accel flash: sevenn fell back to e3nn: flashTP_e3nn is not installed or no CUDA device is "
                         "available (FlashTP has no CPU kernels; https://github.com/SNU-ARC/flashTP)")
    if accel == "openeq" and "OEQConvolution" not in names:
        raise ValueError("--accel openeq: sevenn fell back to e3nn: openequivariance is not installed or no CUDA device is "
                         "available (OpenEquivariance has no CPU kernels; pip install openequivariance)")


def load_sevennet_model(path, modal: Optional[str] = None, accel: str = "e3nn") -> torch.nn.Module:
    """The SevenNet model of a checkpoint file (.pth), prepared as sevenn's own deployment does, in eval mode, no gradients.
    accel "cueq" / "openeq" / "flash" builds it with sevenn's own cuEquivariance / OpenEquivariance / FlashTP tensor products
    (flash: the LAMMPS variant of the kernels, `_flash_lammps`, as `sevenn get_model -flashTP` does)."""
    if accel not in ("e3nn", "cueq", "openeq", "flash"):
        raise ValueError(f"SevenNet supports --accel e3nn, cueq, openeq and flash, not '{accel}'")
    if not Path(path).is_file():
        raise ValueError(f"SevenNet checkpoint file not found: {path} (names such as '7net-0' are not resolved: sevenn "
                         f"would download missing checkpoints; pass the .pth file)")
    from sevenn.checkpoint import SevenNetCheckpoint
    if accel == "flash":
        # sevenn would drop FlashTP with a warning and then fail its own assert on `_flash_lammps`: say it plainly instead
        from sevenn.nn.flash_helper import is_flash_available
        if not is_flash_available():
            raise ValueError("--accel flash: sevenn fell back to e3nn: flashTP_e3nn is not installed or no CUDA device is "
                             "available (FlashTP has no CPU kernels; https://github.com/SNU-ARC/flashTP)")
    model = SevenNetCheckpoint(str(path)).build_model(enable_cueq=accel == "cueq", enable_oeq=accel == "openeq",
                                                      enable_flash=accel == "flash", _flash_lammps=accel == "flash")
    _require_accelerated(model, accel)

    if "force_output" in model._modules:
        model.delete_module_by_key("force_output")
    model.eval_type_map = False
    if modal:
        if model.modal_map is None or modal not in model.modal_map:
            raise ValueError(f"modal '{modal}' is not one of the model's modals: "
                             f"{list(model.modal_map) if model.modal_map else 'none (single-modal model)'}")
        model.prepare_modal_deploy(modal)
    elif model.modal_map:
        raise ValueError(f"this SevenNet model has several modals, choose one with --head: {list(model.modal_map)}")
    model.set_is_batch_data(False)

    from sevenn.nn.node_embedding import OnehotEmbedding
    for key, module in list(model._modules.items()):
        if isinstance(module, OnehotEmbedding):
            model.replace_module(key, _make_onehot_follow_dtype(module))

    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _owned_rows(data: Dict[str, Tensor], n_owned: int) -> Dict[str, Tensor]:
    """The dict of a layer's modules with the per-atom tensors cut to the owned atoms (the first n_owned rows). The other entries
    (the edge tensors, the modal tensors of a multi-modal model) are shared as they are."""
    owned: Dict[str, Tensor] = {}
    for key, value in data.items():
        if key == "x" or key == "node_attr" or key == "atom_type":
            owned[key] = value[:n_owned]
        else:
            owned[key] = value
    return owned


class _OwnedRowsLayer(torch.nn.Module):
    """One interaction layer of a SevenNet model (the modules of sevenn's own sequence, the same weights), for a rank that owns
    n_owned atoms and has ghost atoms after them. The convolution is the only step that needs the ghost rows: they are the senders
    of messages (and the self interaction in front of it makes the features that are sent). Everything else of the layer, the
    self connection in front, the second self interaction, the sum with the self connection and the gate, works on the owned rows
    only (only owned atoms receive edges). The ghost rows would otherwise be computed again on every rank and the work would not
    fall when ranks are added. Returns the dict of the owned atoms."""

    def __init__(self, intro: torch.nn.Module, interaction_1: torch.nn.Module, convolution: torch.nn.Module,
                 interaction_2: torch.nn.Module, outro: torch.nn.Module, gate: torch.nn.Module):
        super().__init__()
        self.intro = intro
        self.interaction_1 = interaction_1
        self.convolution = convolution
        self.interaction_2 = interaction_2
        self.outro = outro
        self.gate = gate

    def forward(self, data: Dict[str, Tensor], n_owned: int) -> Dict[str, Tensor]:
        owned = _owned_rows(data, n_owned)
        owned = self.intro(owned)                  # the self connection of the features that enter the layer
        data = self.interaction_1(data)            # all rows: the senders include the ghosts
        data = self.convolution(data)              # sums onto the receivers (edge_index[0]), which are owned atoms
        owned["x"] = data["x"][:n_owned]
        owned = self.interaction_2(owned)
        owned = self.outro(owned)
        owned = self.gate(owned)
        return owned

class SevenNetAdapter(MLIPAdapter):
    model_family = "sevennet"
    e3nn_compile = True
    supports_shifts = True
    supports_layer_exchange = True
    edge_receiver_row = 0       # sevenn: the convolution sends from edge_index[1] and sums onto edge_index[0]

    def __init__(self, model: torch.nn.Module, modal: Optional[str] = None, model_name: str = "sevennet",
                 accel: str = "e3nn"):
        by_index = sorted(model.type_map.items(), key=lambda item: item[1])
        if [index for _, index in by_index] != list(range(len(by_index))):
            raise ValueError(f"unexpected SevenNet type_map (species indices are not 0..n-1): {model.type_map}")
        # A model can have placeholder species with atomic number 0 (SevenNet-Omni has one): they are not elements, so the adapter's
        # species are the real elements only, and type_index turns a species into the model's own index (the identity when every
        # species is an element).
        real = [(int(z), index) for z, index in by_index if int(z) >= 1]
        super().__init__([z for z, _ in real])
        self.register_buffer("type_index", torch.tensor([index for _, index in real], dtype=torch.int64))
        self.accel = accel
        self.model = model
        self.model_name = model_name
        self.cutoff = float(model.cutoff)
        self.num_layers = sum(key.endswith("_convolution") for key in model._modules)
        if self.num_layers < 1:
            raise ValueError("this SevenNet model has no convolution layer")
        self.head_name = modal

    def enable_layer_exchange(self) -> None:
        """Switches layer exchange on and sorts the modules of the model into what runs before the layers (on all atoms), the layers
        (owned atoms, ghost atoms only as senders) and what runs after them (owned atoms). They share the weights of the model.
        Needs the layout sevenn builds: per layer t the modules {t}_self_connection_intro, {t}_self_interaction_1, {t}_convolution,
        {t}_self_interaction_2, {t}_self_connection_outro, {t}_equivariant_gate, one after the other."""
        from sevenn.nn.linear import AtomReduce
        names = list(self.model._modules.keys())
        steps = ["self_connection_intro", "self_interaction_1", "convolution", "self_interaction_2", "self_connection_outro",
                 "equivariant_gate"]
        if "0_self_connection_intro" not in names:
            raise ValueError("layer exchange: this SevenNet model has no module 0_self_connection_intro, so its layers are not "
                             f"the ones forward_layered knows (modules: {names})")
        first = names.index("0_self_connection_intro")
        position = first
        layers = []
        for t in range(self.num_layers):
            expected = [f"{t}_{step}" for step in steps]
            if names[position:position + len(steps)] != expected:
                raise ValueError(f"layer exchange: layer {t} of this SevenNet model is not {expected} (it is "
                                 f"{names[position:position + len(steps)]})")
            layers.append(_OwnedRowsLayer(*[self.model._modules[name] for name in expected]))
            position += len(steps)
        super().enable_layer_exchange()
        self.pre_modules = torch.nn.ModuleList([self.model._modules[name] for name in names[:first]])
        self.layer_blocks = torch.nn.ModuleList(layers)
        # the sum over the atoms (AtomReduce) is not needed: the per-atom energies are returned
        self.post_modules = torch.nn.ModuleList([self.model._modules[name] for name in names[position:]
                                                 if not isinstance(self.model._modules[name], AtomReduce)])

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        import sevenn
        return {"sevenn": sevenn.__version__, "e3nn": e3nn.__version__}

    @property
    def reference_elements(self) -> List[int]:
        # A chemically ordinary mix for the reference structure: water-like when the model knows H and O.
        if 1 in self.elements and 8 in self.elements:
            return [8, 1]
        return self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        species = self.type_index[species]
        n_atoms = pos.shape[0]
        data: Dict[str, Tensor] = {
            "x": species,
            "edge_index": edge_index,
            "edge_vec": pos[edge_index[1]] - pos[edge_index[0]],
            "num_atoms": torch.tensor([n_atoms], dtype=torch.long, device=pos.device),
            "nlocal": torch.tensor([n_atoms], dtype=torch.long, device=pos.device),
        }
        out = self.model(data)
        return out["atomic_energy"].squeeze(-1).to(pos.dtype)

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """sevenn's sequence of modules with the node features of the owned atoms sent to their ghost copies before every layer but
        the first: the modules before the layers run on all atoms, each layer on the owned atoms (ghosts only as senders), the
        modules after the last layer on the owned atoms. Only the owned atoms are receivers of edges. Returns the per-atom energies
        [N], zero for the ghost atoms."""
        species = self.type_index[species]
        n_all = pos.shape[0]
        n_ghost = n_all - n_owned
        data: Dict[str, Tensor] = {
            "x": species,
            "edge_index": edge_index,
            "edge_vec": pos[edge_index[1]] - pos[edge_index[0]] + edge_shift,
            "num_atoms": torch.tensor([n_all], dtype=torch.long, device=pos.device),
            "nlocal": torch.tensor([n_all], dtype=torch.long, device=pos.device),
        }
        for module in self.pre_modules:
            data = module(data)
        owned: Dict[str, Tensor] = {}
        layer = 0
        for block in self.layer_blocks:
            owned = block(data, n_owned)
            if layer < self.num_layers - 1:
                data["x"] = torch.ops.dsmd.halo_exchange(owned["x"], n_ghost)
            layer += 1
        for module in self.post_modules:
            owned = module(owned)
        energy = owned["atomic_energy"].squeeze(-1).to(pos.dtype)
        return torch.cat([energy, torch.zeros(n_ghost, dtype=pos.dtype, device=pos.device)])

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        species = self.type_index[species]
        n_atoms = pos.shape[0]
        data: Dict[str, Tensor] = {
            "x": species,
            "edge_index": edge_index,
            "edge_vec": pos[edge_index[1]] - pos[edge_index[0]] + edge_shift,
            "num_atoms": torch.tensor([n_atoms], dtype=torch.long, device=pos.device),
            "nlocal": torch.tensor([n_atoms], dtype=torch.long, device=pos.device),
        }
        out = self.model(data)
        return out["atomic_energy"].squeeze(-1).to(pos.dtype)
