"""UMA adapter (fairchem-core 2.14, fairchem/core/models/uma; as read from its source).

What fairchem does, which this adapter relies on:
  * A UMA checkpoint is loaded by fairchem's MLIPPredictUnit: a HydraModel with an eSCN-MD backbone (escn_md.py) and output heads,
    plus "tasks" that hold the normalizer and the element references of each output (omat_energy, omol_forces, ...).
  * The backbone is a mixture of linear experts (escn_moe.py): the experts are mixed by coefficients computed from the composition
    of the whole system (the mean of a composition embedding over its atoms), the total charge, the spin and the task. fairchem's own
    MD setting merge_mole fixes these coefficients for the first system it sees and merges the experts into one plain eSCN-MD model.
    This adapter does the same when it is built, with fairchem's own code (HydraModel.prepare_for_inference), for the composition given
    with --composition: the counts of the whole MD system, because the mean includes a zero start value and so depends on the count.
    A rank of dsMD sees only a part of the system, so the coefficients cannot be computed during the run.
  * Forward of the merged backbone (eSCNMDBackbone.forward with the graph given): edge vectors pos[edge_index[0]] - pos[edge_index[1]]
    + shift (the opposite sign of the standard edge vector), Wigner matrices from them (with a random roll angle, as in fairchem: the
    model is invariant to it, the numbers change at the rounding level), the atom embedding plus the charge-spin-task embedding, an
    edge-degree embedding (a sum over the edges onto edge_index[1]), num_layers message-passing blocks (sums onto edge_index[1]) and a
    final norm. The energy head of a conservative model (MLP_EFS_Head) maps the l=0 features of every atom to one number; the energy of
    the system is their sum, and the forces are its gradient. Per atom here: rmsd * e_i + element reference of Z_i + mean / N, which is
    fairchem's normalizer.denorm and element_references.undo_refs of the sum, split over the atoms (N: the atom count of --composition).
  * The edge-degree embedding is one extra hop: an atom's first features depend on its neighbors, so the ghost layer is
    (num_layers + 1) * cutoff deep (metadata num_layers = blocks + 1). With --layer-exchange (forward_layered) it is one cutoff deep:
    the owned atoms' features go to their ghost copies before every block.
GPU memory: autograd keeps the activations of every edge of every block for the force (backward) pass, about 220 kB per edge in
float32 for a UMA-S-sized model (edge_memory measures it; 3000 water atoms = 257k edges: ~56 GB; float64 and 24000 atoms do not
fit on a 95 GB GH200).
use_edge_chunks() switches on fairchem's own activation checkpointing (escn_md.py activation_checkpointing: the edge part of the
edge-degree embedding and of every block runs in chunks of 131072 edges and is computed again in the backward pass), which keeps
only a few kB per edge. It costs one more forward of the edge part, so it is off by default: dsMD's PyTorch mode (pytorch_runner.py)
switches it on by itself before the first step when edge_memory says the edges of the rank do not fit, or when a step runs out of
GPU memory (not with layer exchange), and keeps it on.
Not supported (refused): direct-force heads (forces not the gradient of the energy), forces normalized differently from the energy.
The code calls fairchem's modules (record_function, autograd Functions, rotations) and is not TorchScript: --backend pytorch only.
"""
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

from .base import MLIPAdapter


def _contiguous_input(module: torch.nn.Module, args):
    """Forward pre-hook of every Linear of the model: its input as a contiguous tensor. fairchem gives some Linears a strided view (an
    m-slice of the rotated edge features). torch.matmul folds a 3-D input into one matrix product only when it is contiguous or the
    weight needs a gradient; here the weights are frozen, so it expanded the weight over the edges and ran a batched product instead
    (2.3x the time of the whole step for UMA-S size). The copy changes no number and keeps no extra memory for the backward pass."""
    x = args[0]
    return None if x.is_contiguous() else (x.contiguous(),) + tuple(args[1:])


def _merge_dataset_experts(head, task: str) -> None:
    """A DatasetSpecificMoEWrapper head mixes its linear layers by the task (one expert per dataset): fix them for `task`."""
    from fairchem.core.models.uma.nn.mole import MOLE
    index = head.dataset_name_to_exp[task]
    head.global_mole_tensors.expert_mixing_coefficients = torch.zeros(1, len(head.dataset_name_to_exp)).index_fill(1, torch.tensor([index]), 1.0)

    def replace(module: torch.nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, MOLE):
                setattr(module, name, child.merged_linear_layer())
            else:
                replace(child)
    replace(head.head)


class UMAAdapter(MLIPAdapter):
    model_family = "uma"
    supports_shifts = True
    edge_receiver_row = 1       # messages and the edge-degree embedding are summed onto edge_index[1]
    supports_layer_exchange = True      # forward_layered: ghost atoms one cutoff deep (--layer-exchange)
    scriptable = False
    display_name = "UMA"            # the name in error messages (the eSEN adapter, adapters/esen.py, reuses this class)
    edge_chunk = 1024 * 128         # fairchem's ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE

    def __init__(self, backbone: torch.nn.Module, energy_block: torch.nn.Module, system_embedding: Tensor, energy_scale: float,
                 energy_offset: Tensor, composition_elements: List[int], task: str, model_name: str, versions: Dict[str, Optional[str]]):
        super().__init__(list(range(1, int(backbone.max_num_elements))))
        self.model_name = model_name
        self.head_name = task
        self.cutoff = float(backbone.cutoff)
        self.num_layers = int(backbone.num_layers) + 1      # the edge-degree embedding is one more hop
        self.backbone = backbone
        self.energy_block = energy_block
        self.energy_scale = float(energy_scale)
        self.register_buffer("system_embedding", system_embedding.detach().reshape(1, -1).clone())
        self.register_buffer("energy_offset", energy_offset.detach().clone())          # [len(elements)], per species
        self.register_buffer("atomic_numbers", torch.tensor(self.elements, dtype=torch.int64))
        self._composition_elements = list(composition_elements)
        self._versions = versions
        for module in list(backbone.modules()) + list(energy_block.modules()):
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(_contiguous_input)

    @classmethod
    def from_checkpoint(cls, path, task: Optional[str] = None, composition: Optional[str] = None, charge: Optional[int] = None,
                        spin: Optional[int] = None) -> "UMAAdapter":
        if not composition:
            raise ValueError("arch 'uma' needs --composition <formula of the whole MD system with its atom counts, e.g. H2000O1000>: the "
                             "mixture of experts of UMA is fixed for that composition when the model is converted")
        return cls._load(path, task, composition, charge, spin)

    @classmethod
    def _load(cls, path, task: Optional[str], composition: Optional[str], charge: Optional[int], spin: Optional[int]) -> "UMAAdapter":
        """Load a fairchem checkpoint with fairchem's own code. composition None (eSEN only): the checkpoint must have no experts and no
        energy mean (both depend on the atom count); the cell fairchem needs to prepare the model is then a placeholder H2O."""
        name = cls.display_name
        if not Path(path).is_file():
            raise ValueError(f"{name} checkpoint not found: {path}")
        import ase
        import fairchem.core
        from fairchem.core.datasets.atomic_data import AtomicData
        from fairchem.core.models.uma.escn_md import MLP_EFS_Head
        from fairchem.core.models.uma.escn_moe import DatasetSpecificMoEWrapper
        from fairchem.core.units.mlip_unit import MLIPPredictUnit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        unit = MLIPPredictUnit(str(path), device="cpu", inference_settings=InferenceSettings(merge_mole=True))
        if not composition and getattr(unit.model.module.backbone, "num_experts", 0):
            raise ValueError(f"this checkpoint is a mixture of experts (a UMA model): convert it with arch 'uma' and --composition")
        tasks = sorted(unit.dataset_to_tasks)
        if task is None:
            if len(tasks) != 1:
                raise ValueError(f"this {name} model has several tasks: choose one with --head ({', '.join(tasks)})")
            task = tasks[0]
        if task not in tasks:
            raise ValueError(f"{name} task '{task}' is not one of the model's tasks: {', '.join(tasks)}")
        try:
            atoms = ase.Atoms(composition or "H2O")
        except Exception as error:      # noqa: BLE001  (ase raises several classes for a bad formula)
            raise ValueError(f"--composition '{composition}' is not a chemical formula with atom counts (e.g. H2000O1000): {error}") from error
        if charge is not None:
            atoms.info["charge"] = int(charge)
        if spin is not None:
            atoms.info["spin"] = int(spin)
        unit.validate_atoms_data(atoms, task)       # fairchem's defaults (omol: spin 1) and range checks
        data = AtomicData.from_ase(atoms, task_name=task, r_edges=False, r_data_keys=["spin", "charge"])

        model = unit.model.module       # the HydraModel with the EMA weights
        with torch.no_grad():
            model.prepare_for_inference(data, unit.inference_settings)     # merges the experts for this composition, charge, spin, task
        backbone = model.backbone.eval()
        if getattr(backbone, "num_experts", 0):
            raise ValueError(f"{name}: the mixture of experts was not merged by fairchem (only mole_type so2 can be merged)")

        head = None
        for candidate in model.output_heads.values():
            if isinstance(getattr(candidate, "head", candidate), MLP_EFS_Head):
                head = candidate
        if head is None:
            raise ValueError(f"{name}: only conservative models are supported (an MLP_EFS_Head: forces = -dE/dx); this checkpoint predicts "
                             "forces directly")
        if isinstance(head, DatasetSpecificMoEWrapper):
            _merge_dataset_experts(head, task)
        energy_block = getattr(head, "head", head).energy_block      # UMA: a per-dataset wrapper; eSEN: the head itself

        by_property = {t.property: t for t in unit.dataset_to_tasks[task]}
        if "energy" not in by_property:
            raise ValueError(f"{name} task '{task}' has no energy output")
        energy_task = by_property["energy"]
        rmsd = float(energy_task.normalizer.rmsd.reshape(-1)[0])
        mean = float(energy_task.normalizer.mean.reshape(-1)[0])
        forces_task = by_property.get("forces")
        if forces_task is not None:
            f_rmsd = float(forces_task.normalizer.rmsd.reshape(-1)[0])
            f_mean = float(forces_task.normalizer.mean.reshape(-1)[0])
            if abs(f_rmsd - rmsd) > 1e-6 * abs(rmsd) or f_mean != 0.0:
                raise ValueError(f"{name} task '{task}': the forces are normalized differently from the energy (rmsd {f_rmsd} vs {rmsd}, mean "
                                 f"{f_mean}), so fairchem's forces are not the gradient of its energy")
        n_total = len(atoms)
        if not composition and mean != 0.0:
            raise ValueError(f"{name} task '{task}': the energy normalizer has a mean ({mean} eV per system), which is split over the "
                             "atoms: give --composition <formula of the whole MD system with its atom counts>")
        n_z = int(backbone.max_num_elements)
        refs = torch.zeros(n_z, dtype=torch.float64)
        if energy_task.element_references is not None:
            table = energy_task.element_references.element_references.detach().to(torch.float64).reshape(-1)
            refs[: min(n_z, table.shape[0])] = table[:n_z]
        offset = refs[1:] + mean / n_total          # species index i is Z = i + 1

        with torch.no_grad():
            system_embedding = backbone.csd_embedding(charge=data["charge"], spin=data["spin"], dataset=data["dataset"])
        for parameter in list(backbone.parameters()) + list(energy_block.parameters()):
            parameter.requires_grad_(False)
        composition_elements = list(dict.fromkeys(int(z) for z in atoms.get_atomic_numbers()))
        return cls(backbone, energy_block, system_embedding, rmsd, offset, composition_elements, task, Path(path).stem,
                   {"fairchem-core": getattr(fairchem.core, "__version__", None)})

    def use_edge_chunks(self, on: bool = True) -> None:
        """fairchem's activation checkpointing of the edge parts (see the module docstring): far less GPU memory, more compute."""
        chunk = self.edge_chunk if on else None
        self.backbone.edge_degree_embedding.activation_checkpoint_chunk_size = chunk
        for block in self.backbone.blocks:
            block.edge_wise.activation_checkpoint_chunk_size = chunk

    @property
    def uses_edge_chunks(self) -> bool:
        return self.backbone.edge_degree_embedding.activation_checkpoint_chunk_size is not None

    def edge_memory(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> float:
        """Bytes per edge that autograd keeps for the backward pass without edge chunks: the tensors saved by two forwards on the
        first 1024 and 2048 edges, their difference over 1024 (what scales with the atoms or is a weight cancels). Only the atoms of
        those edges take part (all atoms of a big rank would fill the GPU). 0 with fewer edges."""
        if edge_index.shape[1] < 2048 or self.uses_edge_chunks:
            return 0.0
        atoms, local = torch.unique(edge_index[:, :2048], return_inverse=True)
        pos, species, edge_index, edge_shift = pos[atoms], species[atoms], local, edge_shift[:2048]
        saved = []
        for k in (1024, 2048):
            storages = {}

            def pack(t: Tensor) -> Tensor:
                storage = t.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
                return t
            with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                p = pos.detach().requires_grad_(True)
                self.forward_with_shifts(p, species, edge_index[:, :k], edge_shift[:k]).sum()
            saved.append(sum(storages.values()))
        return max(0.0, (saved[1] - saved[0]) / 1024.0)

    def package_versions(self) -> Dict[str, Optional[str]]:
        return dict(self._versions)

    @property
    def reference_elements(self):
        return self._composition_elements[:3]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward_with_shifts(pos, species, edge_index, torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device))

    def _first_features(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor):
        """What the blocks take: the node features after the embeddings and the edge-degree embedding, and the edge inputs."""
        bb = self.backbone
        z = self.atomic_numbers[species]
        n_atoms = pos.shape[0]
        # fairchem: pos[edge_index[0]] - pos[edge_index[1]] + shift, the opposite of the standard pos[j] - pos[i] + shift
        vec = -(pos[edge_index[1]] - pos[edge_index[0]] + edge_shift)
        dist = torch.linalg.norm(vec, dim=-1)
        wigner, wigner_inv = bb._get_rotmat_and_wigner(vec)
        system = self.system_embedding.to(pos.dtype).expand(n_atoms, -1)

        x = torch.zeros(n_atoms, bb.sph_feature_size, bb.sphere_channels, dtype=pos.dtype, device=pos.device)
        x[:, 0, :] = bb.sphere_embedding(z) + system
        envelope = bb.envelope(dist / bb.cutoff).reshape(-1, 1, 1)
        x_edge = torch.cat((bb.distance_expansion(dist), bb.source_embedding(z[edge_index[0]]), bb.target_embedding(z[edge_index[1]])), dim=1)
        x = bb.edge_degree_embedding(x, x_edge, edge_index, wigner_inv, envelope, 0)
        return x, (x_edge, dist, edge_index, wigner, wigner_inv, envelope), system

    def _atom_energies(self, x: Tensor, species: Tensor) -> Tensor:
        x = self.backbone.norm(x)
        e = self.energy_block(x.narrow(1, 0, 1).squeeze(1)).reshape(-1)
        return e * self.energy_scale + self.energy_offset.to(e.dtype)[species]

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        x, edges, system = self._first_features(pos, species, edge_index, edge_shift)
        for block in self.backbone.blocks:
            x = block(x, *edges, total_atoms_across_gp_ranks=pos.shape[0], sys_node_embedding=system, node_offset=0)
        return self._atom_energies(x, species)

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """forward_with_shifts on a rank whose ghost atoms are one cutoff deep and whose edges all end on an owned atom (edge_index[1]).
        The edge-degree embedding and every block sum their messages onto edge_index[1], so after each of them the owned rows are
        right and the ghost rows are not: before every block the owned rows are sent to their ghost copies (dsmd::halo_exchange),
        num_layers - 1 exchanges. A block still runs its atom-wise part on the ghost rows too (fairchem's block as it is); those rows
        are thrown away. Per atom energies [N], zero for the ghosts."""
        n_atoms = pos.shape[0]
        n_ghost = n_atoms - n_owned
        x, edges, system = self._first_features(pos, species, edge_index, edge_shift)
        for block in self.backbone.blocks:
            x = torch.ops.dsmd.halo_exchange(x[:n_owned].reshape(n_owned, -1), n_ghost).reshape(x.shape)
            x = block(x, *edges, total_atoms_across_gp_ranks=n_atoms, sys_node_embedding=system, node_offset=0)
        e = self._atom_energies(x[:n_owned], species[:n_owned])
        return torch.cat([e, e.new_zeros(n_ghost)])
