"""PaiNN adapter (schnetpack 2.1.1).

What the installed schnetpack does, which this adapter relies on:
  * a trained model is a schnetpack.model.NeuralNetworkPotential saved whole with torch.save (the `best_model` file of a
    training run): input_modules [PairwiseDistances], a representation (here schnetpack.representation.PaiNN), output modules
    [Atomwise(output_key="energy")] and postprocessors (CastTo64, AddOffsets, ...).
  * PaiNN.forward reads four tensors of the input dict: _atomic_numbers [N] (Z itself: the embedding is indexed by Z),
    _Rij [E, 3] = R[idx_j] - R[idx_i] + offsets, _idx_i [E], _idx_j [E]; messages are summed on idx_i. That is the standard
    input of this converter with idx_i = edge_index[0] (the center) and idx_j = edge_index[1] (the neighbor), and the
    offsets are zero because the ghost atoms are real positions.
  * Atomwise.outnet maps the scalar representation to one energy per atom; the aggregation over atoms and the postprocessors
    (AddOffsets: dataset mean per atom and single-atom references per element) come after it, so they are applied per atom here.
    The adapter does not call the model: it runs the representation itself (PaiNN.forward builds its vector features with a
    float32 torch.zeros, which would mix dtypes in a float64 model) and the output network.
  * cutoff = the cutoff of the cosine cutoff function, num_layers = n_interactions.
  * PaiNN does not know which elements it was trained on (QM9 models have H, C, N, O, F), so --elements is required.
Not supported (the adapter refuses them): electronic embeddings (total charge, spin), a postprocessor other than CastTo32/64
and AddOffsets, intensive offsets, an output module other than one extensive Atomwise energy.
"""
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from .base import MLIPAdapter


def load_painn_model(path) -> torch.nn.Module:
    """The schnetpack model of a file written by torch.save(model, ...) (the `best_model` of a training run)."""
    from pathlib import Path
    if not Path(path).is_file():
        raise ValueError(f"PaiNN model file not found: {path}")
    model = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"{path} does not hold a schnetpack model (torch.save of the whole model): got {type(model).__name__}")
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class PaiNNAdapter(MLIPAdapter):
    model_family = "painn"

    def __init__(self, model: torch.nn.Module, elements: Sequence[int], model_name: str = "painn"):
        import schnetpack as spk
        from schnetpack.atomistic import Atomwise
        from schnetpack.representation import PaiNN
        import schnetpack.transform as trn

        if not isinstance(model, spk.model.NeuralNetworkPotential):
            raise ValueError(f"PaiNN: expected a schnetpack NeuralNetworkPotential, got {type(model).__name__}")
        if not isinstance(model.representation, PaiNN):
            raise ValueError(f"this schnetpack model is a {type(model.representation).__name__}, not a PaiNN")
        heads = [m for m in model.output_modules if isinstance(m, Atomwise)]
        others = [m for m in model.output_modules if not isinstance(m, (Atomwise, spk.atomistic.Forces))]
        if len(heads) != 1 or others:
            raise ValueError("PaiNN: expected one Atomwise output module (the energy), optionally followed by Forces")
        head = heads[0]
        if head.aggregation_mode not in ("sum", None):
            raise ValueError(f"PaiNN: the energy head aggregates with '{head.aggregation_mode}', only an extensive sum is supported")
        rep = model.representation
        if len(rep.electronic_embeddings) != 0:
            raise ValueError("PaiNN: models with electronic embeddings (total charge, spin) are not supported")
        if not elements:
            raise ValueError("PaiNN does not record its elements: pass them with --elements, e.g. --elements 1,6,7,8,9")
        max_z = rep.embedding.num_embeddings
        if max(elements) >= max_z or min(elements) < 1:
            raise ValueError(f"PaiNN embeds atomic numbers below {max_z}: --elements {list(elements)} is out of range")

        super().__init__(list(elements))
        self.model_name = model_name
        self.cutoff = float(rep.cutoff_fn.cutoff)
        self.num_layers = int(rep.n_interactions)
        # the parts of PaiNN.forward, held directly so that the float64 conversion reaches all of them
        self.embedding = rep.embedding
        self.radial_basis = rep.radial_basis
        self.cutoff_fn = rep.cutoff_fn
        self.filter_net = rep.filter_net
        self.interactions = rep.interactions
        self.mixing = rep.mixing
        self.outnet = head.outnet
        self.share_filters = bool(rep.share_filters)
        self.n_interactions = int(rep.n_interactions)
        self.n_atom_basis = int(rep.n_atom_basis)

        # postprocessors, applied per atom: the dataset mean and the single-atom references
        mean = 0.0
        atomref = torch.zeros(max_z)
        for processor in model.postprocessors:
            if isinstance(processor, (trn.CastTo32, trn.CastTo64)):
                continue
            if isinstance(processor, trn.AddOffsets):
                if processor._property != head.output_key:
                    raise ValueError(f"PaiNN: AddOffsets for '{processor._property}', not for the energy '{head.output_key}'")
                if not processor.is_extensive:
                    raise ValueError("PaiNN: an intensive AddOffsets cannot be split over the atoms")
                if processor.add_mean:
                    mean += float(processor.mean.reshape(-1)[0])
                if processor.add_atomrefs:
                    refs = processor.atomref.detach().reshape(-1).to(torch.float64)
                    atomref[: min(max_z, refs.shape[0])] += refs[:max_z].to(atomref.dtype)
                continue
            raise ValueError(f"PaiNN: the postprocessor {type(processor).__name__} is not supported (CastTo32/64 and AddOffsets are)")
        self.register_buffer("energy_per_species",
                             torch.tensor([float(atomref[z]) + mean for z in elements], dtype=torch.float32))
        self.register_buffer("atomic_numbers", torch.tensor(list(elements), dtype=torch.int64))

    def package_versions(self) -> Dict[str, Optional[str]]:
        import schnetpack
        return {"schnetpack": schnetpack.__version__}

    @property
    def reference_elements(self):
        return self.elements[:3]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        idx_i = edge_index[0]
        idx_j = edge_index[1]
        z = self.atomic_numbers[species]
        r_ij = pos[idx_j] - pos[idx_i]
        n_atoms = pos.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)
        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(z).unsqueeze(1)
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), dtype=q.dtype, device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
        q = q.squeeze(1)

        y = self.outnet(q).squeeze(-1)
        return y + self.energy_per_species.to(y.dtype)[species]
