"""eSEN adapter (fairchem-core 2.x: the eSEN checkpoints of OMol25, OC25 and ODAC25 that fairchem's pretrained_models.json lists,
e.g. esen_sm_conserving_all.pt of facebook/OMol25).

fairchem 2.x stores an eSEN model the same way as UMA: a HydraModel whose backbone is eSCNMDBackbone (escn_md.py), the network UMA
is made of, without the mixture of experts. So this adapter is the UMA adapter (adapters/uma.py) with two differences:
  * --composition is not needed: there are no experts to mix. It is needed only when the energy normalizer of the task has a mean
    (fairchem adds it once per system; here it is split over the atoms, so the atom count must be known). A UMA checkpoint (one with
    experts) is refused with a pointer to arch 'uma'.
  * --head is the task only when the checkpoint has several (the eSEN checkpoints have one: omol, oc20 or odac).
Only conservative eSEN models are supported (an MLP_EFS_Head: forces = -dE/dx). The 'direct' checkpoints (esen-sm-direct-all-omol,
esen-md-direct-all-omol, ...) predict the forces with a separate head, so their forces are not the gradient of their energy and an MD run
with them does not conserve energy: they are refused.
The fairchem 1.x eSEN code (fairchem.core.models.esen, e.g. the OMat24 eSEN-30M-OAM checkpoint) is a different package version and is
not supported. Not TorchScript (fairchem's modules): --backend pytorch only.
"""
from typing import Optional

from .uma import UMAAdapter


class ESENAdapter(UMAAdapter):
    model_family = "esen"
    display_name = "eSEN"

    @classmethod
    def from_checkpoint(cls, path, task: Optional[str] = None, composition: Optional[str] = None, charge: Optional[int] = None,
                        spin: Optional[int] = None) -> "ESENAdapter":
        return cls._load(path, task, composition, charge, spin)
