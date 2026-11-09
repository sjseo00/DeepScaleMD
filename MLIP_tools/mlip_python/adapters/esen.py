"""eSEN (fairchem 2.x checkpoints, e.g. esen_sm_conserving_all.pt of OMol25): an adapter of the ASE mode (registered in
adapters/__init__.py). fairchem loads and runs eSEN exactly like UMA (MLIPPredictUnit + FAIRChemCalculator), so this is the UMA
adapter with one difference: no default task. An eSEN checkpoint has one task (omol, oc20 or odac) and fairchem picks it; give
task_name only for a checkpoint with several."""
from __future__ import annotations

from typing import Optional

from .uma import UMAAdapter


class ESENAdapter(UMAAdapter):
    def initialize(self, model_path: Optional[str] = None, model_type: str = "esen", model_size: str = "",
                   task_name: Optional[str] = None, **kwargs) -> bool:
        return super().initialize(model_path, model_type, model_size, task_name=task_name, **kwargs)
