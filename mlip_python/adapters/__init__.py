"""Adapters of the Python MLIP mode, one file per MLIP. ADAPTER_REGISTRY maps the `model_type` of mlip_config to its adapter.

A new MLIP: add its file here and one line to ADAPTER_REGISTRY. The packages of a MLIP are imported inside its initialize(),
so importing all adapters needs none of them installed.
"""
from __future__ import annotations

from .base import MLIPAdapter, _atom_types_to_symbols       # noqa: F401
from .bam import BAMAdapter
from .chgnet import CHGNetAdapter
from .deepmd import DeePMDAdapter
from .gap import GAPAdapter
from .mace import MACEAdapter
from .matgl import MatGLAdapter
from .mattersim import MatterSimAdapter
from .nequip import NequIPAdapter
from .orb import ORBAdapter
from .sevennet import SevenNetAdapter
from .uma import UMAAdapter


ADAPTER_REGISTRY: dict[str, type[MLIPAdapter]] = {
    # canonical keys
    "mace":        MACEAdapter,
    "mace_mp":     MACEAdapter,
    "mace_off":    MACEAdapter,
    "nequip":      NequIPAdapter,
    "allegro":     NequIPAdapter,     # Allegro models are compiled by the nequip package: same loader as NequIP
    "chgnet":      CHGNetAdapter,
    "sevennet":    SevenNetAdapter,
    "7net":        SevenNetAdapter,
    "orb":         ORBAdapter,
    # fairchem UMA
    "uma":         UMAAdapter,
    "uma_s":       UMAAdapter,
    "uma_m":       UMAAdapter,
    "fairchem":    UMAAdapter,
    # new adapters
    "mattersim":   MatterSimAdapter,
    "matgl":       MatGLAdapter,
    "m3gnet":      MatGLAdapter,
    "tensornet":   MatGLAdapter,
    "deepmd":      DeePMDAdapter,
    "deepmd_kit":  DeePMDAdapter,
    "gap":         GAPAdapter,
    "bam":         BAMAdapter,
    "bam_torch":   BAMAdapter,
    "quip":        GAPAdapter,
}
