"""What the CLI knows about each MLIP: how to convert it, where its model files can be found, how the Python MLIP mode names it.

Model files are found in five ways (see sources.py): Hugging Face repos (listed live), GitHub releases or folders (listed live),
files shipped inside the installed Python package, a local file, and an address typed by the user.
Nothing here downloads anything. The repos below are the ones checked on 2026-10-14; the UI lists what is in them at the moment.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Source:
    kind: str            # "hf" | "github_release" | "github_dir" | "package"
    ref: str             # hf: repo id | github_release: owner/name | github_dir: owner/name | package: (unused)
    label: str           # shown in the list, e.g. "Hugging Face (mirror)"
    path: str = ""       # github_dir: folder in the repo; package: folder inside the package, relative to the package root
    folder: bool = False  # hf: the model is the whole repo folder (matgl: model.json + state.pt + model.pt)
    gated: bool = False   # needs an access token (HF_TOKEN) and an accepted license


@dataclass
class MLIP:
    key: str                         # name in the menu
    title: str
    convert_arch: Optional[str]      # mlip_convert architecture; None = no LibTorch converter for it
    python_type: Optional[str]       # model_type in mlip_config of the Python MLIP mode; None = no Python MLIP adapter
    package: str                     # Python module that must be importable to convert (or run) it
    pip: str                         # what to install when it is missing
    extensions: Tuple[str, ...]      # file endings that are model files
    sources: List[Source] = field(default_factory=list)
    hf_queries: Tuple[str, ...] = ()  # words for "search Hugging Face"
    needs_elements: bool = False     # converter needs --elements (PaiNN)
    has_heads: bool = False          # --head (mace head, sevennet modal)
    accels: Tuple[str, ...] = ("e3nn",)
    fixed_dtype: bool = False        # the model carries its own precision (nequip, deepmd): no --dtype
    optional_model: bool = False     # the converter works without a model file (CHGNet uses the weights inside the package)
    layer_exchange: bool = False     # mlip_convert --layer-exchange works for it (its adapter has supports_layer_exchange: mace, nequip,
                                     # sevennet, bam, mattersim, tensornet, painn, deepmd; the toy test models are not in this menu).
                                     # Several GPUs: ghost atoms one cutoff deep, node features sent between the layers
    layer_exchange_local: bool = False  # the model is strictly local (one layer): --layer-exchange then exchanges nothing, it only makes
                                     # a rank compute its own atoms and not its ghost atoms too (deepmd)
    include: Optional[str] = None    # regex: only file names matching it are listed
    exclude: Optional[str] = None    # regex: file names matching it are left out
    note: str = ""


MLIPS: Dict[str, MLIP] = {}


def _add(m: MLIP) -> None:
    MLIPS[m.key] = m


_add(MLIP(
    "mace", "MACE (MACE-MP-0, MPA, OMAT, MATPES, MH, POLAR, OMOL ...)", "mace", "mace", "mace", "mace-torch",
    (".model", ".pt", ".pth"), has_heads=True, accels=("e3nn", "cueq", "openeq"), layer_exchange=True,
    sources=[Source("github_release", "ACEsuit/mace-mp", "GitHub release ACEsuit/mace-mp"),
             Source("hf", "InstaDeepAI/mace-organics", "Hugging Face")],
    hf_queries=("mace",),
    note="MACE-MP-0 is also downloaded by the mace package itself (mace_mp); here the file is saved and converted."))

_add(MLIP(
    "nequip", "NequIP (.nequip.zip / .ckpt)", "nequip", "nequip", "nequip", "nequip",
    (".nequip.zip", ".ckpt", ".zip"), fixed_dtype=True, exclude=r"(?i)allegro", layer_exchange=True,
    sources=[Source("hf", "JinukMoon/oh-my-mlip-mirror-nequip", "Hugging Face (mirror of nequip.net models)"),
             Source("hf", "InstaDeepAI/nequip-organics", "Hugging Face")],
    hf_queries=("nequip",),
    note="Models are also on nequip.net: choose 'type an address' and give nequip.net:mir-group/NequIP-OAM-L:0.1. "
         "Ghost atoms are layers x r_max deep, or r_max with layer exchange (several GPUs)."))

_add(MLIP(
    "allegro", "Allegro (.nequip.zip / .ckpt)", "allegro", "allegro", "allegro", "nequip-allegro",
    (".nequip.zip", ".ckpt", ".zip"), fixed_dtype=True, include=r"(?i)allegro",
    sources=[Source("hf", "JinukMoon/oh-my-mlip-mirror-nequip", "Hugging Face (mirror of nequip.net models)")],
    hf_queries=("allegro nequip",),
    note="Allegro is loaded by the nequip package, same path as NequIP."))

_add(MLIP(
    "sevennet", "SevenNet (7net-0, MF-0, l3i5, ...)", "sevennet", "sevennet", "sevenn", "sevenn",
    (".pth", ".pt"), has_heads=True, accels=("e3nn", "cueq", "openeq", "flash"), layer_exchange=True,
    sources=[Source("package", "", "inside the installed sevenn package", path="pretrained_potentials"),
             Source("github_dir", "MDIL-SNU/SevenNet", "GitHub MDIL-SNU/SevenNet", path="sevenn/pretrained_potentials"),
             Source("github_release", "MDIL-SNU/SevenNet", "GitHub release MDIL-SNU/SevenNet")],
    note="--head is the modal of a multi-fidelity model (e.g. SevenNet-MF-0, omni). Five layers of 5 A: ghost atoms are 25 A deep, "
         "or 5 A with layer exchange (several GPUs)."))

_add(MLIP(
    "painn", "PaiNN (schnetpack)", "painn", None, "schnetpack", "schnetpack",
    (".pt", ".pth", "best_model"), needs_elements=True, layer_exchange=True,
    note="No public pretrained PaiNN potential was found: give the best_model file of your own schnetpack training. "
         "The atomic numbers the model was trained on are asked next (e.g. 1,6,7,8). Ghost atoms are n_interactions x cutoff deep, "
         "or one cutoff with layer exchange (several GPUs)."))

_add(MLIP(
    "chgnet", "CHGNet", "chgnet", "chgnet", "chgnet", "chgnet",
    (".pth.tar", ".pth", ".tar"), optional_model=True,
    sources=[Source("package", "", "the 0.3.0 weights inside the chgnet package (no file needed)"),
             Source("hf", "LostDecimal/chgnet-0.3.0-mirror", "Hugging Face (mirror)")],
    hf_queries=("chgnet",),
    note="No layer exchange for CHGNet: its bond and angle states sit on edges that two atoms share, so several GPUs need ghost "
         "atoms 4 x 6 = 24 A deep (edges that cannot matter are left out, as for the other models)."))

_add(MLIP(
    "mattersim", "MatterSim (M3GNet 1M / 5M)", "mattersim", "mattersim", "mattersim", "mattersim",
    (".pth",), layer_exchange=True,
    sources=[Source("github_dir", "microsoft/mattersim", "GitHub microsoft/mattersim", path="pretrained_models")],
    hf_queries=("mattersim",),
    note="Four blocks of 5 A: ghost atoms are 20 A deep, or 5 A with layer exchange (several GPUs)."))

_add(MLIP(
    "deepmd", "DeePMD-kit (PyTorch checkpoint)", "deepmd", "deepmd", "deepmd", "deepmd-kit",
    (".pt", ".pth"), fixed_dtype=True, layer_exchange=True, layer_exchange_local=True,
    hf_queries=("deepmd", "dpa"),
    note="Local descriptors only (se_e2_a, se_r, se_atten). Give a training checkpoint .pt. DPA-2/DPA-3 and frozen models are not supported by the converter. "
         "Layer exchange (several GPUs) exchanges nothing for this model (ghost atoms are one cutoff deep anyway): a rank computes only its own atoms."))

_add(MLIP(
    "tensornet", "TensorNet / M3GNet / CHGNet-matgl (matgl model folder)", "tensornet", "tensornet", "matgl", "matgl",
    (), layer_exchange=True,
    sources=[Source("hf", "materialyze/TensorNet-PES-MatPES-PBE-2025.2", "Hugging Face", folder=True),
             Source("hf", "materialyze/TensorNet-PES-MatPES-r2SCAN-2025.2", "Hugging Face", folder=True),
             Source("hf", "materialyze/TensorNet-PES-MatPES-PBE-2025.2-m", "Hugging Face", folder=True),
             Source("hf", "materialyze/TensorNet-PES-MatPES-r2SCAN-2025.2-m", "Hugging Face", folder=True),
             Source("hf", "materialyze/TensorNet-PES-ANI-1x-Subset", "Hugging Face", folder=True)],
    hf_queries=("TensorNet",),
    note="The model is a folder (model.pt, state.pt, model.json). The converter takes extensive models only; "
         "a model whose data_mean is not 0 is refused with a reason. Ghost atoms are (blocks + 1) x cutoff deep (the embedding "
         "counts), or one cutoff with layer exchange (several GPUs)."))

_add(MLIP(
    "bam", "BAM (Bayesian Atoms Modeling, RACE; BAM-MP-core)", "bam", "bam", "bam_torch", "git+https://github.com/myung-group/BAM-torch",
    (".pkl",), layer_exchange=True, accels=("e3nn", "cueq", "openeq"),
    sources=[Source("hf", "myung-group/BAM-MP-core", "Hugging Face myung-group/BAM-MP-core")],
    hf_queries=("myung-group BAM", "BAM-torch"),
    note="Needs the bam_torch package (git clone https://github.com/myung-group/BAM-torch; pip install -e .). The model is a .pkl "
         "checkpoint (about 410 MB for BAM-MP-core). Ghost depth is nlayers x cutoff = 30 A for BAM-MP-core, or 6 A with layer exchange (several GPUs). "
         "LibTorch conversion: the 'race' model type only. Kernels: e3nn; cueq needs a checkpoint trained with cuEquivariance (its weights have another layout); "
         "openeq needs the 'fast' interaction block (same weights as e3nn). cueq and openeq have not been run with real "
         "BAM-MP-core weights or with layer exchange. The Python MLIP mode runs bam_torch's own calculator."))

_add(MLIP(
    "uma", "UMA (fairchem) -- Python MLIP mode only", None, "uma", "fairchem", "fairchem-core",
    (".pt",),
    sources=[Source("hf", "facebook/UMA", "Hugging Face facebook/UMA (gated)", gated=True)],
    note="No LibTorch converter exists for UMA (docs/mlip_converter_models.md 4.4). The weights need an accepted license "
         "on huggingface.co/facebook/UMA and an access token: set HF_TOKEN."))

_add(MLIP(
    "orb", "ORB -- Python MLIP mode only", None, "orb", "orb_models", "orb-models",
    (".ckpt",),
    sources=[Source("hf", "Kameshr/orb-v3-conservative-inf-omat-mirror", "Hugging Face (mirror)")],
    optional_model=True,
    note="The orb package fetches its own weights when no model file is given."))

_add(MLIP(
    "gap", "GAP / QUIP (.xml) -- Python MLIP mode only", None, "gap", "quippy", "quippy-ase",
    (".xml",), hf_queries=("GAP quip",),
    note="Give a GAP parameter file (.xml)."))
