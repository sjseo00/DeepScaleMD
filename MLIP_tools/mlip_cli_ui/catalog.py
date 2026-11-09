"""What the CLI knows about each MLIP: how to convert it, where its model files can be found, how the ASE mode names it.

Model files are found in five ways (see sources.py): Hugging Face repos (listed live), GitHub releases or folders (listed live),
files shipped inside the installed Python package, a local file, and an address typed by the user.
Nothing here downloads anything. The repos below are the ones checked on 2026-10-14; the UI lists what is in them at the moment.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Source:
    kind: str            # "hf" | "github_release" | "github_dir" | "package"
    ref: str             # hf: repo id | github_release: owner/name | github_dir: owner/name | package: the package to look in
                         # (empty: the MLIP's own `package`)
    label: str           # shown in the list, e.g. "Hugging Face (mirror)"
    path: str = ""       # github_dir: folder in the repo; package: folder inside the package, relative to the package root
    folder: bool = False  # hf: the model is the whole repo folder (matgl: model.json + state.pt + model.pt)
    gated: bool = False   # needs an access token (HF_TOKEN) and an accepted license


@dataclass
class MLIP:
    key: str                         # name in the menu
    title: str
    convert_arch: Optional[str]      # mlip_convert architecture; None = no LibTorch converter for it
    python_type: Optional[str]       # model_type in mlip_config of the ASE mode; None = no ASE adapter
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
                                     # sevennet, bam, mattersim, tensornet, painn, deepmd, so3lr, uma, esen; the toy test models are not in this menu).
                                     # Several GPUs: ghost atoms one cutoff deep, node features sent between the layers
    layer_exchange_local: bool = False  # the model is strictly local (one layer): --layer-exchange then exchanges nothing, it only makes
                                     # a rank compute its own atoms and not its ghost atoms too (deepmd)
    layer_exchange_accels: Tuple[str, ...] = ("e3nn",)   # kernels layer exchange has been checked with (others: a warning)
    ordinary_ranks: str = ""         # what several ranks do without --layer-exchange, when it is not "ghost atoms layers x cutoff deep"
    backends: Tuple[str, ...] = ("torchscript", "aoti", "pytorch")   # mlip_convert --backend values the model can be converted with
    system_options: Tuple[str, ...] = ()   # mlip_convert --composition / --charge / --spin the model takes (uma, orb)
    kspace: bool = False             # mlip_convert --kspace pme (so3lr): Ewald-split electrostatics and dispersion, k-space sums in dsMD
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
    "painn", "PaiNN (schnetpack)", "painn", "painn", "schnetpack", "schnetpack",
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
         "openeq needs the 'fast' interaction block (same weights as e3nn); with oeq_conv_fusion (BAM-MP-DeNS) it uses the fused convolution, "
         "which needs far less GPU memory per edge (0.15 MB against 0.60 MB with e3nn) and works with layer exchange. openeq was run with "
         "BAM-MP-DeNS only; cueq has not been run. The ASE mode runs bam_torch's own calculator."))

_add(MLIP(
    "uma", "UMA (fairchem)", "uma", "uma", "fairchem", "fairchem-core",
    (".pt",), has_heads=True, backends=("pytorch",), system_options=("composition", "charge", "spin"), layer_exchange=True,
    sources=[Source("hf", "facebook/UMA", "Hugging Face facebook/UMA (gated)", gated=True)],
    note="The weights need an accepted license on huggingface.co/facebook/UMA and an access token: set HF_TOKEN. The converter writes a "
         "PyTorch folder only (--backend pytorch). It fixes the mixture of experts for the composition of your whole MD system "
         "(asked below, e.g. H2000O1000) and for the task (--head: omat, omol, oc20, odac, omc). Ghost atoms are (layers + 1) x 6 A deep, or 6 A with layer exchange (several GPUs)."))

_add(MLIP(
    "esen", "eSEN (fairchem 2.x: OMol25, OC25, ODAC25 conserving models)", "esen", "esen", "fairchem", "fairchem-core",
    (".pt",), backends=("pytorch",), system_options=("composition", "charge", "spin"), exclude=r"(?i)direct", layer_exchange=True,
    sources=[Source("hf", "facebook/OMol25", "Hugging Face facebook/OMol25 (gated, molecules)", gated=True),
             Source("hf", "facebook/OC25", "Hugging Face facebook/OC25 (gated, catalysis)", gated=True),
             Source("hf", "facebook/ODAC25", "Hugging Face facebook/ODAC25 (gated, MOFs)", gated=True)],
    note="The weights need an accepted license on huggingface.co (facebook/OMol25, OC25 or ODAC25) and an access token: set HF_TOKEN. "
         "fairchem 2.x runs eSEN with the same network as UMA, without the mixture of experts, so one converted folder works for any "
         "system and --composition is not needed (the converter asks for it only if the model's energy has a mean). Only the "
         "'conserving' models (forces = gradient of the energy) are supported; 'direct' models are not listed. PyTorch folder only "
         "(--backend pytorch). OMol25 models take the total charge and spin multiplicity (Enter = neutral singlet). Ghost atoms are "
         "(layers + 1) x 6 A deep, or 6 A with layer exchange (several GPUs)."))

_add(MLIP(
    "orb", "ORB (orb-v3 conservative)", "orb", "orb", "orb_models", "orb-models",
    (".ckpt",), backends=("pytorch",), system_options=("charge", "spin"),
    sources=[Source("hf", "Kameshr/orb-v3-conservative-inf-omat-mirror", "Hugging Face (mirror)")],
    optional_model=True,
    note="Converter: orb-v3 conservative 'inf' models, PyTorch folder only (--backend pytorch). The energy head of ORB averages over all "
         "atoms of the system, so the model is not split over GPUs: with several ranks every rank computes all atoms. omol models also "
         "take --charge and --spin. In the ASE mode the orb package fetches its own weights when no model file is given."))

_add(MLIP(
    "so3lr", "SO3LR Tauri (so3lr-tauri-s/m/l: short-range network + electrostatics + dispersion)", "so3lr", None, "numpy", "numpy",
    ("params.pkl",), backends=("torchscript", "pytorch"), system_options=("charge", "spin"), accels=("e3nn", "so3lr_fused"), kspace=True,
    layer_exchange=True, layer_exchange_accels=("e3nn", "so3lr_fused"),
    ordinary_ranks="every rank computes all atoms (the model is not split): no speed-up from more GPUs; fine for one GPU",
    sources=[Source("package", "so3lr", "installed so3lr package", path="models")],
    note="SO3LR is a JAX model: the converter builds it again in PyTorch from the released weights, so converting needs numpy only (no "
         "JAX). A model is a folder (params.pkl + hyperparameters.json). The bundled models are listed when the so3lr package is installed; "
         "otherwise git clone https://github.com/general-molecular-simulations/so3lr and pick with 'L' the folder (or its params.pkl) "
         "so3lr/models/so3lr-tauri-m-2606 (s: faster, l: most accurate). The long-range cutoff of the electrostatics and dispersion is "
         "--cutoff (default 12 A, for periodic boxes; give --cutoff 100 as an extra argument for a molecule in vacuum). Several GPUs: "
         "ordinary = every rank computes all atoms (no speed-up); layer exchange = each rank computes its own atoms, ghost atoms one "
         "--cutoff deep, the ranks send node features between the layers and the charges before the long-range terms, and sum the "
         "total charge over the ranks (needs a dsMD with dsmd::allreduce_sum). Long range: the cutoff (default) or PME (--kspace pme: "
         "the reciprocal sums are computed by dsMD, the input needs 'mlip_kspace: pme'). Charge and spin multiplicity are asked below "
         "(Enter = neutral singlet). Kernels: e3nn = plain "
         "PyTorch; so3lr_fused = its attention as one CUDA operator (mlip_convert/so3lr_ops/, built at "
         "conversion with the CUDA toolkit: nvcc and ninja; convert on a GPU machine). The Tauri weights are CC BY-NC 4.0; the "
         "legacy 'so3lr' model is not supported."))

_add(MLIP(
    "gap", "GAP / QUIP (.xml) -- ASE mode only", None, "gap", "quippy", "quippy-ase",
    (".xml",), hf_queries=("GAP quip",),
    note="Give a GAP parameter file (.xml)."))
