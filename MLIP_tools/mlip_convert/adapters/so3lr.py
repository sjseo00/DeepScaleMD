"""SO3LR adapter (so3lr 0.2.0: the SO3LR Tauri models so3lr-tauri-s/m/l, releases 2602 and 2606; as read from its source,
github.com/general-molecular-simulations/so3lr).

SO3LR is written in JAX (flax). This adapter does not run JAX: it builds the same network again in PyTorch and loads the released
weights of a model folder (params.pkl: nested dicts of numpy arrays, read with an unpickler that accepts numpy arrays only) and its
hyperparameters.json. The model is then plain PyTorch and becomes a TorchScript file like the other adapters.

What so3lr does, which this adapter reproduces (so3lr/mlff/...):
  * Short range (nn/embed, nn/layer): edges closer than the model cutoff (4.5 A; 5 A for -l), vector r_ij = pos[j] - pos[i],
    i = edge_index[0] receives. Bernstein radial basis of the distance, the distance first floored per element pair where the NLH
    repulsion of the pair exceeds 5 eV (floor_descriptor_distance), PhysNet cutoff, real spherical harmonics l = 1..4.
    Atom embedding (+ charge and spin embeddings, SpookyNet style) / sqrt(number of embeddings). Each SO3krates layer: residual
    scalars (x = a * x + b * x0), attention over the edges for the invariant features (num_heads heads) and for the Euclidean
    variables (one head per degree), q/k RMS-normalized, RMSNorm, residual MLP, exchange block.
  * Heads on the final features: per-atom energy (the theory-level head 5 of 16, as so3lr's calculator picks it), partial charges,
    polarizability ratios (a0) and C6 ratios.
  * Physics terms, all in eV: NLH pair repulsion on the short-range edges; electrostatics of the partial charges, erf(r / 4)/r damped,
    shifted-force and switched to zero between 0.45 x and 1 x the long-range cutoff; dispersion C6/C8/C10 from the a0 and C6 ratios
    (free-atom tables, Casimir-Polder mixing, QDO damping), switched off over the last 2 A before the long-range cutoff. Pair terms are
    counted half on each atom (both directions are in the edge list).
  * The long-range cutoff is an inference setting in so3lr (its ASE calculator: 12 A, meant for periodic condensed phases; 100 A for gas
    phase): here it is --cutoff, the cutoff of the edge list dsMD builds. The short-range graph is the part of that list within the
    model cutoff.
  * The partial charges are made to sum to the total charge of the system: q_i = q_raw_i + (Q - sum q_raw) / N. That sum, and the
    charge and spin embeddings, run over the whole system, so by default this adapter sets whole_system: dsMD never splits the model
    over ranks (every rank computes all atoms). With --layer-exchange (forward_layered) the model is split: each rank computes its own
    atoms, its ghost atoms are one long-range cutoff deep, the node features go to the ghost copies between the layers and the
    charges, a0 and C6 ratios before the pair terms (dsmd::halo_exchange), and the whole-system sums are sums over the ranks
    (dsmd::allreduce_sum).
    --charge (default 0) and --spin (spin multiplicity, default 1; so3lr's number of unpaired electrons is multiplicity - 1) are fixed
    at conversion. For a neutral singlet the charge and spin embeddings are exactly zero.
  * --kspace pme (so3lr's k-space option, as the SO3LR Tauri paper ran its liquids and biomolecules): the electrostatics and the C6
    dispersion are split Ewald-style. The edge terms become their real-space parts (so3lr coulomb_erf_shifted_force_smooth_pme: the
    erf(r / (sqrt2 s)) / r of a Gaussian smearing s is left out; DispersionEnergySparse with do_dispersion_kspace: the geometric-mean
    C6 times the long-range r^-6 part is added back), and the reciprocal parts are the operator dsmd::kspace (kspace_op.py; in dsMD
    src/MLIPKspaceOp.cpp): smooth PME of the charges and of the dispersion charges sqrt(C6). Smearings (A, so3lr's defaults 4.0 and
    2.0) are fixed at conversion; the grid spacings and the B-spline order are dsMD settings (YAML), with defaults in metadata.json.
    The box must be periodic in all three directions.
Not supported (refused): the legacy SO3LR model (legacy_so3lr_bool: learnable ZBL, other heads), Ewald and P3M k-space, other layer
options than the released Tauri models use. The released weights are CC BY-NC 4.0.
"""
import ast
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .. import kspace_op as _kspace_op   # noqa: F401  (registers dsmd::kspace, called with --kspace pme)
from .base import MLIPAdapter

# ase.units (CODATA 2014, ase's default), the values so3lr uses
BOHR = 0.5291772105638411           # A
HARTREE = 27.211386024367243        # eV
FINE_STRUCTURE = 0.007297352566206496
KE = 14.399645351950548             # Coulomb constant, eV A / e^2 (so3lr's ke)

HEAD_INDEX = 5                      # theory-level head every released model is read from (so3lr model_registry.DEFAULT_HEAD_INDEX)
MAX_Z = 85                          # Tauri models: trained on H..At
PAIR_FLOOR_ENERGY = 5.0             # eV, so3lr embed_sparse.PAIR_FLOOR_ENERGY
PAIR_FLOOR_WIDTH = 0.05             # A, so3lr embed_sparse.PAIR_FLOOR_WIDTH
BERNSTEIN_GAMMA = 0.9448630629184640
DISPERSION_DAMPING = 2.0            # A, so3lr's dispersion_energy_cutoff_lr_damping at inference
DEFAULT_LR_CUTOFF = 12.0            # A, so3lr's ASE calculator default
ACCELS = ("e3nn", "so3lr_fused")   # e3nn: plain PyTorch (the name every model's default kernel choice has)
REPULSION_SWITCH = 1.5             # A, NLH repulsion switched off from 0 to here (so3lr NLHRepulsionSparse)
RADIAL_TABLE_NODES = 2048           # so3lr_fused: nodes of the radial filter tables (cubic Hermite in the basis distance)
# --kspace pme defaults: so3lr's (kspace_defaults.py): smearings and grid spacings (A); B-spline order of dsMD's smooth PME
KSPACE_SMEARING = 4.0
KSPACE_SPACING = 2.0
KSPACE_DISPERSION_SMEARING = 2.0
KSPACE_DISPERSION_SPACING = 3.0
KSPACE_ORDER = 4
MODEL_NAMES_HELP = "so3lr-tauri-s, so3lr-tauri-m, so3lr-tauri-l (or with the release: so3lr-tauri-m-2606)"


# ---------------------------------------------------------------------------------------------------------------- model files
class _NumpyOnly(pickle.Unpickler):
    """params.pkl holds dicts of numpy arrays: allow the numpy array reconstruction and nothing else."""
    def find_class(self, module, name):
        if module in ("numpy._core.multiarray", "numpy.core.multiarray") and name in ("_reconstruct", "scalar"):
            core = getattr(np, "_core", None) or np.core
            return getattr(core.multiarray, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"params.pkl: refusing to load {module}.{name} (only numpy arrays are expected)")


def _so3lr_package_dir() -> Optional[Path]:
    """The installed so3lr package folder, found without importing it (it would import JAX)."""
    import importlib.util
    spec = importlib.util.find_spec("so3lr")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def resolve_model_dir(model: str) -> Path:
    """A model folder (params.pkl + hyperparameters.json) or its params.pkl, or a model name of the installed so3lr package."""
    path = Path(model).expanduser()
    if path.is_file() and path.name == "params.pkl":
        path = path.parent
    if path.is_dir():
        if not (path / "params.pkl").is_file() or not (path / "hyperparameters.json").is_file():
            raise ValueError(f"SO3LR: {path} is not a model folder (it needs params.pkl and hyperparameters.json)")
        return path.resolve()
    package = _so3lr_package_dir()
    if package is None:
        raise ValueError(f"SO3LR: '{model}' is not a folder, and the so3lr package is not installed to look the name up. Give the model "
                         f"folder, e.g. <so3lr checkout>/so3lr/models/so3lr-tauri-m-2606")
    models = package / "models"
    aliases = {"so3lr-tauri-s": "so3lr-tauri-s-2606", "so3lr-tauri-m": "so3lr-tauri-m-2606", "so3lr-tauri-l": "so3lr-tauri-l-2606"}
    name = aliases.get(model, model)
    if not (models / name / "params.pkl").is_file():
        known = sorted(p.name for p in models.iterdir() if (p / "params.pkl").is_file()) if models.is_dir() else []
        raise ValueError(f"SO3LR: no model '{model}' in {models}; known: {', '.join(known)}")
    return (models / name).resolve()


def _tables_dir(model_dir: Path) -> Path:
    """so3lr/mlff/nn/observable of the checkout or package the model folder belongs to (the free-atom and NLH tables)."""
    candidates = [model_dir.parent.parent / "mlff" / "nn" / "observable"]
    package = _so3lr_package_dir()
    if package is not None:
        candidates.append(package / "mlff" / "nn" / "observable")
    for folder in candidates:
        if (folder / "dispersion_ref_data.py").is_file() and (folder / "nlh_ref.npz").is_file():
            return folder
    raise ValueError(f"SO3LR: cannot find so3lr's reference tables (mlff/nn/observable/dispersion_ref_data.py, nlh_ref.npz) next to "
                     f"{model_dir}; give a model folder inside an so3lr checkout, or install the so3lr package")


def _literal_arrays(source: Path, names) -> Dict[str, np.ndarray]:
    """name = np.array([...]) literals of a Python file, read without running it."""
    found = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names and isinstance(node.value, ast.Call) and node.value.args:
                found[name] = np.asarray(ast.literal_eval(node.value.args[0]), dtype=np.float64)
    missing = [n for n in names if n not in found]
    if missing:
        raise ValueError(f"SO3LR: {source} has no literal table {missing}")
    return found


def load_model_files(model_dir: Path):
    """(hyperparameters, params, tables) of a model folder."""
    hyper = json.loads((model_dir / "hyperparameters.json").read_text(encoding="utf-8"))
    with open(model_dir / "params.pkl", "rb") as handle:
        params = _NumpyOnly(handle).load()
    params = params.get("params", params)
    folder = _tables_dir(model_dir)
    tables = _literal_arrays(folder / "dispersion_ref_data.py", ("alphas", "C6_coef"))
    with np.load(folder / "nlh_ref.npz", allow_pickle=False) as nlh:
        tables["NLH_AA"] = np.asarray(nlh["NLH_AA"], dtype=np.float64)
        tables["NLH_BB"] = np.asarray(nlh["NLH_BB"], dtype=np.float64)
    return hyper, params, tables


def nlh_floor_distance(aa: np.ndarray, bb: np.ndarray, target_energy: float = PAIR_FLOOR_ENERGY) -> np.ndarray:
    """so3lr embed_sparse.nlh_reference_distance: per element pair, where the switched NLH repulsion reaches target_energy (bisection)."""
    def sigma(x):
        return np.where(x > 0, np.exp(-1.0 / np.where(x > 0, x, 1.0)), 0.0)

    def switch(x, x_on, x_off):
        c = (x - x_on) / (x_off - x_on)
        return sigma(1 - c) / (sigma(1 - c) + sigma(c))

    n = aa.shape[0]
    z = np.arange(1, n + 1)
    zi, zj = np.meshgrid(z, z, indexing="ij")
    lo, hi = np.full((n, n), 0.02), np.full((n, n), 1.5)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore", invalid="ignore"):
            y = np.sum(aa * np.exp(-bb * mid[..., None]), axis=-1)
            e = switch(mid, 0.0, 1.5) * KE * zi * zj / mid * y
        above = np.nan_to_num(e, nan=0.0) > target_energy
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def _version_of(model_dir: Path) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("so3lr")
    except Exception:       # noqa: BLE001  (not installed: read the checkout's pyproject.toml)
        pyproject = model_dir.parent.parent.parent / "pyproject.toml"
        if pyproject.is_file():
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                if line.startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


# ---------------------------------------------------------------------------------------------------------------- torch modules
def _linear(p: Dict[str, np.ndarray]) -> torch.nn.Linear:
    """flax Dense (kernel [in, out], optional bias) as torch Linear."""
    kernel = np.asarray(p["kernel"])
    layer = torch.nn.Linear(kernel.shape[0], kernel.shape[1], bias="bias" in p)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(kernel.T)).double())
        if "bias" in p:
            layer.bias.copy_(torch.from_numpy(np.asarray(p["bias"])).double())
    return layer


def _linear_concat(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> torch.nn.Linear:
    """Two flax Dense layers with the same input as one torch Linear whose outputs are a's, then b's."""
    return _linear({"kernel": np.concatenate([np.asarray(a["kernel"]), np.asarray(b["kernel"])], axis=1),
                    "bias": np.concatenate([np.asarray(a["bias"]), np.asarray(b["bias"])])})


def _param(array) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(np.asarray(array))).double(), requires_grad=False)


def _rms(x: Tensor) -> Tensor:
    """flax RMSNorm(use_scale=False) and so3lr's functional_rms_norm (epsilon 1e-6)."""
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)


def _switch(x: Tensor, x_on: float, x_off: float) -> Tensor:
    """so3lr switching_fn: 1 below x_on, 0 above x_off, smooth (exp(-1/u) steps) in between."""
    c = (x - x_on) / (x_off - x_on)
    a = 1.0 - c
    s_a = torch.where(a > 0, torch.exp(-1.0 / torch.where(a > 0, a, torch.ones_like(a))), torch.zeros_like(a))
    s_c = torch.where(c > 0, torch.exp(-1.0 / torch.where(c > 0, c, torch.ones_like(c))), torch.zeros_like(c))
    return s_a / (s_a + s_c)


def _spherical_harmonics(u: Tensor) -> Tensor:
    """so3lr sph_ops.spherical_basis for l = 1..4 of unit vectors [E, 3] -> [E, 3 + 5 + 7 + 9], m = -l..l in so3lr's order."""
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    pi = math.pi
    c1 = math.sqrt(3.0 / (4.0 * pi))
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return torch.stack([
        c1 * y, c1 * z, c1 * x,
        0.5 * math.sqrt(15.0 / pi) * x * y,
        0.5 * math.sqrt(15.0 / pi) * y * z,
        0.25 * math.sqrt(5.0 / pi) * (3.0 * z2 - 1.0),
        0.5 * math.sqrt(15.0 / pi) * x * z,
        0.25 * math.sqrt(15.0 / pi) * (x2 - y2),
        0.25 * math.sqrt(35.0 / (2.0 * pi)) * y * (3.0 * x2 - y2),
        0.5 * math.sqrt(105.0 / pi) * x * y * z,
        0.25 * math.sqrt(21.0 / (2.0 * pi)) * y * (5.0 * z2 - 1.0),
        0.25 * math.sqrt(7.0 / pi) * (5.0 * z2 * z - 3.0 * z),
        0.25 * math.sqrt(21.0 / (2.0 * pi)) * x * (5.0 * z2 - 1.0),
        0.25 * math.sqrt(105.0 / pi) * (x2 - y2) * z,
        0.25 * math.sqrt(35.0 / (2.0 * pi)) * x * (x2 - 3.0 * y2),
        0.75 * math.sqrt(35.0 / pi) * x * y * (x2 - y2),
        0.75 * math.sqrt(35.0 / (2.0 * pi)) * y * (3.0 * x2 - y2) * z,
        0.75 * math.sqrt(5.0 / pi) * x * y * (7.0 * z2 - 1.0),
        0.75 * math.sqrt(5.0 / (2.0 * pi)) * y * (7.0 * z2 * z - 3.0 * z),
        3.0 / 16.0 * math.sqrt(1.0 / pi) * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
        0.75 * math.sqrt(5.0 / (2.0 * pi)) * x * (7.0 * z2 * z - 3.0 * z),
        3.0 / 8.0 * math.sqrt(5.0 / pi) * (x2 - y2) * (7.0 * z2 - 1.0),
        0.75 * math.sqrt(35.0 / (2.0 * pi)) * x * (x2 - 3.0 * y2) * z,
        3.0 / 16.0 * math.sqrt(35.0 / pi) * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)),
    ], dim=1)


_YLM_OFFSET = {1: 0, 2: 3, 3: 8, 4: 15}


class _ChargeSpinEmbed(torch.nn.Module):
    """so3lr ChargeSpinEmbedSparse for one fixed total charge or number of unpaired electrons (psi)."""
    def __init__(self, p: Dict, psi: float, num_features: int):
        super().__init__()
        row = 1 if psi < 0 else 0          # flax Embed of psi // inf: -1 (the last row) for psi < 0, else 0
        self.psi = float(psi)
        self.inv_sqrt_f = 1.0 / math.sqrt(num_features)
        self.q = _param(p["Embed_0"]["embedding"])
        self.k = _param(np.asarray(p["Embed_1"]["embedding"])[row])
        self.v = _param(np.asarray(p["Embed_2"]["embedding"])[row])
        self.r0 = _param(p["Residual_0"]["layers_0"]["kernel"])
        self.r1 = _param(p["Residual_0"]["layers_1"]["kernel"])

    def forward(self, z: Tensor, owned: Tensor) -> Tensor:
        y = self.weights(z)
        return self.embed(y, torch.where(owned, y, torch.zeros_like(y)).sum())

    def weights(self, z: Tensor) -> Tensor:
        """Per-atom attention weights [N] of the total charge / spin over the atoms (not normalized)."""
        return torch.nn.functional.softplus((self.q[z] * self.k).sum(dim=-1) * self.inv_sqrt_f)

    def embed(self, y: Tensor, denominator: Tensor) -> Tensor:
        """The embedding [N, F] from the weights y [N] and their sum over the whole system."""
        a = self.psi * y / denominator
        x = a.unsqueeze(-1) * self.v
        h = torch.nn.functional.silu(x) @ self.r0
        h = torch.nn.functional.silu(h) @ self.r1
        return x + h


class _So3kratesLayer(torch.nn.Module):
    """so3lr SO3kratesLayerSparse (attention block, RMSNorm, residual MLP, exchange block) with its residual scalars."""
    def __init__(self, p: Dict, num_features: int, num_heads: int, num_degrees: int, norm1: float, norm2: float, qk_norm: bool,
                 layer_norm_1: bool, layer_norm_2: bool, resid_lambda: float, x0_lambda: float):
        super().__init__()
        a = p["attention_block"]
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_degrees = num_degrees
        self.norm1 = float(norm1)
        self.norm2 = float(norm2)
        self.qk_norm = qk_norm
        self.layer_norm_1 = layer_norm_1
        self.layer_norm_2 = layer_norm_2
        self.resid_lambda = float(resid_lambda)
        self.x0_lambda = float(x0_lambda)
        # the first layers of the two radial filters (and of the two spherical filters) as one Linear each: same numbers, one GEMM
        self.rf_1 = _linear_concat(a["radial_filter1_layer_1"], a["radial_filter2_layer_1"])
        self.sf_1 = _linear_concat(a["spherical_filter1_layer_1"], a["spherical_filter2_layer_1"])
        self.rf_hidden = int(np.asarray(a["radial_filter1_layer_1"]["kernel"]).shape[1])
        self.sf_hidden = int(np.asarray(a["spherical_filter1_layer_1"]["kernel"]).shape[1])
        self.rf1_2 = _linear(a["radial_filter1_layer_2"])
        self.rf2_2 = _linear(a["radial_filter2_layer_2"])
        self.sf1_2 = _linear(a["spherical_filter1_layer_2"])
        self.sf2_2 = _linear(a["spherical_filter2_layer_2"])
        self.wq1 = _param(a["Wq1"])
        self.wk1 = _param(a["Wk1"])
        self.wv1 = _param(a["Wv1"])
        self.wq2 = _param(a["Wq2"])
        self.wk2 = _param(a["Wk2"])
        self.res1 = _linear(p["res_mlp_1_layer_1"])
        self.res2 = _linear(p["res_mlp_1_layer_2"])
        self.exchange = _linear(p["exchange_block"]["mlp_layer_2"])

    def forward(self, x: Tensor, ev: Tensor, x0: Tensor, radial: Tensor, ylm: Tensor, cut: Tensor, idx_i: Tensor, idx_j: Tensor,
                contraction: Tensor, degree_index: Tensor, cg: Tensor):
        """radial: the radial basis [E, n_rbf] (here), the basis distance [E] (_So3kratesLayerFused, which reads the filters from
        a table)."""
        silu = torch.nn.functional.silu
        x = self.resid_lambda * x + self.x0_lambda * x0
        n = x.shape[0]
        f = self.num_features
        h = self.num_heads
        d = self.num_degrees
        # AttentionBlock
        sph = self._sph(ev, idx_i, idx_j, contraction, degree_index, cg)        # [E, degrees]
        x1 = x.reshape(n, h, f // h)
        x2 = x.reshape(n, d, f // d)
        q1 = torch.einsum("hij,nhj->nhi", self.wq1, x1)
        k1 = torch.einsum("hij,nhj->nhi", self.wk1, x1)
        q2 = torch.einsum("hij,nhj->nhi", self.wq2, x2)
        k2 = torch.einsum("hij,nhj->nhi", self.wk2, x2)
        if self.qk_norm:
            # so3lr normalizes after the gather to the edges; the norm is per row, so normalizing the atoms first is the same
            q1 = _rms(q1)
            k1 = _rms(k1)
            q2 = _rms(q2)
            k2 = _rms(k2)
        v1 = torch.einsum("hij,nhj->nhi", self.wv1, x1)
        x_att, ev_att = self._filtered_attention(q1, k1, v1, q2, k2, radial, sph, cut, ylm, idx_i, idx_j, degree_index)
        x = x + x_att.reshape(n, f)
        ev = ev + ev_att
        if self.layer_norm_1:
            x = _rms(x)
        x = x + self.res2(silu(self.res1(silu(x))))
        # ExchangeBlock
        y = self.exchange(torch.cat([x, (ev * ev) @ contraction], dim=-1))
        x = x + y[:, :f]
        ev = ev + y[:, f:].index_select(1, degree_index) * ev
        if self.layer_norm_2:
            x = _rms(x)
        return x, ev

    def _filtered_attention(self, q1: Tensor, k1: Tensor, v1: Tensor, q2: Tensor, k2: Tensor, radial: Tensor, sph: Tensor, cut: Tensor,
                            ylm: Tensor, idx_i: Tensor, idx_j: Tensor, degree_index: Tensor):
        """The per-edge filters of both kinds of heads (radial filter + spherical filter MLP), then the attention (_attention)."""
        silu = torch.nn.functional.silu
        e = idx_i.shape[0]
        f = self.num_features
        wr = self._radial(radial)                                               # [E, 2F]: the radial parts of both filters
        hs = silu(self.sf_1(sph))
        w1 = (wr[:, :f] + self.sf1_2(hs[:, :self.sf_hidden])).reshape(e, self.num_heads, f // self.num_heads)
        w2 = (wr[:, f:] + self.sf2_2(hs[:, self.sf_hidden:])).reshape(e, self.num_degrees, f // self.num_degrees)
        return self._attention(q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index)

    def _radial(self, rbf: Tensor) -> Tensor:
        """Both radial filters of the layer on the radial basis [E, n_rbf] -> [E, 2F] (filter 1, then filter 2)."""
        hr = torch.nn.functional.silu(self.rf_1(rbf))
        return torch.cat([self.rf1_2(hr[:, :self.rf_hidden]), self.rf2_2(hr[:, self.rf_hidden:])], dim=1)

    def _sph(self, ev: Tensor, idx_i: Tensor, idx_j: Tensor, contraction: Tensor, degree_index: Tensor, cg: Tensor) -> Tensor:
        """Degree-wise l = 0 contraction of the Euclidean-variable differences of the edges [E, degrees] (the spherical filters' input)."""
        # index_select, not ev[idx]: the backward of advanced indexing sorts the indices (slow on the GPU), that of index_select
        # is an atomic index_add
        d_ev = ev.index_select(0, idx_j) - ev.index_select(0, idx_i)
        return (d_ev * d_ev) @ contraction

    def _attention(self, q1: Tensor, k1: Tensor, v1: Tensor, w1: Tensor, q2: Tensor, k2: Tensor, w2: Tensor, cut: Tensor, ylm: Tensor,
                   idx_i: Tensor, idx_j: Tensor, degree_index: Tensor):
        """The edge part of the attention block: per-atom q, k, v [N, heads, channels] and per-edge filters w [E, heads, channels]
        -> x_att [N, heads, channels], ev_att [N, ylm columns]. _So3kratesLayerFused does the same in one CUDA operator."""
        alpha1 = (q1.index_select(0, idx_i) * w1 * k1.index_select(0, idx_j)).sum(dim=-1) / self.norm1 * cut.unsqueeze(-1)  # [E, heads]
        alpha2 = (q2.index_select(0, idx_i) * w2 * k2.index_select(0, idx_j)).sum(dim=-1) / self.norm2 * cut.unsqueeze(-1)  # [E, degrees]
        x_att = torch.zeros_like(q1).index_add_(0, idx_i, alpha1.unsqueeze(-1) * v1.index_select(0, idx_j))
        ev_att = torch.zeros((q1.shape[0], ylm.shape[1]), dtype=ylm.dtype, device=ylm.device).index_add_(
            0, idx_i, alpha2.index_select(1, degree_index) * ylm)
        return x_att, ev_att


class _So3kratesLayerFused(_So3kratesLayer):
    """The same layer with the edge part of the attention in the CUDA operator so3lr::attention (mlip_convert/so3lr_ops/, accel
    so3lr_fused): the operator so3lr::attention_filtered puts each edge's filters together inside the kernel: the radial parts read
    from a table in the basis distance (filled by SO3LRAdapterFused), plus the spherical parts, which stay matrix products (both
    second layers as one block-diagonal product). No radial filter array and no filter sum is stored. The library must be loaded
    (mlip_convert.accel) before this layer is built or scripted."""
    table_x0: float
    table_inv_h: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f = self.num_features
        if not (f == 32 * self.num_heads == 32 * self.num_degrees and f <= 256):
            raise ValueError(f"SO3LR --accel so3lr_fused needs 32 channels per head and as many degree heads as heads (features {f}, "
                             f"heads {self.num_heads}, degree heads {self.num_degrees}); use --accel e3nn")
        self.register_buffer("radial_table", torch.zeros((2, 2, 2 * f), dtype=torch.float64))
        self.table_x0 = 0.0
        self.table_inv_h = 1.0
        # second layers of both spherical filters as one Linear on the hidden units of both: [2F, 2 hidden], block diagonal
        sh = self.sf_hidden
        block = torch.zeros((2 * f, 2 * sh), dtype=torch.float64)
        block[:f, :sh] = self.sf1_2.weight.detach().double()
        block[f:, sh:] = self.sf2_2.weight.detach().double()
        self.register_buffer("sf_block_weight", block)
        self.register_buffer("sf_block_bias", torch.cat([self.sf1_2.bias.detach().double(), self.sf2_2.bias.detach().double()]))

    def set_radial_table(self, basis, lo: float, hi: float, nodes: int) -> None:
        """Tabulate both radial filters (_So3kratesLayer._radial) at `nodes` evenly spaced basis distances in [lo, hi], values and
        derivatives (forward-mode autograd, float64); basis(d) is the radial basis [n, n_rbf] of basis distances d [n]."""
        import torch.autograd.forward_ad as fwad

        def dense(layer: torch.nn.Linear, v: Tensor) -> Tensor:        # in float64, whatever dtype the layer has now
            return torch.nn.functional.linear(v, layer.weight.double(), layer.bias.double())

        x = torch.linspace(lo, hi, nodes, dtype=torch.float64, device=self.radial_table.device)
        with torch.no_grad(), fwad.dual_level():
            hr = torch.nn.functional.silu(dense(self.rf_1, basis(fwad.make_dual(x, torch.ones_like(x)))))
            out = torch.cat([dense(self.rf1_2, hr[:, :self.rf_hidden]), dense(self.rf2_2, hr[:, self.rf_hidden:])], dim=1)
            value, derivative = fwad.unpack_dual(out)
        self.radial_table = torch.stack([value, derivative], dim=1).to(self.radial_table.dtype).contiguous()
        self.table_x0 = float(lo)
        self.table_inv_h = (nodes - 1) / (hi - lo)

    def _radial(self, rbf: Tensor) -> Tensor:
        # rbf is the basis distance [E] here (SO3LRAdapterFused._radial_input)
        return torch.ops.so3lr.radial_filter(rbf, self.radial_table, self.table_x0, self.table_inv_h)

    def _filtered_attention(self, q1: Tensor, k1: Tensor, v1: Tensor, q2: Tensor, k2: Tensor, radial: Tensor, sph: Tensor, cut: Tensor,
                            ylm: Tensor, idx_i: Tensor, idx_j: Tensor, degree_index: Tensor):
        s = torch.nn.functional.linear(torch.nn.functional.silu(self.sf_1(sph)), self.sf_block_weight, self.sf_block_bias)  # [E, 2F]
        out = torch.ops.so3lr.attention_filtered(q1, k1, v1, q2, k2, cut, ylm, radial, s, idx_i, idx_j, degree_index, self.radial_table,
                                                 self.table_x0, self.table_inv_h, self.norm1, self.norm2)
        return out[0], out[1]

    def _attention(self, q1: Tensor, k1: Tensor, v1: Tensor, w1: Tensor, q2: Tensor, k2: Tensor, w2: Tensor, cut: Tensor, ylm: Tensor,
                   idx_i: Tensor, idx_j: Tensor, degree_index: Tensor):
        out = torch.ops.so3lr.attention(q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index, self.norm1, self.norm2)
        return out[0], out[1]

    def _sph(self, ev: Tensor, idx_i: Tensor, idx_j: Tensor, contraction: Tensor, degree_index: Tensor, cg: Tensor) -> Tensor:
        return torch.ops.so3lr.ev_contraction(ev, idx_i, idx_j, degree_index, cg, self.num_degrees)


class _Head(torch.nn.Module):
    """so3lr's per-atom readouts: Dense -> silu -> Dense(1) + element embedding."""
    def __init__(self, regression: Dict, final: Dict, embedding):
        super().__init__()
        self.regression = _linear(regression)
        self.final = _linear(final)
        self.embedding = _param(np.asarray(embedding).reshape(-1))

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        return self.final(torch.nn.functional.silu(self.regression(x))).squeeze(-1) + self.embedding[z]


class _NoKspace(torch.nn.Module):
    """The k-space term of a model without --kspace: nothing (the model never calls dsmd::kspace)."""
    def forward(self, pos: Tensor, q: Tensor, c6: Tensor) -> Tensor:
        return torch.zeros_like(q)


class _PmeKspace(torch.nn.Module):
    """--kspace pme: the reciprocal parts of the electrostatics (charges q) and of the C6 dispersion (charges sqrt(C6), C6 in
    atomic units), per atom, from the operator dsmd::kspace (box, grid and ranks come from the caller's context)."""
    params: List[float]

    def __init__(self, params: List[float]):
        super().__init__()
        self.params = [float(v) for v in params]

    def forward(self, pos: Tensor, q: Tensor, c6: Tensor) -> Tensor:
        charges = torch.stack([q, torch.sqrt(torch.clamp(c6, min=0.0))], dim=1)
        return torch.ops.dsmd.kspace(pos.contiguous(), charges.contiguous(), self.params)


class SO3LRAdapter(MLIPAdapter):
    __jit_unused_properties__ = ["ghost_cutoff", "reference_elements", "reference_ghost_depth", "kspace_metadata"]
    model_family = "so3lr"
    supports_shifts = True
    whole_system = True                 # off with --layer-exchange (enable_layer_exchange)
    supports_layer_exchange = True
    scriptable = True

    def __init__(self, hyper: Dict, params: Dict, tables: Dict[str, np.ndarray], model_name: str, lr_cutoff: Optional[float] = None,
                 charge: Optional[int] = None, spin: Optional[int] = None, versions: Optional[Dict[str, Optional[str]]] = None,
                 accel: str = "e3nn", kspace: Optional[str] = None, kspace_smearing: Optional[float] = None,
                 kspace_dispersion_smearing: Optional[float] = None, kspace_spacing: Optional[float] = None,
                 kspace_dispersion_spacing: Optional[float] = None, kspace_order: Optional[int] = None):
        m = hyper["model"]
        if accel not in ACCELS:
            raise ValueError(f"SO3LR: --accel {accel} is not available; supported: {', '.join(ACCELS)} (e3nn = plain PyTorch)")
        _check_supported(m)
        super().__init__(list(range(1, MAX_Z + 1)))
        self.model_name = model_name
        self.accel = accel
        self.sr_cutoff = float(m["cutoff"])
        self.lr_cutoff = float(DEFAULT_LR_CUTOFF if lr_cutoff is None else lr_cutoff)
        if not self.lr_cutoff > max(self.sr_cutoff, DISPERSION_DAMPING):
            raise ValueError(f"SO3LR: the long-range cutoff (--cutoff {self.lr_cutoff:g}) must be larger than the model cutoff "
                             f"({self.sr_cutoff:g} A) and the dispersion damping width ({DISPERSION_DAMPING:g} A)")
        self.cutoff = self.lr_cutoff                    # the edge list holds every pair the long-range terms need
        self.num_layers = int(m["num_layers"])
        self.charge = 0 if charge is None else int(charge)
        multiplicity = 1 if spin is None else int(spin)
        if multiplicity < 1:
            raise ValueError(f"SO3LR: --spin is the spin multiplicity (1 = singlet), got {multiplicity}")
        self.unpaired = multiplicity - 1
        self._versions = versions or {}

        degrees = [int(v) for v in m["degrees"]]
        if any(v not in _YLM_OFFSET for v in degrees):
            raise ValueError(f"SO3LR: spherical harmonics degrees {degrees}: only 1..4 exist in so3lr")
        num_features = int(m["num_features"])
        num_heads = int(m["num_heads"])
        num_degrees = len(degrees)
        self.num_features = num_features
        columns: List[int] = []
        degree_index: List[int] = []
        cg: List[float] = []
        for k, degree in enumerate(degrees):
            size = 2 * degree + 1
            columns += list(range(_YLM_OFFSET[degree], _YLM_OFFSET[degree] + size))
            degree_index += [k] * size
            cg += [math.sqrt(size) / size] * size      # so3lr make_l0_contraction_fn
        self.register_buffer("ylm_columns", torch.tensor(columns, dtype=torch.int64))
        self.register_buffer("degree_index", torch.tensor(degree_index, dtype=torch.int64))
        contraction = torch.zeros((len(columns), num_degrees), dtype=torch.float64)
        contraction[torch.arange(len(columns)), torch.tensor(degree_index)] = torch.tensor(cg, dtype=torch.float64)
        self.register_buffer("contraction", contraction)
        self.register_buffer("ylm_cg", torch.tensor(cg, dtype=torch.float64))                  # the nonzero of each contraction row
        self.ylm_identity = columns == list(range(len(columns)))                              # degrees 1, 2, 3, 4: no column selection

        # radial basis (Bernstein, n_rbf functions)
        n_rbf = int(m["num_radial_basis_fn"])
        log_binomial = [math.lgamma(n_rbf) - math.lgamma(k + 1) - math.lgamma(n_rbf - k) for k in range(n_rbf)]
        self.register_buffer("rbf_log_binomial", torch.tensor(log_binomial, dtype=torch.float64))
        self.register_buffer("rbf_k", torch.arange(n_rbf, dtype=torch.float64))
        self.register_buffer("rbf_k_rev", torch.arange(n_rbf, dtype=torch.float64).flip(0))

        # embeddings
        self.atom_embedding = _param(params["feature_embeddings_0"]["Embed_0"]["embedding"])
        index = 1
        self.use_charge_embed = bool(m.get("use_charge_embed", False))
        self.use_spin_embed = bool(m.get("use_spin_embed", False))
        if self.use_charge_embed:
            self.charge_embed = _ChargeSpinEmbed(params[f"feature_embeddings_{index}"]["ChargeSpinEmbedSparse_0"], self.charge, num_features)
            index += 1
        else:
            self.charge_embed = _ChargeSpinEmbed(_dummy_charge_spin(num_features), 0.0, num_features)
        if self.use_spin_embed:
            self.spin_embed = _ChargeSpinEmbed(params[f"feature_embeddings_{index}"]["ChargeSpinEmbedSparse_0"], self.unpaired, num_features)
            index += 1
        else:
            self.spin_embed = _ChargeSpinEmbed(_dummy_charge_spin(num_features), 0.0, num_features)
        self.embed_scale = 1.0 / math.sqrt(index)

        # layers
        if m["message_normalization"] == "avg_num_neighbors":
            norm1 = norm2 = float(hyper["data"]["avg_num_neighbors"])
        elif m["message_normalization"] == "sqrt_num_features":
            norm1, norm2 = math.sqrt(num_features // num_heads), math.sqrt(num_features // num_degrees)
        else:
            norm1 = norm2 = 1.0
        use_scalars = bool(m.get("use_residual_scalars", False))
        lambdas = np.asarray(params["resid_lambdas"]) if use_scalars else np.ones(self.num_layers)
        x0_lambdas = np.asarray(params["x0_lambdas"]) if use_scalars else np.zeros(self.num_layers)
        layer_class = _So3kratesLayerFused if accel == "so3lr_fused" else _So3kratesLayer
        self.layers = torch.nn.ModuleList([
            layer_class(params[f"layers_{i}"], num_features, num_heads, num_degrees, norm1, norm2, bool(m.get("qk_norm", False)),
                            bool(m.get("layer_normalization_1", False)), bool(m.get("layer_normalization_2", False)),
                            float(lambdas[i]), float(x0_lambdas[i]))
            for i in range(self.num_layers)])

        # heads
        obs0 = params["observables_0"]
        self.energy_regression = _linear(obs0["energy_dense_regression"])
        final = obs0["energy_dense_final"]
        levels = np.asarray(final["kernel"]).shape[1]
        if levels <= HEAD_INDEX:
            raise ValueError(f"SO3LR: the energy head has {levels} theory levels, so3lr reads level {HEAD_INDEX}")
        self.energy_final = _param(np.asarray(final["kernel"])[:, HEAD_INDEX])
        self.energy_final_bias = float(np.asarray(final["bias"])[HEAD_INDEX]) if "bias" in final else 0.0
        self.atomic_scale = _param(_levels(obs0, "atomic_scales", 1.0)[:, HEAD_INDEX])
        self.atomic_offset = _param(_levels(obs0, "energy_offset", 0.0)[:, HEAD_INDEX])
        pq = obs0["electrostatic_energy"]["partial_charges"]
        self.charge_head = _Head(pq["charge_dense_regression_vec"], pq["charge_dense_final_vec"], pq["Embed_0"]["embedding"])
        o2 = params["observables_2"]
        self.a0_head = _Head(o2["hirshfeld_ratios_dense_regression"], o2["hirshfeld_ratios_dense_final"], o2["Embed_0"]["embedding"])
        o3 = params["observables_3"]
        self.c6_head = _Head(o3["c6_ratios_dense_regression"], o3["c6_ratios_dense_final"], o3["Embed_0"]["embedding"])

        # physics terms
        self.use_repulsion = bool(m.get("zbl_repulsion_bool", False))
        self.use_electrostatics = bool(m.get("electrostatic_energy_bool", False))
        self.use_dispersion = bool(m.get("dispersion_energy_bool", False))
        self.register_buffer("nlh_a", torch.from_numpy(tables["NLH_AA"]))
        self.register_buffer("nlh_b", torch.from_numpy(tables["NLH_BB"]))
        self.register_buffer("floor_distance", torch.from_numpy(nlh_floor_distance(tables["NLH_AA"], tables["NLH_BB"])))
        self.register_buffer("free_alpha", torch.from_numpy(tables["alphas"]))
        self.register_buffer("free_c6", torch.from_numpy(tables["C6_coef"]))
        if self.free_alpha.shape[0] < MAX_Z or self.free_c6.shape[0] < MAX_Z or self.nlh_a.shape[0] < MAX_Z:
            raise ValueError("SO3LR: the reference tables do not cover Z = 1..85")
        self.coulomb_sigma = float(m["electrostatic_energy_scale"])
        self.dispersion_scale = float(m["dispersion_energy_scale"])
        # erf(r/sigma)/r shifted-force constants at the long-range cutoff (so3lr coulomb_erf_shifted_force_smooth)
        sigma = self.coulomb_sigma
        rc = self.lr_cutoff
        # --kspace pme
        if kspace not in (None, "none", "pme"):
            raise ValueError(f"SO3LR: --kspace {kspace} is not supported (pme or none)")
        self.use_kspace = kspace == "pme"
        self.kspace_smearing = float(KSPACE_SMEARING if kspace_smearing is None else kspace_smearing)
        self.kspace_dispersion_smearing = float(KSPACE_DISPERSION_SMEARING if kspace_dispersion_smearing is None
                                                else kspace_dispersion_smearing)
        self.kspace_spacing = float(KSPACE_SPACING if kspace_spacing is None else kspace_spacing)
        self.kspace_dispersion_spacing = float(KSPACE_DISPERSION_SPACING if kspace_dispersion_spacing is None else kspace_dispersion_spacing)
        self.kspace_order = int(KSPACE_ORDER if kspace_order is None else kspace_order)
        if self.use_kspace and not (self.kspace_smearing > 0 and self.kspace_dispersion_smearing > 0 and self.kspace_spacing > 0
                                    and self.kspace_dispersion_spacing > 0 and self.kspace_order in (4, 6)):
            raise ValueError("SO3LR --kspace pme: smearings and grid spacings must be positive and --kspace-order 4 or 6")
        # the Gaussian smearing of the electrostatic Ewald split, as erf(r / pme_smear) / r (0: no split)
        self.pme_smear = math.sqrt(2.0) * self.kspace_smearing if self.use_kspace else 0.0
        self.disp_smear_bohr = self.kspace_dispersion_smearing / BOHR if self.use_kspace else 0.0

        def potential(r: float) -> float:
            v = math.erf(r / sigma) / r
            return v - math.erf(r / self.pme_smear) / r if self.pme_smear > 0 else v

        def derivative(r: float, width: float) -> float:          # d/dr erf(r / width) / r
            return (2.0 * r * math.exp(-(r / width) ** 2) / (math.sqrt(math.pi) * width) - math.erf(r / width)) / r ** 2

        self.coulomb_shift = potential(rc)
        self.coulomb_force_shift = derivative(rc, sigma) - (derivative(rc, self.pme_smear) if self.pme_smear > 0 else 0.0)
        self.coulomb_cuton = 0.45 * rc
        self.vdw_radius_factor = FINE_STRUCTURE ** (-4.0 / 21.0)
        # constants used in forward, as attributes (TorchScript and module-level floats do not mix well)
        self.ke = KE
        self.bohr = BOHR
        self.hartree = HARTREE
        self.floor_width = PAIR_FLOOR_WIDTH
        self.bernstein_gamma = BERNSTEIN_GAMMA
        self.dispersion_damping = DISPERSION_DAMPING
        self.repulsion_switch = REPULSION_SWITCH
        if self.use_kspace:
            self.kspace_term = _PmeKspace([KE, self.kspace_smearing, self.kspace_dispersion_smearing, BOHR, HARTREE,
                                           1.0 if self.use_electrostatics else 0.0, 1.0 if self.use_dispersion else 0.0])
        else:
            self.kspace_term = _NoKspace()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def kspace_metadata(self) -> Optional[Dict]:
        """metadata.json "kspace" (None without --kspace): what dsMD must compute and its default grid settings."""
        if not self.use_kspace:
            return None
        return {"method": "pme", "electrostatics": bool(self.use_electrostatics), "dispersion": bool(self.use_dispersion),
                "smearing": self.kspace_smearing, "dispersion_smearing": self.kspace_dispersion_smearing,
                "grid_spacing": self.kspace_spacing, "dispersion_grid_spacing": self.kspace_dispersion_spacing,
                "order": self.kspace_order}

    @classmethod
    def from_model(cls, model: str, lr_cutoff: Optional[float] = None, charge: Optional[int] = None,
                   spin: Optional[int] = None, accel: str = "e3nn", **kspace) -> "SO3LRAdapter":
        model_dir = resolve_model_dir(model)
        hyper, params, tables = load_model_files(model_dir)
        digest = hashlib.sha256((model_dir / "params.pkl").read_bytes()).hexdigest()[:16]
        klass = SO3LRAdapterFused if accel == "so3lr_fused" else cls
        return klass(hyper, params, tables, model_dir.name, lr_cutoff=lr_cutoff, charge=charge, spin=spin,
                     versions={"so3lr": _version_of(model_dir), "so3lr_params_sha256": digest}, accel=accel, **kspace)

    def package_versions(self) -> Dict[str, Optional[str]]:
        return dict(self._versions)

    @property
    def reference_elements(self):
        return [6, 1, 7, 8]

    def enable_layer_exchange(self) -> None:
        """--layer-exchange: the model is split over the ranks (forward_layered). Edges are summed onto row 0 (i receives); the ghost
        atoms are one long-range cutoff deep (the pair terms read the charges and ratios of atoms up to that far, which their owners
        send; the short-range layers need less)."""
        super().enable_layer_exchange()
        self.whole_system = False
        self.edge_receiver_row = 0

    @property
    def ghost_cutoff(self) -> float:
        """Ghost depth for images given as atoms: the long-range edges reach charges and ratios that depend on num_layers short-range
        hops more (lr + layers x sr); never less than num_layers x cutoff, which the MD program asks of every model. With layer
        exchange: one long-range cutoff."""
        if self.layer_exchange:
            return self.cutoff
        return max(self.num_layers * self.cutoff, self.reference_ghost_depth)

    @property
    def reference_ghost_depth(self) -> float:
        if self.layer_exchange:
            return self.cutoff
        return self.lr_cutoff + self.num_layers * self.sr_cutoff

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        no_shift = torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device)
        return self.forward_whole_system(pos, species, edge_index, no_shift, pos.shape[0])

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        return self.forward_whole_system(pos, species, edge_index, edge_shift, pos.shape[0])

    def forward_whole_system(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        silu = torch.nn.functional.silu
        n_atoms = pos.shape[0]
        z = species + 1                                          # elements are 1..85 in order: species = Z - 1
        owned = torch.arange(n_atoms, device=pos.device) < n_owned
        idx_i_lr = edge_index[0]
        idx_j_lr = edge_index[1]
        idx_i, idx_j, vec, d, d_lr = self._edges(pos, edge_shift, idx_i_lr, idx_j_lr)
        ylm, d_basis, cut, e_rep = self._geometry(vec, d, idx_i, idx_j, z, n_atoms)
        radial = self._radial_input(d_basis)

        # embeddings
        x0 = self.atom_embedding[z]
        if self.use_charge_embed:
            x0 = x0 + self.charge_embed(z, owned)
        if self.use_spin_embed:
            x0 = x0 + self.spin_embed(z, owned)
        x0 = x0 * self.embed_scale
        x = x0
        ev = torch.zeros((n_atoms, self.ylm_columns.shape[0]), dtype=pos.dtype, device=pos.device)
        for layer in self.layers:
            x, ev = layer(x, ev, x0, radial, ylm, cut, idx_i, idx_j, self.contraction, self.degree_index, self.ylm_cg)

        # per-atom energy of theory-level head 5
        hidden = silu(self.energy_regression(x))
        energy = (hidden @ self.energy_final + self.energy_final_bias) * self.atomic_scale[z] + self.atomic_offset[z]

        if self.use_repulsion:          # NLH repulsion (so3lr NLHRepulsionSparse), short-range edges
            energy = energy + e_rep

        if self.use_electrostatics or self.use_dispersion:
            # per-atom inputs of the pair terms: charges (summed to the total charge over the whole system), alpha and C6
            if self.use_electrostatics:
                q_raw = self.charge_head(x, z)
                n_whole = float(n_owned)
                correction = (float(self.charge) - torch.where(owned, q_raw, torch.zeros_like(q_raw)).sum()) / n_whole
                q = q_raw + correction
            else:
                q = torch.zeros_like(energy)
            if self.use_dispersion:
                alpha = self.free_alpha[z - 1] * torch.abs(self.a0_head(x, z))
                c6 = self.free_c6[z - 1] * torch.abs(self.c6_head(x, z))
            else:
                alpha = torch.ones_like(energy)
                c6 = torch.ones_like(energy)
            energy = energy + self._long_range(pos, edge_shift, idx_i_lr, idx_j_lr, d_lr, q, alpha, c6)
            if self.use_kspace:         # reciprocal parts, owned atoms (the first n_owned rows)
                e_k = self.kspace_term(pos[:n_owned], q[:n_owned], c6[:n_owned])
                energy = energy + torch.cat([e_k, torch.zeros(n_atoms - n_owned, dtype=e_k.dtype, device=e_k.device)])

        return torch.where(owned, energy, torch.zeros_like(energy))

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """forward_whole_system split over the ranks (--layer-exchange). The atoms are this rank's n_owned atoms, then ghost copies
        one long-range cutoff deep; edge_index has only the edges whose atom i is owned. The layers compute all rows, but only the
        owned rows are right after a layer: before the next one the owners send theirs to the ghost copies (dsmd::halo_exchange of
        x and ev). The heads run on the owned atoms; their charges, a0 and C6 ratios go to the ghost copies before the pair terms.
        The sums over the whole system (the denominators of the charge and spin embeddings, the charge sum and atom count of the
        charge correction) are dsmd::allreduce_sum of the owned atoms' sums. Pair terms are counted half on atom i, so the energy of
        a pair whose atoms are on two ranks is half on each. Per-atom energies [N], zero for the ghost atoms."""
        silu = torch.nn.functional.silu
        n_atoms = pos.shape[0]
        n_ghost = n_atoms - n_owned
        z = species + 1
        idx_i_lr = edge_index[0]
        idx_j_lr = edge_index[1]
        idx_i, idx_j, vec, d, d_lr = self._edges(pos, edge_shift, idx_i_lr, idx_j_lr)
        ylm, d_basis, cut, e_rep = self._geometry(vec, d, idx_i, idx_j, z, n_atoms)
        radial = self._radial_input(d_basis)

        # embeddings: every rank computes them for its owned and ghost atoms; the denominators are whole-system sums
        x0 = self.atom_embedding[z]
        if self.use_charge_embed or self.use_spin_embed:
            y_charge = self.charge_embed.weights(z)
            y_spin = self.spin_embed.weights(z)
            sums = torch.ops.dsmd.allreduce_sum(torch.stack([y_charge[:n_owned].sum(), y_spin[:n_owned].sum()]))
            if self.use_charge_embed:
                x0 = x0 + self.charge_embed.embed(y_charge, sums[0])
            if self.use_spin_embed:
                x0 = x0 + self.spin_embed.embed(y_spin, sums[1])
        x0 = x0 * self.embed_scale
        x = x0
        ev = torch.zeros((n_atoms, self.ylm_columns.shape[0]), dtype=pos.dtype, device=pos.device)
        f = self.num_features
        k = 0
        for layer in self.layers:
            if k > 0:
                packed = torch.ops.dsmd.halo_exchange(torch.cat([x[:n_owned], ev[:n_owned]], dim=1).contiguous(), n_ghost)
                x = packed[:, :f]
                ev = packed[:, f:]
            x, ev = layer(x, ev, x0, radial, ylm, cut, idx_i, idx_j, self.contraction, self.degree_index, self.ylm_cg)
            k += 1

        # heads on the owned atoms
        x_owned = x[:n_owned]
        z_owned = z[:n_owned]
        hidden = silu(self.energy_regression(x_owned))
        energy_owned = (hidden @ self.energy_final + self.energy_final_bias) * self.atomic_scale[z_owned] + self.atomic_offset[z_owned]
        energy = torch.cat([energy_owned, torch.zeros(n_ghost, dtype=energy_owned.dtype, device=energy_owned.device)])
        if self.use_repulsion:          # onto atom i: owned atoms only
            energy = energy + e_rep

        if self.use_electrostatics or self.use_dispersion:
            if self.use_electrostatics:
                q_raw = self.charge_head(x_owned, z_owned)
                totals = torch.ops.dsmd.allreduce_sum(torch.stack([q_raw.sum(), torch.ones_like(q_raw).sum()]))
                q = q_raw + (float(self.charge) - totals[0]) / totals[1]
            else:
                q = torch.zeros_like(energy_owned)
            if self.use_dispersion:
                alpha = self.free_alpha[z_owned - 1] * torch.abs(self.a0_head(x_owned, z_owned))
                c6 = self.free_c6[z_owned - 1] * torch.abs(self.c6_head(x_owned, z_owned))
            else:
                alpha = torch.ones_like(energy_owned)
                c6 = torch.ones_like(energy_owned)
            per_atom = torch.ops.dsmd.halo_exchange(torch.stack([q, alpha, c6], dim=1).contiguous(), n_ghost)
            energy = energy + self._long_range(pos, edge_shift, idx_i_lr, idx_j_lr, d_lr, per_atom[:, 0].contiguous(),
                                               per_atom[:, 1].contiguous(), per_atom[:, 2].contiguous())
            if self.use_kspace:         # reciprocal parts of this rank's atoms; the grid is summed over the ranks
                e_k = self.kspace_term(pos[:n_owned], q, c6)
                energy = energy + torch.cat([e_k, torch.zeros(n_ghost, dtype=e_k.dtype, device=e_k.device)])
        return energy

    def _geometry(self, vec: Tensor, d: Tensor, idx_i: Tensor, idx_j: Tensor, z: Tensor,
                  n_atoms: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per short-range edge (vectors vec, lengths d): spherical harmonics [E, M], basis distance (NLH distance floor), PhysNet
        cutoff, and the NLH repulsion summed onto atom i [N] (zeros without repulsion). SO3LRAdapterFused: so3lr::edge_geometry."""
        unit = vec / d.unsqueeze(-1)
        ylm = _spherical_harmonics(unit)
        if not self.ylm_identity:
            ylm = ylm.index_select(1, self.ylm_columns)
        zi = torch.minimum(z[idx_i], z[idx_j]) - 1
        zj = torch.maximum(z[idx_i], z[idx_j]) - 1
        d_off = self.floor_distance[zi, zj] + self.floor_width
        d_basis = torch.where(d >= d_off, d, d_off - self.floor_width * torch.tanh((d_off - d) / self.floor_width))
        t = d / self.sr_cutoff
        cut = 1.0 - 6.0 * t ** 5 + 15.0 * t ** 4 - 10.0 * t ** 3     # PhysNet cutoff (every short-range edge has 0 < d < cutoff)
        e_rep = torch.zeros(n_atoms, dtype=vec.dtype, device=vec.device)
        if self.use_repulsion:
            a = self.nlh_a[zi, zj]
            b = self.nlh_b[zi, zj]
            zz = (z[idx_i] * z[idx_j]).to(vec.dtype)
            y = (a * torch.exp(-b * d.unsqueeze(-1))).sum(dim=-1)
            pair = _switch(d, 0.0, self.repulsion_switch) * self.ke * cut * zz / d * y * 0.5
            e_rep = e_rep.index_add_(0, idx_i, pair)
        return ylm, d_basis, cut, e_rep

    def _rbf(self, d_basis: Tensor) -> Tensor:
        """Bernstein radial basis [E, n_rbf] of the basis distances d_basis [E] (so3lr BernsteinBasis with the NLH distance floor)."""
        x_rbf = torch.exp(-self.bernstein_gamma * d_basis).unsqueeze(-1)
        return torch.exp(self.rbf_log_binomial + self.rbf_k * torch.log(x_rbf) + self.rbf_k_rev * torch.log(1.0 - x_rbf))

    def _radial_input(self, d_basis: Tensor) -> Tensor:
        """What the layers get for their radial filters: the radial basis (SO3LRAdapterFused: the basis distance itself)."""
        return self._rbf(d_basis)

    def _edges(self, pos: Tensor, edge_shift: Tensor, idx_i_lr: Tensor, idx_j_lr: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """The short-range graph (edges within the model cutoff) of the long-range edge list: idx_i, idx_j, edge vectors and lengths of
        the short-range edges, and the lengths of all edges (for _long_range)."""
        vec_lr = pos.index_select(0, idx_j_lr) - pos.index_select(0, idx_i_lr) + edge_shift
        d_lr = torch.sqrt((vec_lr * vec_lr).sum(dim=1))
        sr = torch.nonzero(d_lr < self.sr_cutoff).reshape(-1)
        return idx_i_lr.index_select(0, sr), idx_j_lr.index_select(0, sr), vec_lr.index_select(0, sr), d_lr.index_select(0, sr), d_lr

    def _long_range(self, pos: Tensor, edge_shift: Tensor, idx_i_lr: Tensor, idx_j_lr: Tensor, d_lr: Tensor, q: Tensor, alpha: Tensor,
                    c6: Tensor) -> Tensor:
        """Per-atom electrostatics and dispersion on the long-range edges (SO3LRAdapterFused: the operator so3lr::pair_energy)."""
        energy = torch.zeros_like(q)
        if self.use_electrostatics:     # erf-damped, shifted-force, switched Coulomb (so3lr coulomb_erf_shifted_force_smooth)
            r = d_lr
            potential = torch.erf(r / self.coulomb_sigma) / r
            if self.pme_smear > 0.0:        # Ewald split: the smeared part is in the k-space term
                potential = potential - torch.erf(r / self.pme_smear) / r
            shifted = potential - self.coulomb_shift - self.coulomb_force_shift * (r - self.lr_cutoff)
            f_sw = _switch(r, self.coulomb_cuton, self.lr_cutoff)
            pair = 0.5 * self.ke * q.index_select(0, idx_i_lr) * q.index_select(0, idx_j_lr) * (f_sw * (potential - self.coulomb_shift) + (1.0 - f_sw) * shifted)
            pair = torch.where(r < self.lr_cutoff, pair, torch.zeros_like(pair))
            energy = energy + torch.zeros_like(energy).index_add_(0, idx_i_lr, pair)
        if self.use_dispersion:         # QDO-damped C6/C8/C10 (so3lr DispersionEnergySparse, no k-space)
            alpha_i = alpha.index_select(0, idx_i_lr)
            alpha_j = alpha.index_select(0, idx_j_lr)
            c6_i = c6.index_select(0, idx_i_lr)
            c6_j = c6.index_select(0, idx_j_lr)
            alpha_ij = 0.5 * (alpha_i + alpha_j)
            c6_ij = 2.0 * c6_i * c6_j * alpha_j * alpha_i / (alpha_i * alpha_i * c6_j + alpha_j * alpha_j * c6_i)
            radius = self.vdw_radius_factor * torch.pow(alpha_ij, 1.0 / 7.0)
            sigma = -0.00078893 * radius ** 3 + 0.04125273 * radius ** 2 + 0.24428889 * radius - 0.00433008
            gamma = 0.5 / (sigma * sigma)
            c8_ij = 5.0 / gamma * c6_ij
            c10_ij = 245.0 / 8.0 / gamma ** 2 * c6_ij
            p = self.dispersion_scale * 2.0 * 2.54 * torch.pow(alpha_ij, 1.0 / 7.0)
            rb = d_lr / self.bohr
            v = -c6_ij / (rb ** 6 + p ** 6) - c8_ij / (rb ** 8 + p ** 8) - c10_ij / (rb ** 10 + p ** 10)
            if self.disp_smear_bohr > 0.0:  # Ewald split: + geometric-mean C6 times the long-range r^-6 part (in k-space)
                x = rb * rb / (2.0 * self.disp_smear_bohr * self.disp_smear_bohr)
                h = 1.0 - torch.exp(-x) * (1.0 + x + 0.5 * x * x)
                v = v + torch.sqrt(c6_i * c6_j) * h / rb ** 6
            pair = 0.5 * v * self.hartree * _switch(d_lr, self.lr_cutoff - self.dispersion_damping, self.lr_cutoff)
            energy = energy + torch.zeros_like(energy).index_add_(0, idx_i_lr, pair)
        return energy


class SO3LRAdapterFused(SO3LRAdapter):
    """SO3LRAdapter with --accel so3lr_fused: the attention of every layer (_So3kratesLayerFused) and the long-range pair terms run in
    the CUDA operators of mlip_convert/so3lr_ops/; the 12 A edge lengths are not differentiated in PyTorch (the pair operator takes
    the positions and shifts itself). The radial filters of each layer are tabulated in the basis distance (RADIAL_TABLE_NODES nodes,
    cubic Hermite), as in the paper's inference code: no radial basis and no filter MLP per edge. The library must be loaded
    (mlip_convert.accel) before the model is built or scripted."""
    pair_params: List[float]
    geometry_params: List[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_params = [self.lr_cutoff, self.coulomb_sigma, self.coulomb_cuton, self.coulomb_shift, self.coulomb_force_shift, self.ke,
                            self.lr_cutoff - self.dispersion_damping, self.dispersion_scale, self.bohr, self.hartree,
                            self.vdw_radius_factor, 1.0 if self.use_electrostatics else 0.0, 1.0 if self.use_dispersion else 0.0]
        if self.use_kspace:
            self.pair_params = self.pair_params + [self.pme_smear, self.disp_smear_bohr]
        self.geometry_params = [self.sr_cutoff, self.floor_width, self.ke, self.repulsion_switch, 1.0 if self.use_repulsion else 0.0]
        # the basis distance is never below the NLH floor distance of its element pair (d_basis > d_off - width = floor distance)
        lo = 0.9 * float(self.floor_distance.min())
        for layer in self.layers:
            layer.set_radial_table(self._rbf, lo, self.sr_cutoff, RADIAL_TABLE_NODES)

    def _radial_input(self, d_basis: Tensor) -> Tensor:
        return d_basis

    def _geometry(self, vec: Tensor, d: Tensor, idx_i: Tensor, idx_j: Tensor, z: Tensor,
                  n_atoms: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        out = torch.ops.so3lr.edge_geometry(vec, idx_i, idx_j, z, self.floor_distance, self.nlh_a, self.nlh_b, self.ylm_columns, n_atoms,
                                            self.geometry_params)
        return out[0], out[1], out[2], out[3]

    def _edges(self, pos: Tensor, edge_shift: Tensor, idx_i_lr: Tensor, idx_j_lr: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # the short-range edges are picked on detached positions; only their own vectors are differentiated
        pos_d = pos.detach()
        vec_d = pos_d.index_select(0, idx_j_lr) - pos_d.index_select(0, idx_i_lr) + edge_shift.detach()
        sr = torch.nonzero(torch.sqrt((vec_d * vec_d).sum(dim=1)) < self.sr_cutoff).reshape(-1)
        idx_i = idx_i_lr.index_select(0, sr)
        idx_j = idx_j_lr.index_select(0, sr)
        vec = pos.index_select(0, idx_j) - pos.index_select(0, idx_i) + edge_shift.index_select(0, sr)
        d = torch.sqrt((vec * vec).sum(dim=1))
        return idx_i, idx_j, vec, d, torch.zeros(0, dtype=pos.dtype, device=pos.device)

    def _long_range(self, pos: Tensor, edge_shift: Tensor, idx_i_lr: Tensor, idx_j_lr: Tensor, d_lr: Tensor, q: Tensor, alpha: Tensor,
                    c6: Tensor) -> Tensor:
        return torch.ops.so3lr.pair_energy(pos, edge_shift, idx_i_lr, idx_j_lr, q, alpha, c6, self.pair_params)


def _check_supported(m: Dict) -> None:
    if m.get("legacy_so3lr_bool"):
        raise ValueError("SO3LR: the legacy 'so3lr' model (JACS 2025) is not supported, only the SO3LR Tauri models: "
                         + MODEL_NAMES_HELP)
    expected = {"cutoff_fn": "phys", "radial_basis_fn": "bernstein", "activation_fn": "silu", "energy_activation_fn": "silu",
                "qk_non_linearity": "identity", "input_convention": "positions", "residual_mlp_1": True, "residual_mlp_2": False}
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"SO3LR: model setting {key} = {m.get(key)!r} is not supported (the Tauri models use {value!r})")
    if (m.get("layer_normalization_1") or m.get("layer_normalization_2")) and not m.get("use_rms_norm"):
        raise ValueError("SO3LR: LayerNorm with learned scales is not supported (the Tauri models use RMSNorm)")
    if m.get("energy_final_mlp_widths") is not None or m.get("energy_regression_dim") is None:
        raise ValueError("SO3LR: only the single-layer energy readout (energy_regression_dim) is supported")
    if m.get("message_normalization") not in ("avg_num_neighbors", "sqrt_num_features", "identity"):
        raise ValueError(f"SO3LR: message_normalization {m.get('message_normalization')!r} is not supported")


def _levels(obs0: Dict, key: str, default: float) -> np.ndarray:
    """[119, levels] per-element energy scales/offsets; a single-level array is tiled as so3lr does."""
    if key not in obs0:
        return np.full((119, HEAD_INDEX + 1), default)
    array = np.asarray(obs0[key])
    if array.ndim == 1:
        array = np.tile(array[:, None], (1, HEAD_INDEX + 1))
    return array


def _dummy_charge_spin(num_features: int) -> Dict:
    """Zero weights for a model without that embedding (the module is never called then; it only keeps TorchScript's types fixed)."""
    zeros = np.zeros((num_features, num_features))
    return {"Embed_0": {"embedding": np.zeros((119, num_features))}, "Embed_1": {"embedding": np.zeros((2, num_features))},
            "Embed_2": {"embedding": np.zeros((2, num_features))},
            "Residual_0": {"layers_0": {"kernel": zeros}, "layers_1": {"kernel": zeros}}}
