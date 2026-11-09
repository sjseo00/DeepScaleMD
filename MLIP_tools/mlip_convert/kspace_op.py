"""The k-space (reciprocal-space) operator `dsmd::kspace` for models whose long-range terms are split Ewald-style (so3lr with
--kspace pme): smooth particle-mesh Ewald of per-atom "charges" that the model computes at every step.

    dsmd::kspace(Tensor pos, Tensor charges, float[] params) -> Tensor energy
        pos      [n, 3]  positions (A) of this rank's owned atoms (any periodic image)
        charges  [n, 2]  column 0: electrostatic charges q (e); column 1: dispersion charges c = sqrt(C6) (atomic units)
        params   [ke, smearing_e, smearing_d, bohr, hartree, use_elec, use_disp]  (A, eV)
        energy   [n]     per-atom reciprocal energy (eV), self and background corrections included

The box, the grid spacings, the B-spline order and how the grid is summed over ranks are not model parameters: they come from
the *context* the caller sets around the model call (`use_context`): in dsMD the C++ version of this operator
(src/MLIPKspaceOp.cpp) with dsMD's box and the YAML keys mlip_kspace_*; while a model is built, checked and tested, this file.

The formulas are those of jaxpme (the k-space library so3lr uses), with jaxpme's Lagrange interpolation replaced by smooth PME
(cardinal B-splines with the |b(m)|^2 influence function), which converges to the same Ewald sums:
    electrostatics (lengths in A):  G(k) = 4 pi exp(-s^2 k^2 / 2) / k^2,  self sqrt(2/pi)/s,  background pi s^2;  e_i = ke/2 q_i phi_i
    dispersion (lengths in bohr):   G(k) = pi^(3/2)/2 (2 s^2)^(-3/2) Gamma(-3/2, x) x^(3/2), x = s^2 k^2 / 2,
                                    self 1/(6 (2 s^2)^3), background pi^(3/2) (2 s^2)^(-3/2) / (-6);  e_i = -hartree/2 c_i phi_i
    phi_i = (1/V) sum_k G(k) S(k) e^(-ik.r_i) - c_i self - 2 background sum_j c_j / V
The derivatives are computed in the forward pass and returned by the backward (dE/dq_i, dE/dc_i, dE/dr_i), so the backward does
no communication. The energy enters the model's total as the sum over the owned atoms (every row weighted 1).

Virial: the wrapper scales positions by (1 + strain) and differentiates, which gives sum_i x_i dE/dx_i of the k-space energy but
not its explicit dependence on the box. The operator therefore also computes the correction  W_true + sum_i x_i dE/dx_i  per axis
(W_true = -dE/dstrain at fixed fractional coordinates, from the k-space sum) and leaves it in the context; dsMD adds it to the
model's virial. Over several ranks every rank adds its own atoms' part and 1/n_ranks of the grid part.
"""
import math
import threading
from contextlib import contextmanager
from typing import Callable, List, Optional

import torch
from torch import Tensor

PARAMS = ("ke", "smearing_e", "smearing_d", "bohr", "hartree", "use_elec", "use_disp")


def fft_friendly(n: int) -> bool:
    for f in (2, 3, 5, 7):
        while n % f == 0:
            n //= f
    return n == 1


def grid_size(length: float, spacing: float, order: int) -> int:
    """Points of the grid along one axis: at most `spacing` apart, at least 2 x order, a product of 2, 3, 5 and 7 only.
    The C++ operator uses the same rule."""
    n = max(int(math.ceil(length / spacing - 1e-9)), 2 * order)
    while not fft_friendly(n):
        n += 1
    return n


class KspaceContext:
    """box [3] (A), grid spacings (A) for the electrostatic and the dispersion grids, the B-spline order, and how to sum a grid
    over the ranks: grid_sum(grid) -> summed grid (None: one rank holds all atoms), grid_ranks: ranks that share the grid (the
    grid part of the virial correction is divided by it). After a call: virial_correction [3] (eV), added up over the calls."""

    def __init__(self, box, spacing_e: float, spacing_d: float, order: int = 4,
                 grid_sum: Optional[Callable[[Tensor], Tensor]] = None, grid_ranks: int = 1):
        self.box = [float(v) for v in box]
        self.spacing_e = float(spacing_e)
        self.spacing_d = float(spacing_d)
        self.order = int(order)
        self.grid_sum = grid_sum
        self.grid_ranks = int(grid_ranks)
        self.virial_correction = [0.0, 0.0, 0.0]
        self.calls = 0


_local = threading.local()


def context_from(settings: dict, box) -> KspaceContext:
    """The context for a model converted with --kspace (metadata.json "kspace": its default grid settings) in a box (A)."""
    return KspaceContext(box, settings["grid_spacing"], settings["dispersion_grid_spacing"], settings["order"])


@contextmanager
def maybe_context(settings: Optional[dict], box):
    """use_context(context_from(settings, box)), or nothing for a model without k-space (settings None)."""
    if not settings:
        yield None
        return
    with use_context(context_from(settings, box)) as context:
        yield context


def get_context() -> Optional[KspaceContext]:
    return getattr(_local, "context", None)


@contextmanager
def use_context(context: KspaceContext):
    previous = getattr(_local, "context", None)
    _local.context = context
    try:
        yield context
    finally:
        _local.context = previous


# ---------------------------------------------------------------------------------------------------------------- B-splines
def bspline(y: Tensor, order: int) -> Tensor:
    """Cardinal B-spline M_order(y) (support [0, order)), closed form sum_j (-1)^j C(order, j) max(y - j, 0)^(order-1) / (order-1)!"""
    out = torch.zeros_like(y)
    for j in range(order + 1):
        out = out + ((-1) ** j) * math.comb(order, j) * torch.clamp(y - j, min=0.0) ** (order - 1)
    out = out / math.factorial(order - 1)
    return torch.where((y >= 0) & (y < order), out, torch.zeros_like(out))


def bspline_moduli(n: int, order: int, dtype, device) -> Tensor:
    """|sum_{k=0}^{order-2} M_order(k+1) exp(2 pi i m k / n)|^2 for m = 0 .. n-1 (the smooth-PME influence function is its inverse)."""
    m = torch.arange(n, dtype=torch.float64)
    k = torch.arange(order - 1, dtype=torch.float64)
    w = bspline(k + 1.0, order)
    angle = 2.0 * math.pi * m[:, None] * k[None, :] / n
    re = (w * torch.cos(angle)).sum(dim=1)
    im = (w * torch.sin(angle)).sum(dim=1)
    return (re * re + im * im).to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------------------------------------------- kernels
def coulomb_kernel(k2: Tensor, s: float):
    """G(k) and (1/G) dG/dk / k (so that k_a^2/k G'/G = k_a^2 times it), k2 > 0"""
    g = 4.0 * math.pi * torch.exp(-0.5 * s * s * k2) / k2
    ratio = -2.0 / k2 - s * s
    return g, ratio


def dispersion_kernel(k2: Tensor, s: float):
    x = 0.5 * s * s * k2
    sx = torch.sqrt(x)
    ex = torch.exp(-x)
    g12 = math.sqrt(math.pi) * torch.erfc(sx)
    g_m12 = -2.0 * (g12 - ex / sx)
    g_m32 = -2.0 / 3.0 * (g_m12 - ex / (x * sx))           # Gamma(-3/2, x)
    factor = math.pi ** 1.5 / 2.0 * (2.0 * s * s) ** (-1.5)
    g = factor * g_m32 * x * sx
    # k_a^2/k G'/G = k_a^2 s^2 (1.5/x - exp(-x) / (Gamma(-3/2, x) x^(5/2)))
    ratio = s * s * (1.5 / x - ex / (g_m32 * x * x * sx))
    return g, ratio


def coulomb_corrections(s: float):
    return math.sqrt(2.0 / math.pi) / s, math.pi * s * s                       # self, background


def dispersion_corrections(s: float):
    t = 2.0 * s * s
    return 1.0 / (6.0 * t ** 3), math.pi ** 1.5 * t ** (-1.5) / (-6.0)


# ---------------------------------------------------------------------------------------------------------------- one grid
def pme(pos: Tensor, c: Tensor, box: List[float], spacing: float, order: int, smearing: float, kind: str, unit: float,
        grid_sum: Optional[Callable[[Tensor], Tensor]]):
    """Smooth PME of charges c [n] at pos [n, 3] (A) in the box (A), lengths scaled by `unit` (1 for A, 1/bohr for bohr).
    Returns phi [n] (potential in the scaled units, corrections included), dphi_dx [n, 3] (per A), the mesh and background energy
    E_grid = 1/2 sum over all ranks' atoms of c phi_mesh + background (without self) and its virial W [3] (both in the units of
    phi times charge^2, without the prefactor), all for the grid of the whole system."""
    dtype, device = pos.dtype, pos.device
    n_atoms = pos.shape[0]
    ng = [grid_size(box[a], spacing, order) for a in range(3)]
    length = [box[a] * unit for a in range(3)]
    volume = length[0] * length[1] * length[2]
    u = torch.stack([pos[:, a] / box[a] * ng[a] for a in range(3)], dim=1)
    u = u - torch.floor(u / torch.tensor(ng, dtype=dtype, device=device)) * torch.tensor(ng, dtype=dtype, device=device)
    base = torch.floor(u)
    f = u - base
    base = base.to(torch.int64)
    k = torch.arange(order, dtype=dtype, device=device)
    y = f[:, :, None] + k                                         # [n, 3, order]
    w = bspline(y, order)
    dw = bspline(y, order - 1) - bspline(y - 1.0, order - 1)
    kk = torch.arange(order, dtype=torch.int64, device=device)
    ngt = torch.tensor(ng, dtype=torch.int64, device=device)
    idx = torch.remainder(base[:, :, None] - kk, ngt[None, :, None])   # [n, 3, order]
    flat = (idx[:, 0, :, None, None] * ng[1] + idx[:, 1, None, :, None]) * ng[2] + idx[:, 2, None, None, :]
    wx, wy, wz = w[:, 0, :, None, None], w[:, 1, None, :, None], w[:, 2, None, None, :]
    w3 = wx * wy * wz
    total = ng[0] * ng[1] * ng[2]
    grid = torch.zeros(total, dtype=dtype, device=device).index_add_(0, flat.reshape(-1), (c[:, None, None, None] * w3).reshape(-1))
    grid = grid.reshape(ng[0], ng[1], ng[2])
    if grid_sum is not None:
        grid = grid_sum(grid)
    charge_sum = grid.sum()

    s_hat = torch.fft.rfftn(grid)
    mx = torch.fft.fftfreq(ng[0], device=device, dtype=dtype) * ng[0]
    my = torch.fft.fftfreq(ng[1], device=device, dtype=dtype) * ng[1]
    mz = torch.fft.rfftfreq(ng[2], device=device, dtype=dtype) * ng[2]
    kx = (2.0 * math.pi / length[0]) * mx
    ky = (2.0 * math.pi / length[1]) * my
    kz = (2.0 * math.pi / length[2]) * mz
    kx2, ky2, kz2 = kx[:, None, None] ** 2, ky[None, :, None] ** 2, kz[None, None, :] ** 2
    k2 = kx2 + ky2 + kz2
    zero = k2 == 0
    k2s = torch.where(zero, torch.ones_like(k2), k2)
    if kind == "coulomb":
        g, ratio = coulomb_kernel(k2s, smearing * unit)
        self_c, background = coulomb_corrections(smearing * unit)
    else:
        g, ratio = dispersion_kernel(k2s, smearing * unit)
        self_c, background = dispersion_corrections(smearing * unit)
    g = torch.where(zero, torch.zeros_like(g), g)
    ratio = torch.where(zero, torch.zeros_like(ratio), ratio)
    b = 1.0 / (bspline_moduli(ng[0], order, dtype, device)[:, None, None] * bspline_moduli(ng[1], order, dtype, device)[None, :, None]
               * bspline_moduli(ng[2], order, dtype, device)[: ng[2] // 2 + 1][None, None, :])
    gb = g * b
    weight = torch.full((ng[2] // 2 + 1,), 2.0, dtype=dtype, device=device)
    weight[0] = 1.0
    if ng[2] % 2 == 0:
        weight[-1] = 1.0
    e_k = 0.5 / volume * gb * (s_hat.real ** 2 + s_hat.imag ** 2) * weight[None, None, :]
    e_mesh = e_k.sum()
    e_background = -background * charge_sum * charge_sum / volume
    virial = torch.stack([(e_k * (1.0 + kx2 * ratio)).sum(), (e_k * (1.0 + ky2 * ratio)).sum(), (e_k * (1.0 + kz2 * ratio)).sum()])
    virial = virial + e_background

    psi = torch.fft.irfftn(gb * s_hat, s=ng) * (total / volume)
    vals = psi.reshape(-1)[flat]
    phi = (vals * w3).sum(dim=(1, 2, 3))
    dux = (vals * dw[:, 0, :, None, None] * wy * wz).sum(dim=(1, 2, 3))
    duy = (vals * wx * dw[:, 1, None, :, None] * wz).sum(dim=(1, 2, 3))
    duz = (vals * wx * wy * dw[:, 2, None, None, :]).sum(dim=(1, 2, 3))
    dphi = torch.stack([dux * (ng[0] / box[0]), duy * (ng[1] / box[1]), duz * (ng[2] / box[2])], dim=1)   # per A
    phi = phi - c * self_c - 2.0 * background * charge_sum / volume
    del n_atoms
    return phi, dphi, e_mesh + e_background, virial


def kspace_terms(pos: Tensor, charges: Tensor, params: List[float], context: KspaceContext):
    """Per-atom energies [n] (eV), dE/dpos [n, 3] (eV/A), dE/dcharges [n, 2], and this rank's virial correction [3] (eV)."""
    ke, s_e, s_d, bohr, hartree, use_elec, use_disp = params
    energy = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
    d_pos = torch.zeros_like(pos)
    d_charges = torch.zeros_like(charges)
    correction = torch.zeros(3, dtype=torch.float64)
    terms = []
    if use_elec:
        terms.append((0, "coulomb", s_e, 1.0, context.spacing_e, ke))
    if use_disp:
        terms.append((1, "dispersion", s_d, 1.0 / bohr, context.spacing_d, -hartree))
    for column, kind, smearing, unit, spacing, prefactor in terms:
        c = charges[:, column]
        phi, dphi, _e_grid, w_grid = pme(pos, c, context.box, spacing, context.order, smearing, kind, unit, context.grid_sum)
        energy = energy + 0.5 * prefactor * c * phi
        d_charges[:, column] = prefactor * phi
        de_dx = prefactor * c[:, None] * dphi
        d_pos = d_pos + de_dx
        # correction = W_true (grid part shared by the ranks) + sum_i x_i dE/dx_i (this rank's atoms)
        correction = correction + (prefactor * w_grid).double().cpu() / context.grid_ranks + (pos * de_dx).sum(dim=0).double().cpu()
    return energy, d_pos, d_charges, correction


class _Kspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos: Tensor, charges: Tensor, params: List[float]) -> Tensor:
        context = get_context()
        if context is None:
            raise RuntimeError("dsmd::kspace: no k-space context (box, grid) is set: run the model inside dsMD, or inside "
                               "mlip_convert.kspace_op.use_context")
        energy, d_pos, d_charges, correction = kspace_terms(pos.detach(), charges.detach(), [float(v) for v in params], context)
        for a in range(3):
            context.virial_correction[a] += float(correction[a])
        context.calls += 1
        ctx.save_for_backward(d_pos, d_charges)
        return energy

    @staticmethod
    def backward(ctx, grad: Tensor):
        d_pos, d_charges = ctx.saved_tensors
        # the energy enters the total with weight 1 on every row; one weight per call is all the backward supports
        g = grad[:1].reshape(1, 1) if grad.numel() else grad.new_zeros(1, 1)
        if grad.numel() and bool((grad - grad[0]).abs().max() > 1e-6 * max(1.0, float(grad[0].abs()))):
            raise RuntimeError("dsmd::kspace: the per-atom energies must enter the total with one common weight")
        return g * d_pos, g * d_charges, None


def _registered_by_the_md_program() -> bool:
    try:
        torch.ops.dsmd.kspace
    except (AttributeError, RuntimeError):
        return False
    return True


def _backend_impl(pos: Tensor, charges: Tensor, params: List[float]) -> Tensor:
    return _Kspace.apply(pos, charges, params)


def _autograd_impl(pos: Tensor, charges: Tensor, params: List[float]) -> Tensor:
    return _Kspace.apply(pos, charges, params)


if not _registered_by_the_md_program():
    _library = torch.library.Library("dsmd", "FRAGMENT")
    _library.define("kspace(Tensor pos, Tensor charges, float[] params) -> Tensor")
    _library.impl("kspace", _backend_impl, "CompositeExplicitAutograd")
    _library.impl("kspace", _autograd_impl, "Autograd")


def kspace(pos: Tensor, charges: Tensor, params: List[float]) -> Tensor:
    return torch.ops.dsmd.kspace(pos, charges, params)


# ------------------------------------------------------------------------------------------------------- reference (tests)
def ewald_reference(pos: Tensor, c: Tensor, box: List[float], smearing: float, kind: str, unit: float, k_max: int = 12):
    """Plain Ewald k-space sum (all k with |m_a| <= k_max) of the same kernel, float64: phi [n] and the energy 1/2 sum c phi
    (corrections included), for checking pme()."""
    pos = pos.double()
    c = c.double()
    length = [b * unit for b in box]
    volume = length[0] * length[1] * length[2]
    r = torch.arange(-k_max, k_max + 1, dtype=torch.float64)
    m = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    m = m[(m != 0).any(dim=1)]
    kv = 2.0 * math.pi * m / torch.tensor(length, dtype=torch.float64)
    k2 = (kv * kv).sum(dim=1)
    if kind == "coulomb":
        g, _ = coulomb_kernel(k2, smearing * unit)
        self_c, background = coulomb_corrections(smearing * unit)
    else:
        g, _ = dispersion_kernel(k2, smearing * unit)
        self_c, background = dispersion_corrections(smearing * unit)
    phase = (pos * unit) @ kv.T                                    # [n, K]
    cs = (c[:, None] * torch.cos(phase)).sum(dim=0)
    sn = (c[:, None] * torch.sin(phase)).sum(dim=0)
    phi = (g[None, :] * (torch.cos(phase) * cs[None, :] + torch.sin(phase) * sn[None, :])).sum(dim=1) / volume
    phi = phi - c * self_c - 2.0 * background * c.sum() / volume
    return phi, 0.5 * (c * phi).sum()
