"""Boundary reconstruction and horizon tracking from NOFE 3D seismic reductions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter, maximum_filter1d, median_filter
from sklearn.metrics import average_precision_score, roc_auc_score

_EPS = 1e-8
_MIN_DENOM = 1e-6
_MIN_BINARY_CLASSES = 2
_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4
_TOL_3PX = 3.0
_TOL_5PX = 5.0


@dataclass(frozen=True)
class HorizonConfig:
    """Configuration for NOFE-based boundary reconstruction and horizon tracking."""

    out_dir: Path = Path("outputs/f3_horizon_tracking")
    device: str = "cpu"
    seed: int = 42
    train_inline_frac: float = 0.75
    train_stride: tuple[int, int, int] = (11, 20, 6)
    slice_lateral_stride: int = 4
    radius_train: float = 0.15
    radius_slice: float = 0.12
    width: int = 16
    ker_width: int = 48
    depth: int = 2
    out_dim: int = 8
    nystrom: float | None = 0.10
    chunk_size: int = 8192
    epochs: int = 80
    patience: int = 20
    lr: float = 1e-2
    contrast_windows: tuple[int, ...] = (4, 8, 14)
    rgt_w_dip: float = 10.0
    rgt_w_mono: float = 1.0
    max_slope: float = 10.0
    noise_sigma: float = 0.50


@dataclass(frozen=True)
class DipCoherencyField:
    """2D structural slope ``dz/du`` and anisotropy coherency ``[U, Z]``."""

    slope: np.ndarray
    coherency: np.ndarray


def _z_axis(nz: int, z_coords: np.ndarray | None) -> np.ndarray:
    if z_coords is None:
        return np.arange(nz, dtype=np.float32)
    return np.asarray(z_coords, dtype=np.float32)


def extract_true_interfaces(labels: np.ndarray) -> np.ndarray:
    """Return boolean vertical interface mask ``[..., Z - 1]``."""
    lbl = np.asarray(labels)
    if lbl.shape[-1] < _NDIM_2D:
        raise ValueError(f"Need Z >= 2 for interfaces, got shape {lbl.shape}")
    return lbl[..., :-1] != lbl[..., 1:]


def extract_transition_depths(
    labels: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    z_coords: np.ndarray | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Return first vertical transition depth per trace (``NaN`` if absent)."""
    lbl = np.asarray(labels)
    z_axis = _z_axis(lbl.shape[-1], z_coords)
    iface_z = 0.5 * (z_axis[:-1] + z_axis[1:])

    out: dict[tuple[int, int], np.ndarray] = {}
    for ca, cb in pairs:
        trans = (lbl[..., :-1] == ca) & (lbl[..., 1:] == cb)
        present = trans.any(axis=-1)
        idx = np.argmax(trans, axis=-1)
        depths = np.full(lbl.shape[:-1], np.nan, dtype=np.float32)
        depths[present] = iface_z[idx[present]]
        out[(int(ca), int(cb))] = depths
    return out


def extract_unit_top_depths(
    labels: np.ndarray,
    unit_ids: Sequence[int],
    z_coords: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Return top-of-unit boundary depth where ``labels >= unit_id`` first occurs."""
    lbl = np.asarray(labels)
    z_axis = _z_axis(lbl.shape[-1], z_coords)
    iface_z = 0.5 * (z_axis[:-1] + z_axis[1:])

    out: dict[int, np.ndarray] = {}
    for uid in unit_ids:
        ge_mask = lbl >= uid
        present = ge_mask.any(axis=-1)
        first_idx = np.argmax(ge_mask, axis=-1)
        depths = np.full(lbl.shape[:-1], np.nan, dtype=np.float32)
        interior = present & (first_idx > 0)
        at_top = present & (first_idx == 0)
        depths[interior] = iface_z[first_idx[interior] - 1]
        depths[at_top] = z_axis[0]
        out[int(uid)] = depths
    return out


def multiscale_boundary_contrast(
    emb_grid: np.ndarray,
    *,
    windows: tuple[int, ...] = (4, 8, 14),
    lateral_sigma: float = 0.8,
) -> np.ndarray:
    """Compute unsupervised multi-scale two-sided vertical embedding contrast."""
    arr = np.asarray(emb_grid, dtype=np.float32)
    if arr.ndim not in (_NDIM_3D, _NDIM_4D):
        raise ValueError(f"Expected 3D [U, Z, D] or 4D [I, X, Z, D], got {arr.shape}")
    nz = arr.shape[-2]
    if nz < _NDIM_2D:
        raise ValueError(f"Need Z >= 2, got Z={nz}")

    sig = (
        (lateral_sigma, 0.0, 0.0)
        if arr.ndim == _NDIM_3D
        else (lateral_sigma, lateral_sigma, 0.0, 0.0)
    )
    smoothed = gaussian_filter(arr, sigma=sig) if lateral_sigma > 0 else arr

    total = np.zeros((*arr.shape[:-2], nz - 1), dtype=np.float32)
    for w in windows:
        win = max(1, int(w))
        pad_width = ((0, 0),) * (arr.ndim - 2) + ((win - 1, win), (0, 0))
        padded = np.pad(smoothed, pad_width, mode="edge")
        above = np.zeros((*arr.shape[:-2], nz - 1, arr.shape[-1]), dtype=np.float32)
        below = np.zeros_like(above)
        for k in range(win):
            sl_a = (slice(None),) * (arr.ndim - 2) + (
                slice((win - 1) - k, (win - 1) - k + (nz - 1)),
                slice(None),
            )
            sl_b = (slice(None),) * (arr.ndim - 2) + (
                slice(win + k, win + k + (nz - 1)),
                slice(None),
            )
            above += padded[sl_a]
            below += padded[sl_b]
        diff = (below - above) / float(win)
        c_w = np.linalg.norm(diff, axis=-1).astype(np.float32)
        scale = float(np.std(c_w))
        total += c_w / (scale if scale > _EPS else 1.0)
    return (total / float(len(windows))).astype(np.float32)


def spatial_gradient_magnitude(
    emb_grid: np.ndarray, *, smooth_sigma: float = 0.6
) -> np.ndarray:
    """Compute Euclidean spatial gradient norm ``||∇Z||_2`` over 2D or 3D grids."""
    arr = np.asarray(emb_grid, dtype=np.float32)
    if arr.ndim not in (_NDIM_3D, _NDIM_4D):
        raise ValueError(f"Expected 3D or 4D grid, got {arr.shape}")
    spatial_axes = arr.ndim - 1
    sig = (smooth_sigma,) * spatial_axes + (0.0,)
    sm = gaussian_filter(arr, sigma=sig) if smooth_sigma > 0 else arr
    sq_sum = np.zeros(arr.shape[:-1], dtype=np.float32)
    for ax in range(spatial_axes):
        g_ax = np.gradient(sm, axis=ax)
        sq_sum += np.sum(g_ax * g_ax, axis=-1)
    return np.sqrt(np.maximum(sq_sum, 0.0)).astype(np.float32)


def extract_boundary_ridges(
    contrast: np.ndarray,
    *,
    nms_size: int = 11,
    top_k: int = 3,
    min_rel_height: float = 0.25,
) -> np.ndarray:
    """Extract 1-voxel-thick vertical local maxima ridges from boundary contrast."""
    c = np.asarray(contrast, dtype=np.float32)
    if c.ndim != _NDIM_2D:
        raise ValueError(f"Expected 2D contrast [U, Z-1], got {c.shape}")
    is_peak = c == maximum_filter1d(c, size=max(_NDIM_3D, int(nms_size)), axis=1)
    thresh = float(np.percentile(c, 95.0)) * float(min_rel_height)
    ridges = np.zeros_like(is_peak, dtype=bool)
    for u in range(c.shape[0]):
        pks = np.flatnonzero(is_peak[u] & (c[u] >= thresh))
        if pks.size == 0:
            continue
        order = pks[np.argsort(c[u, pks])[::-1][:top_k]]
        ridges[u, order] = True
    return ridges


def estimate_structure_tensor_2d(
    emb_grid: np.ndarray,
    *,
    grad_sigma: tuple[float, float] = (1.0, 1.5),
    tensor_sigma: tuple[float, float] = (3.0, 5.0),
    max_slope: float = 10.0,
) -> DipCoherencyField:
    """Compute multi-channel 2D structure-tensor slope ``dz/du`` and coherency."""
    arr = np.asarray(emb_grid, dtype=np.float32)
    if arr.ndim != _NDIM_3D:
        raise ValueError(f"Expected [U, Z, D] embedding grid, got {arr.shape}")
    sm = gaussian_filter(arr, sigma=(grad_sigma[0], grad_sigma[1], 0.0))
    gu = np.gradient(sm, axis=0)
    gz = np.gradient(sm, axis=1)
    juu = gaussian_filter(np.sum(gu * gu, axis=2), sigma=tensor_sigma)
    jzz = gaussian_filter(np.sum(gz * gz, axis=2), sigma=tensor_sigma)
    juz = gaussian_filter(np.sum(gu * gz, axis=2), sigma=tensor_sigma)

    slope = np.clip(-juz / (jzz + _MIN_DENOM), -max_slope, max_slope).astype(np.float32)
    trace = juu + jzz
    diff = juu - jzz
    rad = np.sqrt(np.maximum(diff * diff + 4.0 * juz * juz, 0.0))
    lam1 = 0.5 * (trace + rad)
    lam2 = 0.5 * (trace - rad)
    coh = ((lam1 - lam2) / (lam1 + lam2 + _EPS)).astype(np.float32)
    return DipCoherencyField(slope=slope, coherency=coh)


def solve_rgt_2d(
    field: DipCoherencyField,
    *,
    w_dip: float = 10.0,
    w_mono: float = 1.0,
    anchor: tuple[int, int] | None = None,
) -> np.ndarray:
    """Solve sparse least-squares 2D Relative Geological Time field ``[U, Z]``."""
    slope = field.slope
    coh = field.coherency
    nu, nz = slope.shape
    n_vars = nu * nz
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    eq_offset = 0

    def _flat(u_idx: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
        return u_idx * nz + z_idx

    if nu > 1 and nz > _NDIM_2D:
        uu, zz = np.meshgrid(np.arange(nu - 1), np.arange(1, nz - 1), indexing="ij")
        uu_f, zz_f = uu.ravel(), zz.ravel()
        w = np.sqrt(w_dip * np.clip(coh[uu_f, zz_f], 0.1, 1.0))
        s = slope[uu_f, zz_f]
        n_eq = uu_f.size
        eqs = np.arange(eq_offset, eq_offset + n_eq)
        rows.extend([eqs, eqs, eqs, eqs])
        cols.extend(
            [
                _flat(uu_f + 1, zz_f),
                _flat(uu_f, zz_f),
                _flat(uu_f, zz_f + 1),
                _flat(uu_f, zz_f - 1),
            ]
        )
        vals.extend([w, -w, 0.5 * s * w, -0.5 * s * w])
        rhs.append(np.zeros(n_eq, dtype=np.float64))
        eq_offset += n_eq

    uu_m, zz_m = np.meshgrid(np.arange(nu), np.arange(nz - 1), indexing="ij")
    uu_mf, zz_mf = uu_m.ravel(), zz_m.ravel()
    n_m = uu_mf.size
    eqs_m = np.arange(eq_offset, eq_offset + n_m)
    wm = np.full(n_m, np.sqrt(w_mono), dtype=np.float64)
    rows.extend([eqs_m, eqs_m])
    cols.extend([_flat(uu_mf, zz_mf + 1), _flat(uu_mf, zz_mf)])
    vals.extend([wm, -wm])
    rhs.append(wm)
    eq_offset += n_m

    au, az = (nu // 2, nz // 2) if anchor is None else anchor
    rows.append(np.array([eq_offset]))
    cols.append(np.array([au * nz + az]))
    vals.append(np.array([10.0]))
    rhs.append(np.array([10.0 * float(az)]))
    eq_offset += 1

    mat = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(eq_offset, n_vars),
    ).tocsc()
    b_vec = np.concatenate(rhs)
    sol = spla.spsolve(mat.T @ mat, mat.T @ b_vec)
    return sol.reshape(nu, nz).astype(np.float32)


def _rgt_contour_at_value(
    rgt: np.ndarray,
    target_t: float,
    z_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Extract sub-sample horizon depth curve ``[U]`` for ``T(u, z) = target_t``."""
    nu, nz = rgt.shape
    z_axis = _z_axis(nz, z_coords)
    hz = np.zeros(nu, dtype=np.float32)
    for u in range(nu):
        col = rgt[u]
        j = int(np.argmin(np.abs(col - target_t)))
        if j > 0 and (col[j - 1] - target_t) * (col[j] - target_t) <= 0:
            denom = float(col[j] - col[j - 1])
            if abs(denom) > _MIN_DENOM:
                frac = float((target_t - col[j - 1]) / denom)
                hz[u] = z_axis[j - 1] + frac * (z_axis[j] - z_axis[j - 1])
                continue
        if j < nz - 1 and (col[j] - target_t) * (col[j + 1] - target_t) <= 0:
            denom = float(col[j + 1] - col[j])
            if abs(denom) > _MIN_DENOM:
                frac = float((target_t - col[j]) / denom)
                hz[u] = z_axis[j] + frac * (z_axis[j + 1] - z_axis[j])
                continue
        hz[u] = z_axis[j]
    return hz


def extract_horizon_iso_rgt(
    rgt: np.ndarray,
    seed_u: int,
    seed_z: float,
    z_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Track horizon depth curve ``[U]`` from ``(seed_u, seed_z)`` via RGT contour."""
    nz = rgt.shape[1]
    target_t = float(
        np.interp(float(seed_z), np.arange(nz, dtype=np.float32), rgt[seed_u])
    )
    return _rgt_contour_at_value(rgt, target_t, z_coords=z_coords)


def discover_horizons_from_rgt(
    rgt: np.ndarray,
    contrast: np.ndarray,
    coherency: np.ndarray,
    *,
    num_horizons: int = 4,
    min_sep: int = 14,
    z_coords: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Discover major boundary horizons without seeds via RGT-flattened contrast."""
    nu, nz = rgt.shape
    t_iface = 0.5 * (rgt[:, :-1] + rgt[:, 1:])
    w_iface = np.clip(0.5 * (coherency[:, :-1] + coherency[:, 1:]), 0.05, 1.0)

    t_lo = float(np.percentile(t_iface, 2.0))
    t_hi = float(np.percentile(t_iface, 98.0))
    tau_grid = np.linspace(t_lo, t_hi, max(nz - 1, 8), dtype=np.float32)

    weighted_sum = np.zeros_like(tau_grid, dtype=np.float64)
    weight_total = np.zeros_like(tau_grid, dtype=np.float64)
    for u in range(nu):
        t_col = np.maximum.accumulate(t_iface[u])
        in_range = (tau_grid >= t_col[0]) & (tau_grid <= t_col[-1])
        c_interp = np.interp(tau_grid, t_col, contrast[u])
        w_interp = np.interp(tau_grid, t_col, w_iface[u]) * in_range
        weighted_sum += c_interp * w_interp
        weight_total += w_interp

    spectrum = np.where(
        weight_total > _MIN_DENOM,
        weighted_sum / np.maximum(weight_total, _MIN_DENOM),
        0.0,
    )
    is_pk = spectrum == maximum_filter1d(spectrum, size=max(_NDIM_3D, int(min_sep)))
    idx_grid = np.arange(tau_grid.size)
    valid_pk = np.flatnonzero(
        is_pk & (idx_grid >= _NDIM_3D) & (idx_grid <= tau_grid.size - 4)
    )
    if valid_pk.size == 0:
        valid_pk = np.argsort(spectrum)[::-1][:num_horizons]

    chosen: list[int] = []
    for idx in valid_pk[np.argsort(spectrum[valid_pk])[::-1]]:
        if all(abs(int(idx) - c_idx) >= min_sep for c_idx in chosen):
            chosen.append(int(idx))
        if len(chosen) >= num_horizons:
            break
    chosen.sort()

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(chosen):
        tau_val = float(tau_grid[idx])
        depths = _rgt_contour_at_value(rgt, tau_val, z_coords=z_coords)
        results.append(
            {
                "rank": rank,
                "tau": round(tau_val, 3),
                "contrast_score": round(float(spectrum[idx]), 4),
                "depths": depths,
            }
        )
    return results


def rgt_aligned_smooth(
    field: np.ndarray,
    rgt: np.ndarray,
    *,
    sigma_u: float = 1.5,
    sigma_z: float = 1.2,
) -> np.ndarray:
    """Smooth a 2D field ``[U, Z, C]`` along RGT-flattened stratigraphy."""
    arr = np.asarray(field, dtype=np.float32)
    nu, nz, nch = arr.shape
    t_lo, t_hi = float(rgt.min()), float(rgt.max())
    tau_grid = np.linspace(t_lo, t_hi, nz, dtype=np.float32)

    flat = np.zeros((nu, nz, nch), dtype=np.float32)
    mask = np.zeros((nu, nz), dtype=np.float32)
    for u in range(nu):
        t_col = np.maximum.accumulate(rgt[u])
        in_rng = ((tau_grid >= t_col[0]) & (tau_grid <= t_col[-1])).astype(np.float32)
        mask[u] = in_rng
        for c in range(nch):
            flat[u, :, c] = np.interp(tau_grid, t_col, arr[u, :, c]) * in_rng

    num = gaussian_filter(flat, sigma=(sigma_u, sigma_z, 0.0))
    den = np.maximum(gaussian_filter(mask, sigma=(sigma_u, sigma_z)), _MIN_DENOM)
    flat_sm = num / den[:, :, None]

    out = np.zeros_like(arr)
    for u in range(nu):
        t_col = np.maximum.accumulate(rgt[u])
        for c in range(nch):
            out[u, :, c] = np.interp(t_col, tau_grid, flat_sm[u, :, c])
    return out


def _dp_trace_splits(logp_trace: np.ndarray) -> np.ndarray:
    """Run exact ``O(C * Z)`` monotonic split DP on a single trace ``[Z, C]``."""
    nz, ncls = logp_trace.shape
    s_pref = np.zeros((nz + 1, ncls), dtype=np.float64)
    s_pref[1:, :] = np.cumsum(logp_trace, axis=0)
    d_prev = np.zeros(nz + 1, dtype=np.float64)
    arg_b = np.zeros((ncls - 1, nz + 1), dtype=np.int32)
    for c in range(1, ncls):
        cand = d_prev + (s_pref[:, c - 1] - s_pref[:, c])
        d_cur = np.empty(nz + 1, dtype=np.float64)
        best_val = -np.inf
        best_idx = 0
        for b in range(nz + 1):
            if cand[b] > best_val:
                best_val = cand[b]
                best_idx = b
            d_cur[b] = best_val
            arg_b[c - 1, b] = best_idx
        d_prev = d_cur

    bs = np.zeros(ncls + 1, dtype=np.int32)
    bs[ncls] = nz
    cur_b = nz
    for c in range(ncls - 1, 0, -1):
        cur_b = int(arg_b[c - 1, cur_b])
        bs[c] = cur_b
    return bs


def _splits_to_outputs(
    bs: np.ndarray,
    cls_arr: np.ndarray,
    z_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert split indices ``[U, C + 1]`` into boundary depths and facies labels."""
    nu = bs.shape[0]
    nz = z_axis.size
    ncls = cls_arr.size
    iface_z = 0.5 * (z_axis[:-1] + z_axis[1:])
    split_depths = np.zeros((nu, max(ncls - 1, 0)), dtype=np.float32)
    seg_labels = np.zeros((nu, nz), dtype=np.int64)
    for u in range(nu):
        for c in range(ncls - 1):
            b_val = int(bs[u, c + 1])
            if b_val <= 0:
                split_depths[u, c] = z_axis[0] - 0.5
            elif b_val >= nz:
                split_depths[u, c] = z_axis[-1] + 0.5
            else:
                split_depths[u, c] = iface_z[b_val - 1]
        for c in range(ncls):
            seg_labels[u, bs[u, c] : bs[u, c + 1]] = int(cls_arr[c])
    return split_depths, seg_labels


def monotonic_dp_boundaries(
    log_probs: np.ndarray,
    classes: np.ndarray,
    z_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``O(C * Z)`` monotonic stratigraphic boundary DP with pinch-out support.

    Returns ``(split_depths [U, C - 1], seg_labels [U, Z])``.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    cls_arr = np.asarray(classes, dtype=np.int64)
    nu, nz, ncls = lp.shape
    z_axis = _z_axis(nz, z_coords)
    if ncls == 1:
        return (
            np.zeros((nu, 0), dtype=np.float32),
            np.full((nu, nz), int(cls_arr[0]), dtype=np.int64),
        )
    bs = np.zeros((nu, ncls + 1), dtype=np.int32)
    for u in range(nu):
        bs[u] = _dp_trace_splits(lp[u])
    return _splits_to_outputs(bs, cls_arr, z_axis)


def _viterbi_step_forward(
    dp_prev: np.ndarray,
    t_cur: np.ndarray,
    t_prev: np.ndarray,
    *,
    lo_u: int,
    hi_u: int,
    lo_p: int,
    hi_p: int,
    lam_lat: float,
    max_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_b = dp_prev.size
    best_c = np.full(n_b, np.inf, dtype=np.float64)
    best_p = np.zeros(n_b, dtype=np.int32)
    for off in range(-max_step, max_step + 1):
        b_lo = max(lo_u, lo_p - off)
        b_hi = min(hi_u, hi_p - off)
        if b_lo > b_hi:
            continue
        bb = np.arange(b_lo, b_hi + 1)
        pb = bb + off
        cand = dp_prev[pb] + lam_lat * np.abs(t_cur[bb] - t_prev[pb])
        m = cand < best_c[bb]
        best_c[bb[m]] = cand[m]
        best_p[bb[m]] = pb[m]
    missing = ~np.isfinite(best_c[lo_u : hi_u + 1])
    if np.any(missing):
        p_best = int(np.argmin(dp_prev[lo_p : hi_p + 1])) + lo_p
        p_val = dp_prev[p_best]
        for b in np.arange(lo_u, hi_u + 1)[missing]:
            best_c[b] = p_val + lam_lat * abs(float(t_cur[b] - t_prev[p_best]))
            best_p[b] = p_best
    return best_c, best_p


def _viterbi_refine_horizon(
    unary_cost: np.ndarray,
    t_grid: np.ndarray,
    lo_bounds: np.ndarray,
    hi_bounds: np.ndarray,
    *,
    lam_lat: float = 0.6,
    max_step: int = 24,
) -> np.ndarray:
    """Global 2D Viterbi path for one ordered split ``b_c(u) in [lo_u, hi_u]``."""
    nu, n_b = unary_cost.shape
    dp = np.full((nu, n_b), np.inf, dtype=np.float64)
    back = np.zeros((nu, n_b), dtype=np.int32)

    lo0, hi0 = int(lo_bounds[0]), int(hi_bounds[0])
    dp[0, lo0 : hi0 + 1] = unary_cost[0, lo0 : hi0 + 1]

    for u in range(1, nu):
        lo_u, hi_u = int(lo_bounds[u]), int(hi_bounds[u])
        lo_p, hi_p = int(lo_bounds[u - 1]), int(hi_bounds[u - 1])
        best_c, best_p = _viterbi_step_forward(
            dp[u - 1],
            t_grid[u],
            t_grid[u - 1],
            lo_u=lo_u,
            hi_u=hi_u,
            lo_p=lo_p,
            hi_p=hi_p,
            lam_lat=lam_lat,
            max_step=max_step,
        )
        dp[u, lo_u : hi_u + 1] = (
            best_c[lo_u : hi_u + 1] + unary_cost[u, lo_u : hi_u + 1]
        )
        back[u, lo_u : hi_u + 1] = best_p[lo_u : hi_u + 1]

    path = np.zeros(nu, dtype=np.int32)
    lo_last, hi_last = int(lo_bounds[-1]), int(hi_bounds[-1])
    cur = lo_last + int(np.argmin(dp[-1, lo_last : hi_last + 1]))
    path[-1] = cur
    for u in range(nu - 2, -1, -1):
        cur = int(back[u + 1, cur])
        path[u] = cur
    return path


def coupled_monotonic_boundaries_2d(
    log_probs: np.ndarray,
    classes: np.ndarray,
    rgt: np.ndarray,
    z_coords: np.ndarray | None = None,
    *,
    lam_lat: float = 4.0,
    max_step: int = 24,
    n_sweeps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """RGT-coupled 2D monotonic Viterbi boundary reconstruction."""
    lp = np.asarray(log_probs, dtype=np.float64)
    cls_arr = np.asarray(classes, dtype=np.int64)
    nu, nz, ncls = lp.shape
    z_axis = _z_axis(nz, z_coords)

    if ncls == 1:
        return (
            np.zeros((nu, 0), dtype=np.float32),
            np.full((nu, nz), int(cls_arr[0]), dtype=np.int64),
        )

    s_pref = np.zeros((nu, nz + 1, ncls), dtype=np.float64)
    s_pref[:, 1:, :] = np.cumsum(lp, axis=1)
    t_ext = np.zeros((nu, nz + 1), dtype=np.float64)
    t_ext[:, 0] = rgt[:, 0] - 0.5
    t_ext[:, 1:nz] = 0.5 * (rgt[:, :-1] + rgt[:, 1:])
    t_ext[:, nz] = rgt[:, -1] + 0.5

    bs = np.zeros((nu, ncls + 1), dtype=np.int32)
    for u in range(nu):
        bs[u] = _dp_trace_splits(lp[u])
    if nu >= _NDIM_4D + 1:
        bs = median_filter(bs, size=(min(9, nu), 1), mode="nearest").astype(np.int32)
        bs[:, 0] = 0
        bs[:, ncls] = nz

    for _ in range(n_sweeps):
        for c in range(1, ncls):
            unary = -(s_pref[:, :, c - 1] - s_pref[:, :, c])
            bs[:, c] = _viterbi_refine_horizon(
                unary,
                t_ext,
                bs[:, c - 1],
                bs[:, c + 1],
                lam_lat=lam_lat,
                max_step=max_step,
            )

    return _splits_to_outputs(bs, cls_arr, z_axis)


def _append_dip_equations_3d(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    vals: list[np.ndarray],
    rhs: list[np.ndarray],
    *,
    slope: np.ndarray,
    coh: np.ndarray,
    axis: int,
    w_dip: float,
    eq_offset: int,
) -> int:
    """Append 3D along-dip finite-difference equations along inline/crossline."""
    ni, nx, nz = slope.shape
    if nz <= _NDIM_2D or (axis == 0 and ni <= 1) or (axis == 1 and nx <= 1):
        return eq_offset
    i_max = ni - 1 if axis == 0 else ni
    x_max = nx - 1 if axis == 1 else nx
    ii, xx, zz = np.meshgrid(
        np.arange(i_max), np.arange(x_max), np.arange(1, nz - 1), indexing="ij"
    )
    ii_f, xx_f, zz_f = ii.ravel(), xx.ravel(), zz.ravel()
    w = np.sqrt(w_dip * coh[ii_f, xx_f, zz_f])
    s = slope[ii_f, xx_f, zz_f]
    n_eq = ii_f.size
    eqs = np.arange(eq_offset, eq_offset + n_eq)

    def _flat(i_a: np.ndarray, x_a: np.ndarray, z_a: np.ndarray) -> np.ndarray:
        return (i_a * nx + x_a) * nz + z_a

    i_next = ii_f + 1 if axis == 0 else ii_f
    x_next = xx_f + 1 if axis == 1 else xx_f
    rows.extend([eqs, eqs, eqs, eqs])
    cols.extend(
        [
            _flat(i_next, x_next, zz_f),
            _flat(ii_f, xx_f, zz_f),
            _flat(ii_f, xx_f, zz_f + 1),
            _flat(ii_f, xx_f, zz_f - 1),
        ]
    )
    vals.extend([w, -w, 0.5 * s * w, -0.5 * s * w])
    rhs.append(np.zeros(n_eq, dtype=np.float64))
    return eq_offset + n_eq


def solve_rgt_3d(
    emb_3d: np.ndarray,
    *,
    grad_sigma: tuple[float, float, float] = (0.8, 0.8, 1.2),
    tensor_sigma: tuple[float, float, float] = (2.0, 2.0, 3.5),
    max_slope: float = 8.0,
    w_dip: float = 10.0,
    w_mono: float = 1.0,
) -> np.ndarray:
    """Solve 3D Relative Geological Time ``T[I, X, Z]`` from ``[I, X, Z, D]``."""
    arr = np.asarray(emb_3d, dtype=np.float32)
    if arr.ndim != _NDIM_4D:
        raise ValueError(f"Expected 4D [I, X, Z, D] volume, got {arr.shape}")
    ni, nx, nz, _ = arr.shape
    sm = gaussian_filter(arr, sigma=(*grad_sigma, 0.0))
    gi = np.gradient(sm, axis=0)
    gx = np.gradient(sm, axis=1)
    gz = np.gradient(sm, axis=2)

    jii = gaussian_filter(np.sum(gi * gi, axis=3), sigma=tensor_sigma)
    jxx = gaussian_filter(np.sum(gx * gx, axis=3), sigma=tensor_sigma)
    jzz = gaussian_filter(np.sum(gz * gz, axis=3), sigma=tensor_sigma)
    jiz = gaussian_filter(np.sum(gi * gz, axis=3), sigma=tensor_sigma)
    jxz = gaussian_filter(np.sum(gx * gz, axis=3), sigma=tensor_sigma)

    p_i = np.clip(-jiz / (jzz + _MIN_DENOM), -max_slope, max_slope)
    p_x = np.clip(-jxz / (jzz + _MIN_DENOM), -max_slope, max_slope)
    coh = np.clip(jzz / (jii + jxx + jzz + _MIN_DENOM), 0.1, 1.0)

    n_vars = ni * nx * nz
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    eq_offset = _append_dip_equations_3d(
        rows, cols, vals, rhs, slope=p_i, coh=coh, axis=0, w_dip=w_dip, eq_offset=0
    )
    eq_offset = _append_dip_equations_3d(
        rows,
        cols,
        vals,
        rhs,
        slope=p_x,
        coh=coh,
        axis=1,
        w_dip=w_dip,
        eq_offset=eq_offset,
    )

    ii_m, xx_m, zz_m = np.meshgrid(
        np.arange(ni), np.arange(nx), np.arange(nz - 1), indexing="ij"
    )
    ii_mf, xx_mf, zz_mf = ii_m.ravel(), xx_m.ravel(), zz_m.ravel()
    n_m = ii_mf.size
    eqs_m = np.arange(eq_offset, eq_offset + n_m)
    wm = np.full(n_m, np.sqrt(w_mono), dtype=np.float64)
    flat_base = (ii_mf * nx + xx_mf) * nz + zz_mf
    rows.extend([eqs_m, eqs_m])
    cols.extend([flat_base + 1, flat_base])
    vals.extend([wm, -wm])
    rhs.append(wm)
    eq_offset += n_m

    anchor_idx = ((ni // 2) * nx + (nx // 2)) * nz + (nz // 2)
    rows.append(np.array([eq_offset]))
    cols.append(np.array([anchor_idx]))
    vals.append(np.array([10.0]))
    rhs.append(np.array([10.0 * float(nz // 2)]))
    eq_offset += 1

    mat = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(eq_offset, n_vars),
    ).tocsr()
    b_vec = np.concatenate(rhs)
    ata = (mat.T @ mat).tocsr()
    atb = mat.T @ b_vec
    diag = np.maximum(ata.diagonal(), _MIN_DENOM)
    precond = sp.diags(1.0 / diag)
    sol, _ = spla.cg(ata, atb, M=precond, rtol=1e-5, maxiter=400)
    return sol.reshape(ni, nx, nz).astype(np.float32)


def extract_surface_iso_rgt_3d(
    rgt_3d: np.ndarray,
    seed_i: int,
    seed_x: int,
    seed_z: float,
    z_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Extract 2D horizon surface ``[I, X]`` from 3D RGT at seed point."""
    ni, nx, nz = rgt_3d.shape
    flat_rgt = rgt_3d.reshape(ni * nx, nz)
    target = float(
        np.interp(
            float(seed_z), np.arange(nz, dtype=np.float32), rgt_3d[seed_i, seed_x]
        )
    )
    surf = _rgt_contour_at_value(flat_rgt, target, z_coords=z_coords)
    return surf.reshape(ni, nx)


def evaluate_boundary_map(
    true_iface: np.ndarray,
    pred_score: np.ndarray,
    pred_ridges: np.ndarray,
    z_coords: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate continuous boundary likelihood (AUC/AP) and 1-px ridge distances."""
    y_true = np.asarray(true_iface, dtype=bool)
    score = np.asarray(pred_score, dtype=np.float32)
    ridges = np.asarray(pred_ridges, dtype=bool)

    out: dict[str, float] = {}
    if np.unique(y_true).size == _MIN_BINARY_CLASSES:
        out["boundary_roc_auc"] = round(
            float(roc_auc_score(y_true.ravel(), score.ravel())), 4
        )
        out["boundary_pr_auc"] = round(
            float(average_precision_score(y_true.ravel(), score.ravel())), 4
        )
    else:
        out["boundary_roc_auc"] = float("nan")
        out["boundary_pr_auc"] = float("nan")

    nz = y_true.shape[-1] + 1
    z_axis = _z_axis(nz, z_coords)
    iface_z = 0.5 * (z_axis[:-1] + z_axis[1:])

    y_2d = y_true.reshape(-1, nz - 1)
    r_2d = ridges.reshape(-1, nz - 1)
    p2t: list[float] = []
    t2p: list[float] = []
    for u in range(y_2d.shape[0]):
        tz = iface_z[y_2d[u]]
        pz = iface_z[r_2d[u]]
        if tz.size > 0 and pz.size > 0:
            dmat = np.abs(pz[:, None] - tz[None, :])
            p2t.extend(dmat.min(axis=1).tolist())
            t2p.extend(dmat.min(axis=0).tolist())

    if p2t and t2p:
        dp = np.asarray(p2t, dtype=np.float32)
        dt = np.asarray(t2p, dtype=np.float32)
        out["ridge_p2t_med"] = round(float(np.median(dp)), 2)
        out["ridge_p2t_mean"] = round(float(dp.mean()), 2)
        out["ridge_t2p_med"] = round(float(np.median(dt)), 2)
        out["ridge_t2p_mean"] = round(float(dt.mean()), 2)
        out["ridge_prec_3px"] = round(float((dp <= _TOL_3PX).mean()), 4)
        out["ridge_prec_5px"] = round(float((dp <= _TOL_5PX).mean()), 4)
        out["ridge_rec_3px"] = round(float((dt <= _TOL_3PX).mean()), 4)
        out["ridge_rec_5px"] = round(float((dt <= _TOL_5PX).mean()), 4)
    return out


def evaluate_horizon_curve(
    true_depth: np.ndarray, pred_depth: np.ndarray
) -> dict[str, float]:
    """Compare true and predicted horizon depths (``NaN`` marks absent traces)."""
    t = np.asarray(true_depth, dtype=np.float32).ravel()
    p = np.asarray(pred_depth, dtype=np.float32).ravel()
    t_mask = np.isfinite(t)
    p_mask = np.isfinite(p)
    both = t_mask & p_mask

    tp = int((t_mask & p_mask).sum())
    fp = int((~t_mask & p_mask).sum())
    fn = int((t_mask & ~p_mask).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    res: dict[str, float] = {
        "n_true": float(t_mask.sum()),
        "n_pred": float(p_mask.sum()),
        "n_both": float(both.sum()),
        "presence_f1": round(float(f1), 4),
    }
    if both.any():
        err = np.abs(p[both] - t[both])
        res["mae"] = round(float(err.mean()), 3)
        res["medae"] = round(float(np.median(err)), 3)
        res["rmse"] = round(float(np.sqrt(np.mean(err * err))), 3)
        res["within_3px"] = round(float((err <= _TOL_3PX).mean()), 4)
        res["within_5px"] = round(float((err <= _TOL_5PX).mean()), 4)
    else:
        res["mae"] = float("nan")
        res["medae"] = float("nan")
        res["rmse"] = float("nan")
        res["within_3px"] = 0.0
        res["within_5px"] = 0.0
    return res
