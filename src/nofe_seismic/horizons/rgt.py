"""Structure-tensor dip estimation and Relative Geological Time (RGT) solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter, maximum_filter1d

from .boundaries import z_axis_coords

_EPS = 1e-8
_MIN_DENOM = 1e-6
_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4


@dataclass(frozen=True)
class DipCoherencyField:
    """2D structural slope ``dz/du`` and anisotropy coherency ``[U, Z]``."""

    slope: np.ndarray
    coherency: np.ndarray


@dataclass(frozen=True)
class DiscoveredHorizon:
    """Zero-seed horizon extracted at a peak of RGT-flattened contrast."""

    rank: int
    tau: float
    contrast_score: float
    depths: np.ndarray


@dataclass
class _SparseEquations:
    rows: list[np.ndarray]
    cols: list[np.ndarray]
    vals: list[np.ndarray]
    rhs: list[np.ndarray]
    eq_offset: int = 0

    def append(
        self,
        row_list: list[np.ndarray],
        col_list: list[np.ndarray],
        val_list: list[np.ndarray],
        rhs_vec: np.ndarray,
    ) -> None:
        self.rows.extend(row_list)
        self.cols.extend(col_list)
        self.vals.extend(val_list)
        self.rhs.append(rhs_vec)
        self.eq_offset += int(rhs_vec.size)


def _flat_2d(u_idx: np.ndarray, z_idx: np.ndarray, nz: int) -> np.ndarray:
    return u_idx * nz + z_idx


def _flat_3d(
    i_idx: np.ndarray, x_idx: np.ndarray, z_idx: np.ndarray, *, nx: int, nz: int
) -> np.ndarray:
    return (i_idx * nx + x_idx) * nz + z_idx


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
    gu = np.asarray(np.gradient(sm, axis=0), dtype=np.float32)
    gz = np.asarray(np.gradient(sm, axis=1), dtype=np.float32)
    juu = gaussian_filter(np.sum(gu * gu, axis=2), sigma=tensor_sigma)
    jzz = gaussian_filter(np.sum(gz * gz, axis=2), sigma=tensor_sigma)
    juz = gaussian_filter(np.sum(gu * gz, axis=2), sigma=tensor_sigma)

    slope = np.clip(-juz / (jzz + _MIN_DENOM), -max_slope, max_slope).astype(np.float32)
    trace, diff = juu + jzz, juu - jzz
    rad = np.sqrt(np.maximum(diff * diff + 4.0 * juz * juz, 0.0))
    lam1, lam2 = 0.5 * (trace + rad), 0.5 * (trace - rad)
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
    slope, coh = field.slope, field.coherency
    nu, nz = slope.shape
    eqs = _SparseEquations(rows=[], cols=[], vals=[], rhs=[])

    if nu > 1 and nz > _NDIM_2D:
        uu, zz = np.meshgrid(np.arange(nu - 1), np.arange(1, nz - 1), indexing="ij")
        uu_f, zz_f = uu.ravel(), zz.ravel()
        w = np.sqrt(w_dip * np.clip(coh[uu_f, zz_f], 0.1, 1.0))
        s = slope[uu_f, zz_f]
        idx = np.arange(eqs.eq_offset, eqs.eq_offset + uu_f.size)
        eqs.append(
            [idx, idx, idx, idx],
            [
                _flat_2d(uu_f + 1, zz_f, nz),
                _flat_2d(uu_f, zz_f, nz),
                _flat_2d(uu_f, zz_f + 1, nz),
                _flat_2d(uu_f, zz_f - 1, nz),
            ],
            [w, -w, 0.5 * s * w, -0.5 * s * w],
            np.zeros(uu_f.size, dtype=np.float64),
        )

    uu_m, zz_m = np.meshgrid(np.arange(nu), np.arange(nz - 1), indexing="ij")
    uu_mf, zz_mf = uu_m.ravel(), zz_m.ravel()
    idx_m = np.arange(eqs.eq_offset, eqs.eq_offset + uu_mf.size)
    wm = np.full(uu_mf.size, np.sqrt(w_mono), dtype=np.float64)
    eqs.append(
        [idx_m, idx_m],
        [_flat_2d(uu_mf, zz_mf + 1, nz), _flat_2d(uu_mf, zz_mf, nz)],
        [wm, -wm],
        wm,
    )

    au, az = (nu // 2, nz // 2) if anchor is None else anchor
    eqs.append(
        [np.array([eqs.eq_offset])],
        [np.array([au * nz + az])],
        [np.array([10.0])],
        np.array([10.0 * float(az)]),
    )

    mat = sp.coo_matrix(
        (
            np.concatenate(eqs.vals),
            (np.concatenate(eqs.rows), np.concatenate(eqs.cols)),
        ),
        shape=(eqs.eq_offset, nu * nz),
    ).tocsc()
    b_vec = np.concatenate(eqs.rhs)
    sol = spla.spsolve(mat.T @ mat, mat.T @ b_vec)
    return sol.reshape(nu, nz).astype(np.float32)


def _rgt_contour_at_value(
    rgt: np.ndarray, target_t: float, z_coords: np.ndarray | None = None
) -> np.ndarray:
    nu, nz = rgt.shape
    z_axis = z_axis_coords(nz, z_coords)
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
) -> list[DiscoveredHorizon]:
    """Discover major boundary horizons without seeds via RGT-flattened contrast."""
    nu, nz = rgt.shape
    t_iface = 0.5 * (rgt[:, :-1] + rgt[:, 1:])
    w_iface = np.clip(0.5 * (coherency[:, :-1] + coherency[:, 1:]), 0.05, 1.0)
    tau_grid = np.linspace(
        float(np.percentile(t_iface, 2.0)),
        float(np.percentile(t_iface, 98.0)),
        max(nz - 1, 8),
        dtype=np.float32,
    )

    weighted_sum = np.zeros_like(tau_grid, dtype=np.float64)
    weight_total = np.zeros_like(tau_grid, dtype=np.float64)
    for u in range(nu):
        t_col = np.maximum.accumulate(t_iface[u])
        in_range = (tau_grid >= t_col[0]) & (tau_grid <= t_col[-1])
        w_interp = np.interp(tau_grid, t_col, w_iface[u]) * in_range
        weighted_sum += np.interp(tau_grid, t_col, contrast[u]) * w_interp
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

    return [
        DiscoveredHorizon(
            rank=rank,
            tau=round(float(tau_grid[idx]), 3),
            contrast_score=round(float(spectrum[idx]), 4),
            depths=_rgt_contour_at_value(rgt, float(tau_grid[idx]), z_coords=z_coords),
        )
        for rank, idx in enumerate(chosen)
    ]


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
    tau_grid = np.linspace(float(rgt.min()), float(rgt.max()), nz, dtype=np.float32)

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


def _append_dip_equations_3d(
    eqs: _SparseEquations,
    slope: np.ndarray,
    coh: np.ndarray,
    *,
    axis: int,
    w_dip: float,
) -> None:
    ni, nx, nz = slope.shape
    if nz <= _NDIM_2D or (axis == 0 and ni <= 1) or (axis == 1 and nx <= 1):
        return
    i_max = ni - 1 if axis == 0 else ni
    x_max = nx - 1 if axis == 1 else nx
    ii, xx, zz = np.meshgrid(
        np.arange(i_max), np.arange(x_max), np.arange(1, nz - 1), indexing="ij"
    )
    ii_f, xx_f, zz_f = ii.ravel(), xx.ravel(), zz.ravel()
    w = np.sqrt(w_dip * coh[ii_f, xx_f, zz_f])
    s = slope[ii_f, xx_f, zz_f]
    idx = np.arange(eqs.eq_offset, eqs.eq_offset + ii_f.size)
    i_next = ii_f + 1 if axis == 0 else ii_f
    x_next = xx_f + 1 if axis == 1 else xx_f
    eqs.append(
        [idx, idx, idx, idx],
        [
            _flat_3d(i_next, x_next, zz_f, nx=nx, nz=nz),
            _flat_3d(ii_f, xx_f, zz_f, nx=nx, nz=nz),
            _flat_3d(ii_f, xx_f, zz_f + 1, nx=nx, nz=nz),
            _flat_3d(ii_f, xx_f, zz_f - 1, nx=nx, nz=nz),
        ],
        [w, -w, 0.5 * s * w, -0.5 * s * w],
        np.zeros(ii_f.size, dtype=np.float64),
    )


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
    gi = np.asarray(np.gradient(sm, axis=0), dtype=np.float32)
    gx = np.asarray(np.gradient(sm, axis=1), dtype=np.float32)
    gz = np.asarray(np.gradient(sm, axis=2), dtype=np.float32)

    jii = gaussian_filter(np.sum(gi * gi, axis=3), sigma=tensor_sigma)
    jxx = gaussian_filter(np.sum(gx * gx, axis=3), sigma=tensor_sigma)
    jzz = gaussian_filter(np.sum(gz * gz, axis=3), sigma=tensor_sigma)
    jiz = gaussian_filter(np.sum(gi * gz, axis=3), sigma=tensor_sigma)
    jxz = gaussian_filter(np.sum(gx * gz, axis=3), sigma=tensor_sigma)

    p_i = np.clip(-jiz / (jzz + _MIN_DENOM), -max_slope, max_slope)
    p_x = np.clip(-jxz / (jzz + _MIN_DENOM), -max_slope, max_slope)
    coh = np.clip(jzz / (jii + jxx + jzz + _MIN_DENOM), 0.1, 1.0)

    eqs = _SparseEquations(rows=[], cols=[], vals=[], rhs=[])
    _append_dip_equations_3d(eqs, p_i, coh, axis=0, w_dip=w_dip)
    _append_dip_equations_3d(eqs, p_x, coh, axis=1, w_dip=w_dip)

    ii_m, xx_m, zz_m = np.meshgrid(
        np.arange(ni), np.arange(nx), np.arange(nz - 1), indexing="ij"
    )
    ii_mf, xx_mf, zz_mf = ii_m.ravel(), xx_m.ravel(), zz_m.ravel()
    idx_m = np.arange(eqs.eq_offset, eqs.eq_offset + ii_mf.size)
    wm = np.full(ii_mf.size, np.sqrt(w_mono), dtype=np.float64)
    flat_base = _flat_3d(ii_mf, xx_mf, zz_mf, nx=nx, nz=nz)
    eqs.append([idx_m, idx_m], [flat_base + 1, flat_base], [wm, -wm], wm)

    anchor_idx = ((ni // 2) * nx + (nx // 2)) * nz + (nz // 2)
    eqs.append(
        [np.array([eqs.eq_offset])],
        [np.array([anchor_idx])],
        [np.array([10.0])],
        np.array([10.0 * float(nz // 2)]),
    )

    mat = sp.coo_matrix(
        (
            np.concatenate(eqs.vals),
            (np.concatenate(eqs.rows), np.concatenate(eqs.cols)),
        ),
        shape=(eqs.eq_offset, ni * nx * nz),
    ).tocsr()
    ata = (mat.T @ mat).tocsr()
    atb = mat.T @ np.concatenate(eqs.rhs)
    precond = sp.diags(1.0 / np.maximum(ata.diagonal(), _MIN_DENOM))
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
    target = float(
        np.interp(
            float(seed_z), np.arange(nz, dtype=np.float32), rgt_3d[seed_i, seed_x]
        )
    )
    return _rgt_contour_at_value(
        rgt_3d.reshape(ni * nx, nz), target, z_coords=z_coords
    ).reshape(ni, nx)
