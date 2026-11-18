"""1D and RGT-coupled 2D monotonic stratigraphic boundary dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.ndimage import median_filter

from .boundaries import z_axis_coords

_MIN_SMOOTH_TRACES = 5


class BoundarySegmentation(NamedTuple):
    """Stratigraphic boundary depths ``[U, C - 1]`` and unit labels ``[U, Z]``."""

    split_depths: np.ndarray
    labels: np.ndarray


class _ViterbiStep(NamedTuple):
    cost: np.ndarray
    prev_idx: np.ndarray


@dataclass(frozen=True)
class _TraceBounds:
    lo_u: int
    hi_u: int
    lo_p: int
    hi_p: int


def _dp_trace_splits(logp_trace: np.ndarray) -> np.ndarray:
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
    bs: np.ndarray, cls_arr: np.ndarray, z_axis: np.ndarray
) -> BoundarySegmentation:
    nu, nz, ncls = bs.shape[0], z_axis.size, cls_arr.size
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
    return BoundarySegmentation(split_depths=split_depths, labels=seg_labels)


def monotonic_dp_boundaries(
    log_probs: np.ndarray,
    classes: np.ndarray,
    z_coords: np.ndarray | None = None,
) -> BoundarySegmentation:
    """Exact ``O(C * Z)`` monotonic stratigraphic boundary DP with pinch-outs."""
    lp = np.asarray(log_probs, dtype=np.float64)
    cls_arr = np.asarray(classes, dtype=np.int64)
    nu, nz, ncls = lp.shape
    z_axis = z_axis_coords(nz, z_coords)
    if ncls == 1:
        return BoundarySegmentation(
            split_depths=np.zeros((nu, 0), dtype=np.float32),
            labels=np.full((nu, nz), int(cls_arr[0]), dtype=np.int64),
        )
    bs = np.stack([_dp_trace_splits(lp[u]) for u in range(nu)], axis=0)
    return _splits_to_outputs(bs, cls_arr, z_axis)


def _viterbi_step_forward(
    dp_prev: np.ndarray,
    t_cur: np.ndarray,
    t_prev: np.ndarray,
    bounds: _TraceBounds,
    *,
    lam_lat: float,
    max_step: int,
) -> _ViterbiStep:
    best_c = np.full(dp_prev.size, np.inf, dtype=np.float64)
    best_p = np.zeros(dp_prev.size, dtype=np.int32)
    for off in range(-max_step, max_step + 1):
        b_lo = max(bounds.lo_u, bounds.lo_p - off)
        b_hi = min(bounds.hi_u, bounds.hi_p - off)
        if b_lo > b_hi:
            continue
        bb = np.arange(b_lo, b_hi + 1)
        pb = bb + off
        cand = dp_prev[pb] + lam_lat * np.abs(t_cur[bb] - t_prev[pb])
        m = cand < best_c[bb]
        best_c[bb[m]] = cand[m]
        best_p[bb[m]] = pb[m]
    missing = ~np.isfinite(best_c[bounds.lo_u : bounds.hi_u + 1])
    if np.any(missing):
        p_best = int(np.argmin(dp_prev[bounds.lo_p : bounds.hi_p + 1])) + bounds.lo_p
        p_val = dp_prev[p_best]
        for b in np.arange(bounds.lo_u, bounds.hi_u + 1)[missing]:
            best_c[b] = p_val + lam_lat * abs(float(t_cur[b] - t_prev[p_best]))
            best_p[b] = p_best
    return _ViterbiStep(cost=best_c, prev_idx=best_p)


def _viterbi_refine_horizon(
    unary_cost: np.ndarray,
    t_grid: np.ndarray,
    lo_bounds: np.ndarray,
    hi_bounds: np.ndarray,
    *,
    lam_lat: float = 0.6,
    max_step: int = 24,
) -> np.ndarray:
    nu, n_b = unary_cost.shape
    dp = np.full((nu, n_b), np.inf, dtype=np.float64)
    back = np.zeros((nu, n_b), dtype=np.int32)
    lo0, hi0 = int(lo_bounds[0]), int(hi_bounds[0])
    dp[0, lo0 : hi0 + 1] = unary_cost[0, lo0 : hi0 + 1]

    for u in range(1, nu):
        bounds = _TraceBounds(
            lo_u=int(lo_bounds[u]),
            hi_u=int(hi_bounds[u]),
            lo_p=int(lo_bounds[u - 1]),
            hi_p=int(hi_bounds[u - 1]),
        )
        step = _viterbi_step_forward(
            dp[u - 1],
            t_grid[u],
            t_grid[u - 1],
            bounds,
            lam_lat=lam_lat,
            max_step=max_step,
        )
        dp[u, bounds.lo_u : bounds.hi_u + 1] = (
            step.cost[bounds.lo_u : bounds.hi_u + 1]
            + unary_cost[u, bounds.lo_u : bounds.hi_u + 1]
        )
        back[u, bounds.lo_u : bounds.hi_u + 1] = step.prev_idx[
            bounds.lo_u : bounds.hi_u + 1
        ]

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
) -> BoundarySegmentation:
    """RGT-coupled 2D monotonic Viterbi boundary reconstruction."""
    lp = np.asarray(log_probs, dtype=np.float64)
    cls_arr = np.asarray(classes, dtype=np.int64)
    nu, nz, ncls = lp.shape
    z_axis = z_axis_coords(nz, z_coords)
    if ncls == 1:
        return BoundarySegmentation(
            split_depths=np.zeros((nu, 0), dtype=np.float32),
            labels=np.full((nu, nz), int(cls_arr[0]), dtype=np.int64),
        )

    s_pref = np.zeros((nu, nz + 1, ncls), dtype=np.float64)
    s_pref[:, 1:, :] = np.cumsum(lp, axis=1)
    t_ext = np.zeros((nu, nz + 1), dtype=np.float64)
    t_ext[:, 0] = rgt[:, 0] - 0.5
    t_ext[:, 1:nz] = 0.5 * (rgt[:, :-1] + rgt[:, 1:])
    t_ext[:, nz] = rgt[:, -1] + 0.5

    bs = np.stack([_dp_trace_splits(lp[u]) for u in range(nu)], axis=0)
    if nu >= _MIN_SMOOTH_TRACES:
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
