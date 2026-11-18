"""Seismic boundary extraction, multi-scale contrast and evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score

_EPS = 1e-8
_MIN_BINARY_CLASSES = 2
_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4
_TOL_3PX = 3.0
_TOL_5PX = 5.0


def z_axis_coords(nz: int, z_coords: np.ndarray | None) -> np.ndarray:
    """Return physical depth coordinates or integer sample indices ``0..nz-1``."""
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
    z_axis = z_axis_coords(lbl.shape[-1], z_coords)
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
    z_axis = z_axis_coords(lbl.shape[-1], z_coords)
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
        c_w = np.linalg.norm((below - above) / float(win), axis=-1).astype(np.float32)
        scale = float(np.std(c_w))
        total += c_w / (scale if scale > _EPS else 1.0)
    return (total / float(len(windows))).astype(np.float32)


def spatial_gradient_magnitude(
    emb_grid: np.ndarray, *, smooth_sigma: float = 0.6
) -> np.ndarray:
    """Compute Euclidean spatial gradient norm over 2D or 3D embedding grids."""
    arr = np.asarray(emb_grid, dtype=np.float32)
    if arr.ndim not in (_NDIM_3D, _NDIM_4D):
        raise ValueError(f"Expected 3D or 4D grid, got {arr.shape}")
    spatial_axes = arr.ndim - 1
    sig = (smooth_sigma,) * spatial_axes + (0.0,)
    sm = gaussian_filter(arr, sigma=sig) if smooth_sigma > 0 else arr
    sq_sum = np.zeros(arr.shape[:-1], dtype=np.float32)
    for ax in range(spatial_axes):
        g_ax = np.asarray(np.gradient(sm, axis=ax), dtype=np.float32)
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
    z_axis = z_axis_coords(nz, z_coords)
    iface_z = 0.5 * (z_axis[:-1] + z_axis[1:])

    y_2d, r_2d = y_true.reshape(-1, nz - 1), ridges.reshape(-1, nz - 1)
    p2t: list[float] = []
    t2p: list[float] = []
    for u in range(y_2d.shape[0]):
        tz, pz = iface_z[y_2d[u]], iface_z[r_2d[u]]
        if tz.size > 0 and pz.size > 0:
            dmat = np.abs(pz[:, None] - tz[None, :])
            p2t.extend(dmat.min(axis=1).tolist())
            t2p.extend(dmat.min(axis=0).tolist())

    if p2t and t2p:
        dp = np.asarray(p2t, dtype=np.float32)
        dt = np.asarray(t2p, dtype=np.float32)
        out.update(
            {
                "ridge_p2t_med": round(float(np.median(dp)), 2),
                "ridge_p2t_mean": round(float(dp.mean()), 2),
                "ridge_t2p_med": round(float(np.median(dt)), 2),
                "ridge_t2p_mean": round(float(dt.mean()), 2),
                "ridge_prec_3px": round(float((dp <= _TOL_3PX).mean()), 4),
                "ridge_prec_5px": round(float((dp <= _TOL_5PX).mean()), 4),
                "ridge_rec_3px": round(float((dt <= _TOL_3PX).mean()), 4),
                "ridge_rec_5px": round(float((dt <= _TOL_5PX).mean()), 4),
            }
        )
    return out


def evaluate_horizon_curve(
    true_depth: np.ndarray, pred_depth: np.ndarray
) -> dict[str, float]:
    """Compare true and predicted horizon depths (``NaN`` marks absent traces)."""
    t = np.asarray(true_depth, dtype=np.float32).ravel()
    p = np.asarray(pred_depth, dtype=np.float32).ravel()
    t_mask, p_mask = np.isfinite(t), np.isfinite(p)
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
    if not both.any():
        res.update(
            {
                "mae": float("nan"),
                "medae": float("nan"),
                "rmse": float("nan"),
                "within_3px": 0.0,
                "within_5px": 0.0,
            }
        )
        return res

    err = np.abs(p[both] - t[both])
    res.update(
        {
            "mae": round(float(err.mean()), 3),
            "medae": round(float(np.median(err)), 3),
            "rmse": round(float(np.sqrt(np.mean(err * err))), 3),
            "within_3px": round(float((err <= _TOL_3PX).mean()), 4),
            "within_5px": round(float((err <= _TOL_5PX).mean()), 4),
        }
    )
    return res
