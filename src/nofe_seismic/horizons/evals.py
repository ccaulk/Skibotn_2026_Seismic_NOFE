"""Evaluation helpers for NOFE horizon boundary and surface reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from nofe import NOFE
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume, volume_to_points
from nofe_seismic.graph import (
    SeismicPointGraph,
    SharedStandardizer,
    embed_numpy,
    graph_summary,
)
from nofe_seismic.types import IndexRange, RowDict, Stride3D

from .boundaries import (
    evaluate_boundary_map,
    evaluate_horizon_curve,
    extract_boundary_ridges,
    extract_true_interfaces,
    extract_unit_top_depths,
    multiscale_boundary_contrast,
    spatial_gradient_magnitude,
)
from .config import HorizonConfig
from .dp import coupled_monotonic_boundaries_2d
from .rgt import (
    DipCoherencyField,
    DiscoveredHorizon,
    discover_horizons_from_rgt,
    estimate_structure_tensor_2d,
    extract_horizon_iso_rgt,
    rgt_aligned_smooth,
    solve_rgt_2d,
)

_MIN_BINARY_CLASSES = 2
_NMS_SIZE = 11
_TOP_K_RIDGES = 3
_DISCOVERED_COUNT = 4
_DISCOVERED_MIN_SEP = 14
_PROBE_HIDDEN_LAYERS = (128, 64)
_PROBE_MAX_ITER = 120
_SMOOTH_SIGMA_U = 1.8
_SMOOTH_SIGMA_Z = 1.2
_PROB_EPS = 1e-6


class HorizonSeed(NamedTuple):
    """Single seed coordinate ``(u, z)`` on a 2D section."""

    u: int
    z: int


class _NamedSectionGrid(NamedTuple):
    """Representation name paired with a 2D section tensor."""

    method: str
    grid: np.ndarray


@dataclass(frozen=True)
class SliceEmbedModels:
    """Standardiser and dimensionality-reduction models used to embed sections."""

    scaler: SharedStandardizer
    nofe: NOFE
    pca: PCA


@dataclass(frozen=True)
class NativeSlice:
    """2D seismic section embedded at native vertical resolution."""

    u_coords: np.ndarray
    z_coords: np.ndarray
    xlabel: str
    amplitude: np.ndarray
    labels: np.ndarray | None
    locations: np.ndarray
    x_raw: np.ndarray
    nofe: np.ndarray
    pca: np.ndarray
    graph: dict[str, float | int]


@dataclass(frozen=True)
class UnsupervisedMethodArt:
    """Unsupervised boundary and RGT artefacts for one representation."""

    contrast: np.ndarray
    ridges: np.ndarray
    grad_mag: np.ndarray
    dip_field: DipCoherencyField
    rgt: np.ndarray
    discovered: list[DiscoveredHorizon]
    seeded_curves: dict[int, np.ndarray]


@dataclass(frozen=True)
class UnsupervisedEvalResult:
    """Rows and artefacts produced by unsupervised boundary and RGT checks."""

    unsup_rows: list[RowDict]
    rgt_rows: list[RowDict]
    artifacts: dict[str, UnsupervisedMethodArt]


@dataclass(frozen=True)
class SeededRgtResult:
    """Predicted horizon curves and metric rows for seeded RGT tracking."""

    curves: dict[int, np.ndarray]
    rows: list[RowDict]


@dataclass(frozen=True)
class ProbeMethodArt:
    """Supervised probe segmentation and horizon curve artefacts for one method."""

    seg_pred: np.ndarray
    pred_tops: dict[int, np.ndarray]
    p_iface: np.ndarray
    voxel_acc: float


@dataclass(frozen=True)
class ProbeEvalResult:
    """Rows, artefacts and fitted probe pipelines across methods."""

    rows: list[RowDict]
    artifacts: dict[str, ProbeMethodArt]
    probes: dict[str, Pipeline]


def embed_slice_native_z(
    volume: SeismicVolume | LazySeismicVolume,
    models: SliceEmbedModels,
    *,
    axis: str,
    index: int,
    lateral_stride: int,
    radius: float,
) -> NativeSlice:
    """Embed a 2D seismic section at native vertical resolution (``sz = 1``)."""
    ni, nx, nz = volume.shape
    lat = max(1, int(lateral_stride))
    if axis == "inline":
        pts = volume_to_points(
            volume,
            inline_range=IndexRange(index, index + 1),
            stride=Stride3D(1, lat, 1),
        )
        u_coords = np.arange(0, nx, lat, dtype=np.float32)
        xlabel = "Crossline"
    elif axis == "crossline":
        pts = volume_to_points(
            volume,
            crossline_range=IndexRange(index, index + 1),
            stride=Stride3D(lat, 1, 1),
        )
        u_coords = np.arange(0, ni, lat, dtype=np.float32)
        xlabel = "Inline"
    else:
        raise ValueError(f"axis must be 'inline' or 'crossline', got {axis!r}")

    z_coords = np.arange(nz, dtype=np.float32)
    nu = int(u_coords.size)
    xs = models.scaler.transform_features(pts.features)
    ps = models.scaler.transform_locations(pts.locations)
    graph = SeismicPointGraph(mode="radius", radius=radius, normalize=False).build(
        xs, ps
    )
    z_nofe = embed_numpy(models.nofe, graph).reshape(nu, nz, -1)
    z_pca = models.pca.transform(xs).astype(np.float32).reshape(nu, nz, -1)

    return NativeSlice(
        u_coords=u_coords,
        z_coords=z_coords,
        xlabel=xlabel,
        amplitude=pts.amplitude.reshape(nu, nz),
        labels=pts.labels.reshape(nu, nz) if pts.labels is not None else None,
        locations=pts.locations,
        x_raw=xs.reshape(nu, nz, -1),
        nofe=z_nofe,
        pca=z_pca,
        graph=graph_summary(graph),
    )


def seed_for_unit_top(labels_2d: np.ndarray, unit_id: int) -> HorizonSeed | None:
    """Choose a median valid trace and depth for seeding ``labels >= unit_id``."""
    ge_mask = labels_2d >= unit_id
    valid_u = np.flatnonzero(ge_mask.any(axis=1))
    if valid_u.size == 0:
        return None
    seed_u = int(valid_u[valid_u.size // 2])
    first_z = int(np.argmax(ge_mask[seed_u]))
    return HorizonSeed(seed_u, max(0, first_z))


def _section_sources(sl_data: NativeSlice) -> list[_NamedSectionGrid]:
    return [
        _NamedSectionGrid("NOFE-8D", sl_data.nofe),
        _NamedSectionGrid("PCA-8D", sl_data.pca),
        _NamedSectionGrid("Raw-11D", sl_data.x_raw),
        _NamedSectionGrid("Amplitude-1D", sl_data.amplitude[:, :, None]),
    ]


def _score_seeded_rgt(
    lbl: np.ndarray,
    rgt: np.ndarray,
    true_tops: dict[int, np.ndarray],
    *,
    slice_key: str,
    method: str,
    unit_ids: Sequence[int],
    z_coords: np.ndarray,
) -> SeededRgtResult:
    seeded_curves: dict[int, np.ndarray] = {}
    rows: list[RowDict] = []
    for uid in unit_ids:
        seed = seed_for_unit_top(lbl, uid)
        if seed is None:
            continue
        pred_curve = extract_horizon_iso_rgt(rgt, seed.u, seed.z, z_coords=z_coords)
        seeded_curves[uid] = pred_curve
        ev: RowDict = dict(evaluate_horizon_curve(true_tops[uid], pred_curve))
        ev["slice"] = slice_key
        ev["method"] = method
        ev["horizon"] = f"Top_Unit_{uid}"
        ev["mode"] = "seeded_rgt"
        rows.append(ev)
    return SeededRgtResult(curves=seeded_curves, rows=rows)


def eval_unsupervised_and_rgt(
    sl_data: NativeSlice,
    slice_key: str,
    cfg: HorizonConfig,
    *,
    unit_ids: Sequence[int],
) -> UnsupervisedEvalResult:
    """Run unsupervised contrast, ridges, RGT tracking and zero-seed discovery."""
    lbl = sl_data.labels
    z_coords = sl_data.z_coords
    true_iface = extract_true_interfaces(lbl) if lbl is not None else None
    true_tops = (
        extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
    )

    unsup_rows: list[RowDict] = []
    rgt_rows: list[RowDict] = []
    artifacts: dict[str, UnsupervisedMethodArt] = {}

    for m_name, grid in _section_sources(sl_data):
        contrast = multiscale_boundary_contrast(grid, windows=cfg.contrast_windows)
        ridges = extract_boundary_ridges(
            contrast, nms_size=_NMS_SIZE, top_k=_TOP_K_RIDGES
        )
        grad_mag = spatial_gradient_magnitude(grid)

        if true_iface is not None:
            b_eval: RowDict = dict(
                evaluate_boundary_map(true_iface, contrast, ridges, z_coords=z_coords)
            )
            b_eval["slice"] = slice_key
            b_eval["method"] = m_name
            grad_iface = 0.5 * (grad_mag[:, :-1] + grad_mag[:, 1:])
            b_eval["grad_roc_auc"] = (
                round(float(roc_auc_score(true_iface.ravel(), grad_iface.ravel())), 4)
                if np.unique(true_iface).size == _MIN_BINARY_CLASSES
                else float("nan")
            )
            unsup_rows.append(b_eval)

        dip_field = estimate_structure_tensor_2d(grid, max_slope=cfg.max_slope)
        rgt = solve_rgt_2d(dip_field, w_dip=cfg.rgt_w_dip, w_mono=cfg.rgt_w_mono)
        discovered = discover_horizons_from_rgt(
            rgt,
            contrast,
            dip_field.coherency,
            num_horizons=_DISCOVERED_COUNT,
            min_sep=_DISCOVERED_MIN_SEP,
            z_coords=z_coords,
        )

        seeded_curves: dict[int, np.ndarray] = {}
        if lbl is not None:
            seeded = _score_seeded_rgt(
                lbl,
                rgt,
                true_tops,
                slice_key=slice_key,
                method=m_name,
                unit_ids=unit_ids,
                z_coords=z_coords,
            )
            seeded_curves = seeded.curves
            rgt_rows.extend(seeded.rows)

        artifacts[m_name] = UnsupervisedMethodArt(
            contrast=contrast,
            ridges=ridges,
            grad_mag=grad_mag,
            dip_field=dip_field,
            rgt=rgt,
            discovered=discovered,
            seeded_curves=seeded_curves,
        )

    return UnsupervisedEvalResult(
        unsup_rows=unsup_rows,
        rgt_rows=rgt_rows,
        artifacts=artifacts,
    )


def score_discovered_horizons(
    sl_inline: NativeSlice,
    discovered: list[DiscoveredHorizon],
    unit_ids: Sequence[int],
) -> list[dict[str, int | float | None]]:
    """Match zero-seed RGT horizons to the closest ground-truth unit top."""
    if sl_inline.labels is None:
        return []
    true_tops_in = extract_unit_top_depths(
        sl_inline.labels, unit_ids, sl_inline.z_coords
    )
    out: list[dict[str, int | float | None]] = []
    for item in discovered:
        best_uid: int | None = None
        best_mae = float("inf")
        best_ev: dict[str, float] = {}
        for uid in unit_ids:
            if not np.isfinite(true_tops_in[uid]).any():
                continue
            ev = evaluate_horizon_curve(true_tops_in[uid], item.depths)
            if np.isfinite(ev["mae"]) and ev["mae"] < best_mae:
                best_mae = ev["mae"]
                best_uid = uid
                best_ev = ev
        out.append(
            {
                "rank": item.rank,
                "contrast_score": item.contrast_score,
                "matched_unit": best_uid,
                "mae": best_ev.get("mae", float("nan")),
                "medae": best_ev.get("medae", float("nan")),
                "within_5px": best_ev.get("within_5px", 0.0),
            }
        )
    return out


def _eval_single_probe(
    clf: Pipeline,
    sl_data: NativeSlice,
    rgt: np.ndarray,
    *,
    method: str,
    unit_ids: Sequence[int],
) -> ProbeMethodArt:
    grid = (
        sl_data.nofe
        if method == "NOFE-8D"
        else (sl_data.pca if method == "PCA-8D" else sl_data.x_raw)
    )
    nu, nz, _ = grid.shape
    probs = clf.predict_proba(grid.reshape(-1, grid.shape[2])).reshape(nu, nz, -1)
    logp_raw = np.log(np.clip(probs, _PROB_EPS, 1.0)).astype(np.float32)
    logp_sm = rgt_aligned_smooth(
        logp_raw, rgt, sigma_u=_SMOOTH_SIGMA_U, sigma_z=_SMOOTH_SIGMA_Z
    )
    classes = np.asarray(clf.classes_, dtype=np.int64)
    _, seg_pred = coupled_monotonic_boundaries_2d(
        logp_sm,
        classes,
        rgt,
        z_coords=sl_data.z_coords,
        lam_lat=4.0,
        max_step=24,
    )
    pred_tops = extract_unit_top_depths(seg_pred, unit_ids, sl_data.z_coords)
    p_iface = 1.0 - np.sum(probs[:, :-1, :] * probs[:, 1:, :], axis=2)
    voxel_acc = (
        float((seg_pred == sl_data.labels).mean())
        if sl_data.labels is not None
        else float("nan")
    )
    return ProbeMethodArt(
        seg_pred=seg_pred,
        pred_tops=pred_tops,
        p_iface=p_iface,
        voxel_acc=voxel_acc,
    )


def eval_probe_horizons(
    train_embeds: dict[str, np.ndarray],
    train_labels: np.ndarray,
    sl_data: NativeSlice,
    unsup_artifacts: dict[str, UnsupervisedMethodArt],
    *,
    slice_key: str,
    cfg: HorizonConfig,
    unit_ids: Sequence[int],
    probes_cache: dict[str, Pipeline] | None = None,
) -> ProbeEvalResult:
    """Train or reuse MLP probes and solve coupled monotonic boundaries."""
    lbl = sl_data.labels
    z_coords = sl_data.z_coords
    true_iface = extract_true_interfaces(lbl) if lbl is not None else None
    true_tops = (
        extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
    )
    has_binary = (
        true_iface is not None and np.unique(true_iface).size == _MIN_BINARY_CLASSES
    )

    probes: dict[str, Pipeline] = {} if probes_cache is None else probes_cache
    rows: list[RowDict] = []
    out_art: dict[str, ProbeMethodArt] = {}

    for m_name, z_tr in train_embeds.items():
        if m_name not in probes:
            probes[m_name] = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=_PROBE_HIDDEN_LAYERS,
                    max_iter=_PROBE_MAX_ITER,
                    random_state=cfg.seed,
                ),
            ).fit(z_tr, train_labels)
        art = _eval_single_probe(
            probes[m_name],
            sl_data,
            unsup_artifacts[m_name].rgt,
            method=m_name,
            unit_ids=unit_ids,
        )
        out_art[m_name] = art
        if lbl is None:
            continue

        iface_auc = (
            float(roc_auc_score(true_iface.ravel(), art.p_iface.ravel()))
            if has_binary and true_iface is not None
            else float("nan")
        )
        iface_ap = (
            float(average_precision_score(true_iface.ravel(), art.p_iface.ravel()))
            if has_binary and true_iface is not None
            else float("nan")
        )
        for uid in unit_ids:
            if not np.isfinite(true_tops[uid]).any():
                continue
            ev: RowDict = dict(
                evaluate_horizon_curve(true_tops[uid], art.pred_tops[uid])
            )
            ev["slice"] = slice_key
            ev["method"] = m_name
            ev["horizon"] = f"Top_Unit_{uid}"
            ev["voxel_acc"] = round(art.voxel_acc, 4)
            ev["iface_roc_auc"] = round(iface_auc, 4)
            ev["iface_pr_auc"] = round(iface_ap, 4)
            rows.append(ev)

    return ProbeEvalResult(rows=rows, artifacts=out_art, probes=probes)
