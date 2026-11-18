"""End-to-end horizon boundary reconstruction pipeline and visualisations."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from nofe import NOFE
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    LazySeismicVolume,
    SeismicVolume,
    extract_seismic_attributes,
    volume_to_points,
)
from .graph import (
    SeismicPointGraph,
    SharedStandardizer,
    coord_map_2d,
    embed_numpy,
    graph_summary,
    train_nofe,
)
from .horizons import (
    HorizonConfig,
    coupled_monotonic_boundaries_2d,
    discover_horizons_from_rgt,
    estimate_structure_tensor_2d,
    evaluate_boundary_map,
    evaluate_horizon_curve,
    extract_boundary_ridges,
    extract_horizon_iso_rgt,
    extract_surface_iso_rgt_3d,
    extract_true_interfaces,
    extract_unit_top_depths,
    multiscale_boundary_contrast,
    rgt_aligned_smooth,
    solve_rgt_2d,
    solve_rgt_3d,
    spatial_gradient_magnitude,
)
from .metrics import norm_rmse
from .viz import embedding_to_rgb, project_embedding_pca

RowDict: TypeAlias = dict[str, float | int | str | None]
Extent2D: TypeAlias = tuple[float, float, float, float]

_MIN_BINARY_CLASSES = 2
_LARGE_INLINE_THRESHOLD = 600
_LARGE_CROSSLINE_THRESHOLD = 900
_LARGE_SLAB_THRESHOLD = 400
_NOISE_HALO = 5


def _slice_extent(u_coords: np.ndarray, z_coords: np.ndarray) -> Extent2D:
    return (
        float(u_coords.min()),
        float(u_coords.max()),
        float(z_coords.max()),
        float(z_coords.min()),
    )


def _embed_slice_native_z(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe_model: NOFE,
    pca_model: PCA,
    *,
    axis: str,
    index: int,
    lateral_stride: int,
    radius: float,
) -> dict[str, Any]:
    """Embed a 2D seismic section at native vertical resolution (``sz = 1``)."""
    ni, nx, nz = volume.shape
    lat = max(1, int(lateral_stride))
    if axis == "inline":
        pts = volume_to_points(
            volume, inline_range=(index, index + 1), stride=(1, lat, 1)
        )
        u_coords = np.arange(0, nx, lat, dtype=np.float32)
        xlabel = "Crossline"
    elif axis == "crossline":
        pts = volume_to_points(
            volume, crossline_range=(index, index + 1), stride=(lat, 1, 1)
        )
        u_coords = np.arange(0, ni, lat, dtype=np.float32)
        xlabel = "Inline"
    else:
        raise ValueError(f"axis must be 'inline' or 'crossline', got {axis!r}")

    z_coords = np.arange(nz, dtype=np.float32)
    nu = int(u_coords.size)
    xs = scaler.transform_features(pts.features)
    ps = scaler.transform_locations(pts.locations)
    graph = SeismicPointGraph(mode="radius", radius=radius, normalize=False).build(
        xs, ps
    )
    z_nofe = embed_numpy(nofe_model, graph).reshape(nu, nz, -1)
    z_pca = pca_model.transform(xs).astype(np.float32).reshape(nu, nz, -1)

    return {
        "u_coords": u_coords,
        "z_coords": z_coords,
        "xlabel": xlabel,
        "amplitude": pts.amplitude.reshape(nu, nz),
        "labels": pts.labels.reshape(nu, nz) if pts.labels is not None else None,
        "locations": pts.locations,
        "x_raw": xs.reshape(nu, nz, -1),
        "nofe": z_nofe,
        "pca": z_pca,
        "graph": graph_summary(graph),
    }


def _seed_for_unit_top(
    labels_2d: np.ndarray,
    unit_id: int,
) -> tuple[int, int] | None:
    """Choose a median valid trace and depth for seeding ``labels >= unit_id``."""
    ge_mask = labels_2d >= unit_id
    valid_u = np.flatnonzero(ge_mask.any(axis=1))
    if valid_u.size == 0:
        return None
    seed_u = int(valid_u[valid_u.size // 2])
    first_z = int(np.argmax(ge_mask[seed_u]))
    return seed_u, max(0, first_z)


def _eval_unsupervised_and_rgt(
    sl_data: dict[str, Any],
    slice_key: str,
    cfg: HorizonConfig,
    *,
    unit_ids: Sequence[int],
) -> tuple[list[RowDict], list[RowDict], dict[str, Any]]:
    """Run unsupervised contrast, ridges, RGT tracking, and zero-seed discovery."""
    lbl = sl_data["labels"]
    z_coords = sl_data["z_coords"]
    true_iface = extract_true_interfaces(lbl) if lbl is not None else None
    true_tops = (
        extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
    )

    unsup_rows: list[RowDict] = []
    rgt_rows: list[RowDict] = []
    artifacts: dict[str, Any] = {}

    sources: list[tuple[str, np.ndarray]] = [
        ("NOFE-8D", sl_data["nofe"]),
        ("PCA-8D", sl_data["pca"]),
        ("Raw-11D", sl_data["x_raw"]),
        ("Amplitude-1D", sl_data["amplitude"][:, :, None]),
    ]

    for m_name, grid in sources:
        contrast = multiscale_boundary_contrast(grid, windows=cfg.contrast_windows)
        ridges = extract_boundary_ridges(contrast, nms_size=11, top_k=3)
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
            num_horizons=4,
            min_sep=14,
            z_coords=z_coords,
        )

        seeded_curves: dict[int, np.ndarray] = {}
        if lbl is not None:
            for uid in unit_ids:
                seed = _seed_for_unit_top(lbl, uid)
                if seed is None:
                    continue
                pred_curve = extract_horizon_iso_rgt(
                    rgt, seed[0], seed[1], z_coords=z_coords
                )
                seeded_curves[uid] = pred_curve
                ev: RowDict = dict(evaluate_horizon_curve(true_tops[uid], pred_curve))
                ev["slice"] = slice_key
                ev["method"] = m_name
                ev["horizon"] = f"Top_Unit_{uid}"
                ev["mode"] = "seeded_rgt"
                rgt_rows.append(ev)

        artifacts[m_name] = {
            "contrast": contrast,
            "ridges": ridges,
            "grad_mag": grad_mag,
            "dip_field": dip_field,
            "rgt": rgt,
            "discovered": discovered,
            "seeded_curves": seeded_curves,
        }

    return unsup_rows, rgt_rows, artifacts


def _eval_probe_horizons(
    train_embeds: dict[str, np.ndarray],
    train_labels: np.ndarray,
    sl_data: dict[str, Any],
    unsup_artifacts: dict[str, Any],
    *,
    slice_key: str,
    cfg: HorizonConfig,
    unit_ids: Sequence[int],
    probes_cache: dict[str, Pipeline] | None = None,
) -> tuple[list[RowDict], dict[str, Any], dict[str, Pipeline]]:
    """Train/apply MLP probe + RGT-aligned smoothing + monotonic DP boundaries."""
    lbl = sl_data["labels"]
    z_coords = sl_data["z_coords"]
    true_iface = extract_true_interfaces(lbl) if lbl is not None else None
    true_tops = (
        extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
    )

    probes: dict[str, Pipeline] = {} if probes_cache is None else probes_cache
    rows: list[RowDict] = []
    out_art: dict[str, Any] = {}

    for m_name, z_tr in train_embeds.items():
        if m_name not in probes:
            probes[m_name] = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=120,
                    random_state=cfg.seed,
                ),
            ).fit(z_tr, train_labels)
        clf = probes[m_name]
        classes = np.asarray(clf.classes_, dtype=np.int64)
        grid_key = (
            "nofe"
            if m_name == "NOFE-8D"
            else ("pca" if m_name == "PCA-8D" else "x_raw")
        )
        grid = sl_data[grid_key]
        nu, nz, _ = grid.shape
        probs = clf.predict_proba(grid.reshape(-1, grid.shape[2])).reshape(nu, nz, -1)
        logp_raw = np.log(np.clip(probs, 1e-6, 1.0)).astype(np.float32)

        rgt = unsup_artifacts[m_name]["rgt"]
        logp_sm = rgt_aligned_smooth(logp_raw, rgt, sigma_u=1.8, sigma_z=1.2)
        _, seg_pred = coupled_monotonic_boundaries_2d(
            logp_sm, classes, rgt, z_coords=z_coords, lam_lat=4.0, max_step=24
        )
        pred_tops = extract_unit_top_depths(seg_pred, unit_ids, z_coords)

        p_iface = 1.0 - np.sum(probs[:, :-1, :] * probs[:, 1:, :], axis=2)
        voxel_acc = float((seg_pred == lbl).mean()) if lbl is not None else float("nan")
        has_binary = (
            true_iface is not None and np.unique(true_iface).size == _MIN_BINARY_CLASSES
        )
        iface_auc = (
            float(roc_auc_score(true_iface.ravel(), p_iface.ravel()))
            if has_binary and true_iface is not None
            else float("nan")
        )
        iface_ap = (
            float(average_precision_score(true_iface.ravel(), p_iface.ravel()))
            if has_binary and true_iface is not None
            else float("nan")
        )

        if lbl is not None:
            for uid in unit_ids:
                if not np.isfinite(true_tops[uid]).any():
                    continue
                ev: RowDict = dict(
                    evaluate_horizon_curve(true_tops[uid], pred_tops[uid])
                )
                ev["slice"] = slice_key
                ev["method"] = m_name
                ev["horizon"] = f"Top_Unit_{uid}"
                ev["voxel_acc"] = round(voxel_acc, 4)
                ev["iface_roc_auc"] = round(iface_auc, 4)
                ev["iface_pr_auc"] = round(iface_ap, 4)
                rows.append(ev)

        out_art[m_name] = {
            "seg_pred": seg_pred,
            "pred_tops": pred_tops,
            "p_iface": p_iface,
            "voxel_acc": voxel_acc,
        }

    return rows, out_art, probes


def _evaluate_noise_stress(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    pca8: PCA,
    *,
    inline_idx: int,
    sl_clean: dict[str, Any],
    cfg: HorizonConfig,
    unit_ids: Sequence[int],
) -> tuple[list[RowDict], dict[str, dict[int, np.ndarray]]]:
    """Inject 3D Gaussian trace noise and evaluate RGT & boundary degradation."""
    ni, _, nz = volume.shape
    lat = cfg.slice_lateral_stride
    rng = np.random.default_rng(cfg.seed)

    i0 = max(0, inline_idx - _NOISE_HALO)
    i1 = min(ni, inline_idx + _NOISE_HALO + 1)
    center_offset = inline_idx - i0
    slab_clean = np.asarray(volume.amplitude[i0:i1, ::lat, :], dtype=np.float32)
    amp_scale = (
        volume.amp_scale
        if isinstance(volume, LazySeismicVolume)
        else (float(volume.amplitude.std()) or 1.0)
    )
    slab_noisy = slab_clean + rng.normal(
        scale=cfg.noise_sigma * amp_scale, size=slab_clean.shape
    ).astype(np.float32)
    feats_noisy, _ = extract_seismic_attributes(slab_noisy, scale=amp_scale)
    xs_noisy = scaler.transform_features(
        feats_noisy[center_offset].reshape(-1, feats_noisy.shape[-1])
    )
    ps_clean = scaler.transform_locations(sl_clean["locations"])
    g_noisy = SeismicPointGraph(
        mode="radius", radius=cfg.radius_slice, normalize=False
    ).build(xs_noisy, ps_clean)
    nu = int(sl_clean["u_coords"].size)
    z_nofe_noisy = embed_numpy(nofe8, g_noisy).reshape(nu, nz, -1)
    z_pca_noisy = pca8.transform(xs_noisy).astype(np.float32).reshape(nu, nz, -1)
    amp_noisy = slab_noisy[center_offset]

    lbl = sl_clean["labels"]
    z_coords = sl_clean["z_coords"]
    true_iface = extract_true_interfaces(lbl) if lbl is not None else None
    true_tops = (
        extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
    )

    stress_rows: list[RowDict] = []
    noisy_curves: dict[str, dict[int, np.ndarray]] = {}
    for m_name, grid in [
        ("NOFE-8D", z_nofe_noisy),
        ("PCA-8D", z_pca_noisy),
        ("Raw-11D", xs_noisy.reshape(nu, nz, -1)),
        ("Amplitude-1D", amp_noisy[:, :, None]),
    ]:
        contrast = multiscale_boundary_contrast(grid, windows=cfg.contrast_windows)
        ridges = extract_boundary_ridges(contrast, nms_size=11, top_k=3)
        b_ev = (
            evaluate_boundary_map(true_iface, contrast, ridges, z_coords=z_coords)
            if true_iface is not None
            else {}
        )
        dip_f = estimate_structure_tensor_2d(grid, max_slope=cfg.max_slope)
        rgt = solve_rgt_2d(dip_f, w_dip=cfg.rgt_w_dip, w_mono=cfg.rgt_w_mono)
        noisy_curves[m_name] = {}
        if lbl is None:
            continue
        for uid in unit_ids:
            seed = _seed_for_unit_top(lbl, uid)
            if seed is None:
                continue
            pred_c = extract_horizon_iso_rgt(rgt, seed[0], seed[1], z_coords=z_coords)
            noisy_curves[m_name][uid] = pred_c
            h_ev = evaluate_horizon_curve(true_tops[uid], pred_c)
            stress_rows.append(
                {
                    "test": f"noise_sigma_{cfg.noise_sigma}",
                    "method": m_name,
                    "horizon": f"Top_Unit_{uid}",
                    "mae": h_ev["mae"],
                    "medae": h_ev["medae"],
                    "within_3px": h_ev["within_3px"],
                    "boundary_roc_auc": b_ev.get("boundary_roc_auc", float("nan")),
                }
            )
    return stress_rows, noisy_curves


def _evaluate_tile_seam(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    *,
    inline_idx: int,
    sl_clean: dict[str, Any],
    cfg: HorizonConfig,
) -> dict[str, Any]:
    """Compare West/East overlapping tile seam consistency for NOFE vs per-tile PCA."""
    _, nx, nz = volume.shape
    lat = cfg.slice_lateral_stride
    mid_x = nx // 2
    ov_lo = max(0, (int(0.35 * nx) // lat) * lat)
    ov_hi = min(nx, max(ov_lo + lat * 2, int(0.65 * nx)))
    pts_w = volume_to_points(
        volume,
        inline_range=(inline_idx, inline_idx + 1),
        crossline_range=(0, ov_hi),
        stride=(1, lat, 1),
    )
    pts_e = volume_to_points(
        volume,
        inline_range=(inline_idx, inline_idx + 1),
        crossline_range=(ov_lo, nx),
        stride=(1, lat, 1),
    )
    xw = scaler.transform_features(pts_w.features)
    pw = scaler.transform_locations(pts_w.locations)
    xe = scaler.transform_features(pts_e.features)
    pe = scaler.transform_locations(pts_e.locations)

    gw = SeismicPointGraph(
        mode="radius", radius=cfg.radius_slice, normalize=False
    ).build(xw, pw)
    ge = SeismicPointGraph(
        mode="radius", radius=cfg.radius_slice, normalize=False
    ).build(xe, pe)
    zn_w = embed_numpy(nofe8, gw)
    zn_e = embed_numpy(nofe8, ge)
    zp_w = (
        PCA(n_components=cfg.out_dim, random_state=cfg.seed)
        .fit_transform(xw)
        .astype(np.float32)
    )
    zp_e = (
        PCA(n_components=cfg.out_dim, random_state=cfg.seed)
        .fit_transform(xe)
        .astype(np.float32)
    )

    mw = coord_map_2d(pts_w.locations[:, [1, 2]])
    me = coord_map_2d(pts_e.locations[:, [1, 2]])
    margin = min(max(2, lat * 2), max(1, (ov_hi - ov_lo) // 4))
    shared = [k for k in mw if k in me and (ov_lo + margin) <= k[0] <= (ov_hi - margin)]
    iw = [mw[k] for k in shared]
    ie = [me[k] for k in shared]
    nofe_seam = norm_rmse(zn_w[iw], zn_e[ie]) if shared else float("nan")
    pca_seam = norm_rmse(zp_w[iw], zp_e[ie]) if shared else float("nan")

    u_full = sl_clean["u_coords"]
    contrast_mosaic: dict[str, np.ndarray] = {}
    for m_name, zw_flat, ze_flat, locs_w, locs_e in [
        ("NOFE-8D", zn_w, zn_e, pts_w.locations, pts_e.locations),
        ("PCA-8D (Per-Tile)", zp_w, zp_e, pts_w.locations, pts_e.locations),
    ]:
        uw = np.unique(locs_w[:, 1]).size
        ue = np.unique(locs_e[:, 1]).size
        cw = multiscale_boundary_contrast(
            zw_flat.reshape(uw, nz, -1), windows=cfg.contrast_windows
        )
        ce = multiscale_boundary_contrast(
            ze_flat.reshape(ue, nz, -1), windows=cfg.contrast_windows
        )
        xw_u = np.unique(locs_w[:, 1])
        xe_u = np.unique(locs_e[:, 1])
        mos = np.zeros((u_full.size, nz - 1), dtype=np.float32)
        for idx_u, x_val in enumerate(u_full):
            if x_val < mid_x:
                mos[idx_u] = cw[int(np.argmin(np.abs(xw_u - x_val)))]
            else:
                mos[idx_u] = ce[int(np.argmin(np.abs(xe_u - x_val)))]
        contrast_mosaic[m_name] = mos

    return {
        "mid_x": float(mid_x),
        "shared_interior_points": len(shared),
        "nofe_seam_norm_rmse": round(float(nofe_seam), 4),
        "pca_tile_seam_norm_rmse": round(float(pca_seam), 4),
        "contrast_mosaic": contrast_mosaic,
    }


def _plot_fig1_unsupervised(
    sl_data: dict[str, Any],
    unsup_art: dict[str, Any],
    z_tr_nofe: np.ndarray,
    *,
    slice_name: str,
    out_path: Path,
    seed: int,
) -> None:
    u_coords = sl_data["u_coords"]
    z_coords = sl_data["z_coords"]
    extent = _slice_extent(u_coords, z_coords)
    amp = sl_data["amplitude"]
    lbl = sl_data["labels"]

    nofe_art = unsup_art["NOFE-8D"]
    contrast = nofe_art["contrast"]
    ridges = nofe_art["ridges"]
    discovered = nofe_art["discovered"]

    proj, _ = project_embedding_pca(
        sl_data["nofe"].reshape(-1, sl_data["nofe"].shape[2]),
        fit_embedding=z_tr_nofe,
        n_components=3,
        seed=seed,
    )
    rgb = embedding_to_rgb(proj).reshape(u_coords.size, z_coords.size, 3)

    n_panels = 5 if lbl is not None else 4
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.3 * n_panels, 4.5), constrained_layout=True
    )
    lim = float(np.percentile(np.abs(amp), 99.0)) or 1.0
    p = 0

    axes[p].imshow(
        amp.T, cmap="gray", vmin=-lim, vmax=lim, extent=extent, aspect="auto"
    )
    axes[p].set_title("Seismic Amplitude")
    axes[p].set_ylabel("Depth Sample")
    p += 1

    if lbl is not None:
        axes[p].imshow(
            lbl.T, cmap="tab10", vmin=0, vmax=9, extent=extent, aspect="auto"
        )
        axes[p].set_title("Ground-Truth Facies")
        p += 1

    axes[p].imshow(np.transpose(rgb, (1, 0, 2)), extent=extent, aspect="auto")
    axes[p].set_title("NOFE-8D (PCA-3 RGB)")
    p += 1

    vmax_c = float(np.percentile(contrast, 98.0)) or 1.0
    axes[p].imshow(
        contrast.T, cmap="magma", vmin=0.0, vmax=vmax_c, extent=extent, aspect="auto"
    )
    ru, rz = np.where(ridges)
    axes[p].scatter(
        u_coords[ru],
        0.5 * (z_coords[rz] + z_coords[rz + 1]),
        s=2.5,
        c="cyan",
        alpha=0.75,
        label="NMS Ridges",
    )
    axes[p].set_title("NOFE Boundary Contrast + Ridges")
    axes[p].legend(loc="upper right", fontsize=8, markerscale=2.5)
    p += 1

    axes[p].imshow(
        amp.T, cmap="gray", vmin=-lim, vmax=lim, extent=extent, aspect="auto"
    )
    colors = ["#00e5ff", "#ffeb3b", "#ff5252", "#69f0ae"]
    for idx, item in enumerate(discovered):
        axes[p].plot(
            u_coords,
            item["depths"],
            color=colors[idx % len(colors)],
            linewidth=1.8,
            label=f"RGT Horizon #{idx + 1}",
        )
    axes[p].set_title("Zero-Seed Auto-Discovered\nNOFE RGT Horizons")
    axes[p].legend(loc="lower left", fontsize=7.5)

    for ax in axes:
        ax.set_xlabel(sl_data["xlabel"])
    fig.suptitle(
        f"Unsupervised Boundary Reconstruction from NOFE-8D ({slice_name})", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_horizon_panel(
    ax: Axes,
    sl: dict[str, Any],
    curves: dict[int, np.ndarray],
    true_tops: dict[int, np.ndarray],
    *,
    title: str,
    colors: dict[int, str],
    show_legend: bool = False,
) -> None:
    u_coords = sl["u_coords"]
    z_coords = sl["z_coords"]
    extent = _slice_extent(u_coords, z_coords)
    amp = sl["amplitude"]
    lim = float(np.percentile(np.abs(amp), 99.0)) or 1.0
    ax.imshow(amp.T, cmap="gray", vmin=-lim, vmax=lim, extent=extent, aspect="auto")
    for uid, curve in curves.items():
        if not np.isfinite(curve).any():
            continue
        if uid in true_tops and np.isfinite(true_tops[uid]).any():
            ax.plot(
                u_coords,
                true_tops[uid],
                color="white",
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
            )
        ax.plot(
            u_coords,
            curve,
            color=colors.get(uid, "cyan"),
            linewidth=1.8,
            label=f"Unit {uid}",
        )
    ax.set_title(title)
    ax.set_xlabel(sl["xlabel"])
    if show_legend:
        ax.legend(loc="lower left", fontsize=7.5)


def _plot_fig2_tracking_comparison(
    sl_inline: dict[str, Any],
    sl_cross: dict[str, Any],
    *,
    unsup_in: dict[str, Any],
    unsup_xl: dict[str, Any],
    probe_in: dict[str, Any],
    probe_xl: dict[str, Any],
    unit_ids: Sequence[int],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)
    row_specs = [
        ("Held-Out Inline", sl_inline, unsup_in, probe_in),
        ("Crossline", sl_cross, unsup_xl, probe_xl),
    ]
    h_colors = {1: "#ffeb3b", 2: "#00e5ff", 3: "#ff5252", 4: "#e040fb", 5: "#69f0ae"}

    for r_idx, (r_title, sl, u_art, p_art) in enumerate(row_specs):
        lbl = sl["labels"]
        z_coords = sl["z_coords"]
        true_tops = (
            extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
        )
        _draw_horizon_panel(
            axes[r_idx, 0],
            sl,
            true_tops,
            {},
            title=f"{r_title}: True Boundaries",
            colors=h_colors,
            show_legend=(r_idx == 0),
        )
        axes[r_idx, 0].set_ylabel("Depth Sample")
        _draw_horizon_panel(
            axes[r_idx, 1],
            sl,
            u_art["NOFE-8D"]["seeded_curves"],
            true_tops,
            title=f"{r_title}: NOFE-8D Seeded RGT",
            colors=h_colors,
        )
        _draw_horizon_panel(
            axes[r_idx, 2],
            sl,
            p_art["NOFE-8D"]["pred_tops"],
            true_tops,
            title=f"{r_title}: NOFE-8D Probe + Monotonic DP",
            colors=h_colors,
        )
        _draw_horizon_panel(
            axes[r_idx, 3],
            sl,
            p_art["PCA-8D"]["pred_tops"],
            true_tops,
            title=f"{r_title}: PCA-8D Probe + Monotonic DP",
            colors=h_colors,
        )

    fig.suptitle(
        "Horizon Tracking & Boundary Reconstruction: True (dotted) vs. NOFE & PCA",
        fontsize=12.5,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig3_robustness(
    sl_inline: dict[str, Any],
    seam_info: dict[str, Any],
    stress_rows: list[RowDict],
    cfg: HorizonConfig,
    out_path: Path,
) -> None:
    u_coords = sl_inline["u_coords"]
    z_coords = sl_inline["z_coords"]
    extent = _slice_extent(u_coords, z_coords)
    mid_x = seam_info["mid_x"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    c_nofe = seam_info["contrast_mosaic"]["NOFE-8D"]
    vmax_n = float(np.percentile(c_nofe, 98.0)) or 1.0
    axes[0].imshow(
        c_nofe.T, cmap="magma", vmin=0.0, vmax=vmax_n, extent=extent, aspect="auto"
    )
    axes[0].axvline(mid_x, color="cyan", linestyle="--", linewidth=1.3)
    axes[0].set_title(
        f"NOFE-8D 2-Tile Boundary Contrast\n"
        f"(Seam NRMSE={seam_info['nofe_seam_norm_rmse']:.3f})"
    )
    axes[0].set_xlabel(sl_inline["xlabel"])
    axes[0].set_ylabel("Depth Sample")

    c_pca = seam_info["contrast_mosaic"]["PCA-8D (Per-Tile)"]
    vmax_p = float(np.percentile(c_pca, 98.0)) or 1.0
    axes[1].imshow(
        c_pca.T, cmap="magma", vmin=0.0, vmax=vmax_p, extent=extent, aspect="auto"
    )
    axes[1].axvline(mid_x, color="cyan", linestyle="--", linewidth=1.3)
    axes[1].set_title(
        f"Per-Tile PCA-8D Boundary Contrast\n"
        f"(Seam NRMSE={seam_info['pca_tile_seam_norm_rmse']:.3f})"
    )
    axes[1].set_xlabel(sl_inline["xlabel"])

    sub = [r for r in stress_rows if r["horizon"] == "Top_Unit_1"] or stress_rows[:4]
    methods = [str(r["method"]) for r in sub]
    maes = [float(r["mae"]) for r in sub if r["mae"] is not None]
    colors = ["#1b7837", "#762a83", "#d95f02", "#555555"][: len(methods)]
    bars = axes[2].bar(methods, maes, color=colors, width=0.55)
    axes[2].set_title(
        f"Noisy RGT Horizon Error (std={cfg.noise_sigma:.2f})\nLower MAE = Better"
    )
    axes[2].set_ylabel("Depth MAE (samples)")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", rotation=15)
    for b, v in zip(bars, maes, strict=True):
        axes[2].text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "Robustness of NOFE Boundary Reconstruction Across Tile Seams & Noise",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig4_3d_surfaces(
    i_coords: np.ndarray,
    x_coords: np.ndarray,
    true_surf: np.ndarray,
    pred_surf: np.ndarray,
    *,
    err_title: str,
    out_path: Path,
) -> None:
    ii, xx = np.meshgrid(i_coords, x_coords, indexing="ij")
    fig = plt.figure(figsize=(14.5, 4.8), constrained_layout=True)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(ii, xx, true_surf, cmap="viridis", linewidth=0, antialiased=True)
    ax1.invert_zaxis()
    ax1.set_title("True 3D Horizon Surface (Unit 1 Top)")
    ax1.set_xlabel("Inline")
    ax1.set_ylabel("Crossline")
    ax1.set_zlabel("Depth")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(ii, xx, pred_surf, cmap="viridis", linewidth=0, antialiased=True)
    ax2.invert_zaxis()
    ax2.set_title("NOFE-8D Reconstructed 3D Surface")
    ax2.set_xlabel("Inline")
    ax2.set_ylabel("Crossline")
    ax2.set_zlabel("Depth")

    ax3 = fig.add_subplot(1, 3, 3)
    diff = np.abs(pred_surf - true_surf)
    extent: Extent2D = (
        float(x_coords.min()),
        float(x_coords.max()),
        float(i_coords.max()),
        float(i_coords.min()),
    )
    finite_diff = diff[np.isfinite(diff)]
    vmax = (float(np.percentile(finite_diff, 96.0)) or 1.0) if finite_diff.size else 1.0
    im = ax3.imshow(
        diff, cmap="inferno", vmin=0.0, vmax=vmax, extent=extent, aspect="auto"
    )
    ax3.set_title(f"Absolute Depth Error |Δz|\n({err_title})")
    ax3.set_xlabel("Crossline")
    ax3.set_ylabel("Inline")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(
        "3D Held-Out Slab Boundary Surface Reconstruction from NOFE-8D", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _evaluate_3d_slab(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    *,
    split_i: int,
    unit_ids: Sequence[int],
    probes_cache: dict[str, Pipeline] | None,
    cfg: HorizonConfig,
    fig_dir: Path,
) -> dict[str, Any]:
    """Reconstruct and evaluate 3D horizon surfaces on the held-out 3D slab."""
    ni = volume.shape[0]
    slab_stride = (8, 16, 2) if ni >= _LARGE_SLAB_THRESHOLD else (2, 2, 1)
    slab_pts = volume_to_points(volume, inline_range=(split_i, ni), stride=slab_stride)
    i_vals = np.unique(slab_pts.locations[:, 0])
    x_vals = np.unique(slab_pts.locations[:, 1])
    z_vals = np.unique(slab_pts.locations[:, 2])
    s_ni, s_nx, s_nz = i_vals.size, x_vals.size, z_vals.size

    xs_slab = scaler.transform_features(slab_pts.features)
    ps_slab = scaler.transform_locations(slab_pts.locations)
    g_slab = SeismicPointGraph(
        mode="radius", radius=cfg.radius_train, normalize=False
    ).build(xs_slab, ps_slab)
    z_slab_nofe = embed_numpy(nofe8, g_slab).reshape(s_ni, s_nx, s_nz, -1)
    rgt_3d = solve_rgt_3d(
        z_slab_nofe, max_slope=cfg.max_slope, w_dip=cfg.rgt_w_dip, w_mono=cfg.rgt_w_mono
    )

    metrics: dict[str, Any] = {
        "slab_shape": [s_ni, s_nx, s_nz],
        "stride": list(slab_stride),
    }
    if slab_pts.labels is None or not unit_ids:
        return metrics

    lbl_3d = slab_pts.labels.reshape(s_ni, s_nx, s_nz)
    true_tops_3d = extract_unit_top_depths(lbl_3d, unit_ids, z_vals)
    uid_main = unit_ids[0]
    true_surf = true_tops_3d[uid_main]
    valid_ix = np.argwhere(np.isfinite(true_surf))
    if valid_ix.size == 0:
        return metrics

    si_mid, sx_mid = valid_ix[len(valid_ix) // 2]
    seed_z_phys = float(true_surf[si_mid, sx_mid])
    seed_z_idx = float(
        np.interp(seed_z_phys, z_vals, np.arange(s_nz, dtype=np.float32))
    )
    pred_surf_rgt = extract_surface_iso_rgt_3d(
        rgt_3d, int(si_mid), int(sx_mid), seed_z_idx, z_coords=z_vals
    )
    rgt_3d_ev = evaluate_horizon_curve(true_surf, pred_surf_rgt)
    metrics[f"rgt_3d_Top_Unit_{uid_main}"] = rgt_3d_ev

    pred_surf_for_fig = pred_surf_rgt
    err_title = f"3D RGT MAE={rgt_3d_ev['mae']:.2f}, MedAE={rgt_3d_ev['medae']:.2f}"
    if probes_cache is not None and "NOFE-8D" in probes_cache:
        clf = probes_cache["NOFE-8D"]
        probs_3d = clf.predict_proba(
            z_slab_nofe.reshape(-1, z_slab_nofe.shape[-1])
        ).reshape(s_ni, s_nx, s_nz, -1)
        logp_3d = np.log(np.clip(probs_3d, 1e-6, 1.0)).astype(np.float32)
        seg_3d = np.zeros((s_ni, s_nx, s_nz), dtype=np.int64)
        for idx_i in range(s_ni):
            logp_sl = rgt_aligned_smooth(
                logp_3d[idx_i], rgt_3d[idx_i], sigma_u=1.5, sigma_z=1.0
            )
            _, seg_3d[idx_i] = coupled_monotonic_boundaries_2d(
                logp_sl,
                np.asarray(clf.classes_, dtype=np.int64),
                rgt_3d[idx_i],
                z_coords=z_vals,
                lam_lat=4.0,
            )
        pred_tops_3d = extract_unit_top_depths(seg_3d, unit_ids, z_vals)
        dp_3d_ev = evaluate_horizon_curve(true_surf, pred_tops_3d[uid_main])
        metrics[f"probe_dp_3d_Top_Unit_{uid_main}"] = dp_3d_ev
        pred_surf_for_fig = pred_tops_3d[uid_main]
        err_title = (
            f"RGT MAE={rgt_3d_ev['mae']:.2f} | Probe-DP MAE={dp_3d_ev['mae']:.2f}"
        )

    _plot_fig4_3d_surfaces(
        i_vals,
        x_vals,
        true_surf,
        pred_surf_for_fig,
        err_title=err_title,
        out_path=fig_dir / "fig4_3d_horizon_surfaces.png",
    )
    return metrics


def _score_discovered_horizons(
    sl_inline: dict[str, Any],
    discovered: list[dict[str, Any]],
    unit_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if sl_inline["labels"] is None:
        return []
    true_tops_in = extract_unit_top_depths(
        sl_inline["labels"], unit_ids, sl_inline["z_coords"]
    )
    out: list[dict[str, Any]] = []
    for item in discovered:
        best_uid = None
        best_mae = float("inf")
        best_ev: dict[str, float] = {}
        for uid in unit_ids:
            if not np.isfinite(true_tops_in[uid]).any():
                continue
            ev = evaluate_horizon_curve(true_tops_in[uid], item["depths"])
            if np.isfinite(ev["mae"]) and ev["mae"] < best_mae:
                best_mae = ev["mae"]
                best_uid = uid
                best_ev = ev
        out.append(
            {
                "rank": item["rank"],
                "contrast_score": item["contrast_score"],
                "matched_unit": best_uid,
                "mae": best_ev.get("mae", float("nan")),
                "medae": best_ev.get("medae", float("nan")),
                "within_5px": best_ev.get("within_5px", 0.0),
            }
        )
    return out


def run_horizon_reconstruction(
    volume: SeismicVolume | LazySeismicVolume,
    cfg: HorizonConfig,
) -> dict[str, Any]:
    """Train 3D NOFE-8D on training inlines and evaluate boundary & horizon tracking."""
    out_dir = cfg.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ni, nx, _ = volume.shape
    split_i = max(8, round(ni * cfg.train_inline_frac))
    tr_pts = volume_to_points(
        volume, inline_range=(0, split_i), stride=cfg.train_stride
    )
    scaler = SharedStandardizer.fit(tr_pts.features, tr_pts.locations)
    x_tr = scaler.transform_features(tr_pts.features)
    p_tr = scaler.transform_locations(tr_pts.locations)
    g_tr = SeismicPointGraph(
        mode="radius", radius=cfg.radius_train, normalize=False
    ).build(x_tr, p_tr)

    t0 = time.perf_counter()
    nofe8 = NOFE(
        out_dim=cfg.out_dim,
        width=cfg.width,
        ker_width=cfg.ker_width,
        depth=cfg.depth,
        nystrom=cfg.nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
    )
    hist = train_nofe(
        nofe8, g_tr, epochs=cfg.epochs, patience=cfg.patience, lr=cfg.lr, seed=cfg.seed
    )
    t_train = time.perf_counter() - t0

    z_tr_nofe = embed_numpy(nofe8, g_tr)
    pca8 = PCA(n_components=min(cfg.out_dim, x_tr.shape[1]), random_state=cfg.seed).fit(
        x_tr
    )
    z_tr_pca = pca8.transform(x_tr).astype(np.float32)

    inline_idx = (
        540
        if ni >= _LARGE_INLINE_THRESHOLD
        else min(ni - 2, split_i + max(1, (ni - split_i) // 2))
    )
    cross_idx = 450 if nx >= _LARGE_CROSSLINE_THRESHOLD else nx // 2

    sl_inline = _embed_slice_native_z(
        volume,
        scaler,
        nofe8,
        pca8,
        axis="inline",
        index=inline_idx,
        lateral_stride=cfg.slice_lateral_stride,
        radius=cfg.radius_slice,
    )
    sl_cross = _embed_slice_native_z(
        volume,
        scaler,
        nofe8,
        pca8,
        axis="crossline",
        index=cross_idx,
        lateral_stride=cfg.slice_lateral_stride,
        radius=cfg.radius_slice,
    )

    present_classes = (
        sorted(int(c) for c in np.unique(tr_pts.labels))
        if tr_pts.labels is not None
        else [0, 1, 2]
    )
    unit_ids = [c for c in present_classes if c > present_classes[0]]

    unsup_in, rgt_in, art_in = _eval_unsupervised_and_rgt(
        sl_inline, f"inline_{inline_idx}", cfg, unit_ids=unit_ids
    )
    unsup_xl, rgt_xl, art_xl = _eval_unsupervised_and_rgt(
        sl_cross, f"crossline_{cross_idx}", cfg, unit_ids=unit_ids
    )
    unsup_rows = unsup_in + unsup_xl
    rgt_rows = rgt_in + rgt_xl
    discovered_eval = _score_discovered_horizons(
        sl_inline, art_in["NOFE-8D"]["discovered"], unit_ids
    )

    probe_rows: list[RowDict] = []
    probe_in: dict[str, Any] = {}
    probe_xl: dict[str, Any] = {}
    probes_cache: dict[str, Pipeline] | None = None
    if tr_pts.labels is not None:
        train_embeds = {"NOFE-8D": z_tr_nofe, "PCA-8D": z_tr_pca, "Raw-11D": x_tr}
        probe_rows_in, probe_in, probes_cache = _eval_probe_horizons(
            train_embeds,
            tr_pts.labels,
            sl_inline,
            art_in,
            slice_key=f"inline_{inline_idx}",
            cfg=cfg,
            unit_ids=unit_ids,
        )
        probe_rows_xl, probe_xl, _ = _eval_probe_horizons(
            train_embeds,
            tr_pts.labels,
            sl_cross,
            art_xl,
            slice_key=f"crossline_{cross_idx}",
            cfg=cfg,
            unit_ids=unit_ids,
            probes_cache=probes_cache,
        )
        probe_rows = probe_rows_in + probe_rows_xl

    stress_rows, _ = _evaluate_noise_stress(
        volume,
        scaler,
        nofe8,
        pca8,
        inline_idx=inline_idx,
        sl_clean=sl_inline,
        cfg=cfg,
        unit_ids=unit_ids,
    )
    seam_info = _evaluate_tile_seam(
        volume, scaler, nofe8, inline_idx=inline_idx, sl_clean=sl_inline, cfg=cfg
    )
    slab_3d_metrics = _evaluate_3d_slab(
        volume,
        scaler,
        nofe8,
        split_i=split_i,
        unit_ids=unit_ids,
        probes_cache=probes_cache,
        cfg=cfg,
        fig_dir=fig_dir,
    )

    _plot_fig1_unsupervised(
        sl_inline,
        art_in,
        z_tr_nofe,
        slice_name=f"Held-Out Inline {inline_idx}",
        out_path=fig_dir / "fig1_unsupervised_boundary_reconstruction.png",
        seed=cfg.seed,
    )
    if probe_in and probe_xl:
        _plot_fig2_tracking_comparison(
            sl_inline,
            sl_cross,
            unsup_in=art_in,
            unsup_xl=art_xl,
            probe_in=probe_in,
            probe_xl=probe_xl,
            unit_ids=unit_ids,
            out_path=fig_dir / "fig2_horizon_tracking_and_probe.png",
        )
    _plot_fig3_robustness(
        sl_inline,
        seam_info,
        stress_rows,
        cfg,
        fig_dir / "fig3_noise_and_seam_robustness.png",
    )

    for fname, rows in (
        ("unsupervised_boundaries.csv", unsup_rows),
        ("rgt_horizon_tracking.csv", rgt_rows),
        ("probe_horizon_reconstruction.csv", probe_rows),
        ("robustness_stress.csv", stress_rows),
    ):
        pd.DataFrame(rows).to_csv(out_dir / fname, index=False)

    summary = {
        "dataset": volume.name,
        "volume_shape": list(volume.shape),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        },
        "nofe_train_sec": round(t_train, 2),
        "history_last": hist[-1] if hist else None,
        "train_graph": graph_summary(g_tr),
        "unsupervised_boundaries": unsup_rows,
        "rgt_horizon_tracking": rgt_rows,
        "zero_seed_discovered_horizons": discovered_eval,
        "probe_horizon_reconstruction": probe_rows,
        "robustness_stress": stress_rows,
        "tile_seam_consistency": {
            "shared_interior_points": seam_info["shared_interior_points"],
            "nofe_seam_norm_rmse": seam_info["nofe_seam_norm_rmse"],
            "pca_tile_seam_norm_rmse": seam_info["pca_tile_seam_norm_rmse"],
        },
        "heldout_3d_slab": slab_3d_metrics,
    }
    (out_dir / "horizon_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
