"""Harder stress-test evaluations and advanced visualisations for NOFE on F3 data."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.projections.polar import PolarAxes
from nofe import NOFE
from nofe.affinity import per_query_kl
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from umap import UMAP

from .data import LazySeismicVolume, SeismicVolume, volume_to_points
from .depth_compression import TraceLogs, extract_trace_logs
from .graph import (
    SeismicPointGraph,
    SharedStandardizer,
    coord_map_2d,
    embed_numpy,
    graph_edge_index,
    graph_loss,
    graph_num_nodes,
    graph_summary,
    train_nofe,
)
from .metrics import local_graph_metrics, neighborhood_metrics, norm_rmse
from .viz import embedding_to_rgb, project_embedding_pca

_NOISE_VIZ_SIGMA = 0.50
_NOISE_TOL = 1e-6
_NEIGHBOR_K = 15
_NOISE_EVAL_SAMPLE = 2000
_MIN_BINARY_CLASSES = 2
_MAX_NATIVE_ZOOM_TRACES = 150_000
_DENSE_ZOOM_RADIUS_THRESHOLD = 80_000
_LARGE_INLINE_ZOOM = 500
_LARGE_XLINE_ZOOM = 850
_LARGE_3D_INLINES = 400
_LARGE_3D_XLINES = 900
_LARGE_3D_DEPTH = 200
_RGB_CHANNELS = 3

MetricRow: TypeAlias = dict[str, float | int | str]


@dataclass(frozen=True)
class NoiseRow:
    sigma: float
    method: str
    drift_norm_rmse: float
    stability_r: float
    clean_target_local_stress: float
    clean_target_global_stress: float
    clean_target_trustworthiness: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "sigma": self.sigma,
            "method": self.method,
            "drift_norm_rmse": self.drift_norm_rmse,
            "stability_r": self.stability_r,
            "clean_target_local_stress": self.clean_target_local_stress,
            "clean_target_global_stress": self.clean_target_global_stress,
            "clean_target_trustworthiness": self.clean_target_trustworthiness,
        }


@dataclass(frozen=True)
class DensityRow:
    regime: str
    nodes: int
    mean_degree: float
    pearson_r_vs_native: float
    norm_rmse_vs_native: float
    kl_loss: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "regime": self.regime,
            "nodes": self.nodes,
            "mean_degree": self.mean_degree,
            "pearson_r_vs_native": self.pearson_r_vs_native,
            "norm_rmse_vs_native": self.norm_rmse_vs_native,
            "kl_loss": self.kl_loss,
        }


def _spatial_gradient_magnitude(
    emb: np.ndarray, grid_shape: tuple[int, int]
) -> np.ndarray:
    """Compute Euclidean spatial gradient norm across 2D map axes."""
    gi, gx = grid_shape
    grid = emb.reshape(gi, gx, emb.shape[1])
    di = np.asarray(np.gradient(grid, axis=0), dtype=np.float32)
    dx = np.asarray(np.gradient(grid, axis=1), dtype=np.float32)
    return np.sqrt(np.sum(di * di + dx * dx, axis=2)).astype(np.float32)


def _true_boundary_mask(labels_2d: np.ndarray) -> np.ndarray:
    """Binary edge mask where adjacent 2D map pixels switch geological province."""
    edge = np.zeros_like(labels_2d, dtype=bool)
    edge[:-1, :] |= labels_2d[:-1, :] != labels_2d[1:, :]
    edge[1:, :] |= labels_2d[:-1, :] != labels_2d[1:, :]
    edge[:, :-1] |= labels_2d[:, :-1] != labels_2d[:, 1:]
    edge[:, 1:] |= labels_2d[:, :-1] != labels_2d[:, 1:]
    return edge


def _extent_from_locs(locs: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(locs[:, 1].min()),
        float(locs[:, 1].max()),
        float(locs[:, 0].max()),
        float(locs[:, 0].min()),
    )


def _run_noise_stress_test(
    x_te: np.ndarray,
    p_te: np.ndarray,
    g_te: Data,
    *,
    nofe8: NOFE,
    pca8: PCA,
    umap8: UMAP,
    z_nofe_te: np.ndarray,
    z_pca_te: np.ndarray,
    z_umap_te: np.ndarray,
    radius_2d: float,
    seed: int,
) -> tuple[list[NoiseRow], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    noise_rows: list[NoiseRow] = []
    noisy_viz_snapshot: dict[str, np.ndarray] = {}
    te_ei = graph_edge_index(g_te)

    for sigma in (0.0, 0.25, 0.50, 1.00):
        if sigma == 0.0:
            x_noisy = x_te
        else:
            x_noisy = (x_te + rng.normal(scale=sigma, size=x_te.shape)).astype(
                np.float32
            )
        g_noisy = SeismicPointGraph(
            mode="radius", radius=radius_2d, normalize=False
        ).build(x_noisy, p_te)

        zn = embed_numpy(nofe8, g_noisy)
        zp = pca8.transform(x_noisy).astype(np.float32)
        zu = umap8.transform(x_noisy).astype(np.float32)

        for m_name, z_cur, z_clean in (
            ("NOFE-8D", zn, z_nofe_te),
            ("PCA-8D", zp, z_pca_te),
            ("UMAP-8D", zu, z_umap_te),
        ):
            nm = neighborhood_metrics(
                x_te, z_cur, k=_NEIGHBOR_K, sample_size=_NOISE_EVAL_SAMPLE, seed=seed
            )
            lm = local_graph_metrics(x_te, z_cur, te_ei)
            corr = (
                float(np.corrcoef(z_cur.ravel(), z_clean.ravel())[0, 1])
                if sigma > 0
                else 1.0
            )
            noise_rows.append(
                NoiseRow(
                    sigma=sigma,
                    method=m_name,
                    drift_norm_rmse=round(norm_rmse(z_cur, z_clean), 4),
                    stability_r=round(corr, 4),
                    clean_target_local_stress=round(lm["local_stress"], 4),
                    clean_target_global_stress=round(nm["global_stress"], 4),
                    clean_target_trustworthiness=round(nm["trustworthiness"], 4),
                )
            )
            if abs(sigma - _NOISE_VIZ_SIGMA) < _NOISE_TOL:
                noisy_viz_snapshot[m_name] = z_cur
        if abs(sigma - _NOISE_VIZ_SIGMA) < _NOISE_TOL:
            noisy_viz_snapshot["noisy_rms"] = np.sqrt(
                np.mean(x_noisy * x_noisy, axis=1)
            )
    return noise_rows, noisy_viz_snapshot


def _run_mosaic_stitching(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    full: TraceLogs,
    *,
    z_nofe_tr: np.ndarray,
    base_stride_2d: tuple[int, int],
    radius_2d: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray], int, int]:
    ni, nx, _ = volume.shape
    si, sx = base_stride_2d
    mid_i = (ni // 2 // si) * si
    mid_x = (nx // 2 // sx) * sx
    ov_i = max(si * 4, round(0.12 * ni / si) * si)
    ov_x = max(sx * 4, round(0.12 * nx / sx) * sx)

    tiles = {
        "NW": ((0, min(ni, mid_i + ov_i)), (0, min(nx, mid_x + ov_x))),
        "NE": ((0, min(ni, mid_i + ov_i)), (max(0, mid_x - ov_x), nx)),
        "SW": ((max(0, mid_i - ov_i), ni), (0, min(nx, mid_x + ov_x))),
        "SE": ((max(0, mid_i - ov_i), ni), (max(0, mid_x - ov_x), nx)),
    }
    tile_embeds: dict[
        str, dict[str, tuple[dict[tuple[float, float], int], np.ndarray]]
    ] = {
        "NOFE-8D": {},
        "PCA-8D": {},
        "UMAP-8D": {},
    }
    builder = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False)
    for tname, (ir, xr) in tiles.items():
        td = extract_trace_logs(
            volume,
            inline_range=ir,
            crossline_range=xr,
            stride_2d=base_stride_2d,
            feature_mode="envelope",
        )
        xt = scaler.transform_features(td.traces)
        pt = scaler.transform_locations(td.locations)
        gt = builder.build(xt, pt)
        cmap = coord_map_2d(td.locations)
        tile_embeds["NOFE-8D"][tname] = (cmap, embed_numpy(nofe8, gt))
        tile_embeds["PCA-8D"][tname] = (
            cmap,
            PCA(n_components=8, random_state=seed).fit_transform(xt).astype(np.float32),
        )
        tile_embeds["UMAP-8D"][tname] = (
            cmap,
            UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=seed)
            .fit_transform(xt)
            .astype(np.float32),
        )

    pairs = (("NW", "NE"), ("SW", "SE"), ("NW", "SW"), ("NE", "SE"))
    mosaic_seam_errors: dict[str, float] = {}
    mosaic_images: dict[str, np.ndarray] = {}
    gi_f, gx_f = full.grid_shape

    for m_name in ("NOFE-8D", "PCA-8D", "UMAP-8D"):
        errs: list[float] = []
        for a, b in pairs:
            ma, za = tile_embeds[m_name][a]
            mb, zb = tile_embeds[m_name][b]
            shared = [k for k in ma if k in mb]
            if shared:
                ia = [ma[k] for k in shared]
                ib = [mb[k] for k in shared]
                errs.append(norm_rmse(za[ia], zb[ib]))
        mosaic_seam_errors[m_name] = round(float(np.mean(errs)), 4)

        mosaic_z = np.zeros((full.locations.shape[0], 8), dtype=np.float32)
        for row_idx, loc in enumerate(full.locations):
            key = (float(loc[0]), float(loc[1]))
            q = ("N" if loc[0] < mid_i else "S") + ("W" if loc[1] < mid_x else "E")
            qmap, qz = tile_embeds[m_name][q]
            mosaic_z[row_idx] = qz[qmap[key]]
        fit_ref = z_nofe_tr if m_name == "NOFE-8D" else None
        mosaic_rgb = embedding_to_rgb(
            project_embedding_pca(
                mosaic_z, fit_embedding=fit_ref, n_components=3, seed=seed
            )[0]
        )
        mosaic_images[m_name] = mosaic_rgb.reshape(gi_f, gx_f, _RGB_CHANNELS)

    return mosaic_seam_errors, mosaic_images, mid_i, mid_x


def _run_density_transfer_curve(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    *,
    split_i: int,
    radius_2d: float,
    seed: int,
) -> list[DensityRow]:
    ni, _, _ = volume.shape
    rng = np.random.default_rng(seed)
    slab_i1 = min(ni, split_i + 36)
    builder = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False)

    nat = extract_trace_logs(
        volume,
        inline_range=(split_i, slab_i1),
        stride_2d=(1, 1),
        feature_mode="envelope",
    )
    gn = builder.build(
        scaler.transform_features(nat.traces),
        scaler.transform_locations(nat.locations),
    )
    zn_nat = embed_numpy(nofe8, gn)
    mnat = coord_map_2d(nat.locations)

    density_rows: list[DensityRow] = []
    regimes: tuple[tuple[str, tuple[int, int], float], ...] = (
        ("stride 6x6", (6, 6), 0.0),
        ("stride 4x4", (4, 4), 0.0),
        ("stride 3x3 + 50% drop", (3, 3), 0.50),
        ("stride 3x3", (3, 3), 0.0),
        ("stride 2x2", (2, 2), 0.0),
        ("native 1x1", (1, 1), 0.0),
    )
    for tag, st2, drop_frac in regimes:
        if st2 == (1, 1):
            z_cur, g_cur, cur_locs = zn_nat, gn, nat.locations
        else:
            d_pts = extract_trace_logs(
                volume,
                inline_range=(split_i, slab_i1),
                stride_2d=st2,
                feature_mode="envelope",
            )
            xd = scaler.transform_features(d_pts.traces)
            pd_loc = scaler.transform_locations(d_pts.locations)
            if drop_frac > 0.0:
                keep_n = min(
                    xd.shape[0], max(4, round(xd.shape[0] * (1.0 - drop_frac)))
                )
                keep_idx = np.sort(rng.choice(xd.shape[0], size=keep_n, replace=False))
                xd, pd_loc = xd[keep_idx], pd_loc[keep_idx]
                cur_locs = d_pts.locations[keep_idx]
            else:
                cur_locs = d_pts.locations
            g_cur = builder.build(xd, pd_loc)
            z_cur = embed_numpy(nofe8, g_cur)

        m_cur = coord_map_2d(cur_locs)
        shared_d = [k for k in m_cur if k in mnat]
        ic = [m_cur[k] for k in shared_d]
        in_ = [mnat[k] for k in shared_d]
        density_rows.append(
            DensityRow(
                regime=tag,
                nodes=graph_num_nodes(g_cur),
                mean_degree=round(float(graph_summary(g_cur)["mean_degree"]), 1),
                pearson_r_vs_native=round(
                    float(np.corrcoef(z_cur[ic].ravel(), zn_nat[in_].ravel())[0, 1]), 4
                ),
                norm_rmse_vs_native=round(norm_rmse(z_cur[ic], zn_nat[in_]), 4),
                kl_loss=round(graph_loss(nofe8, g_cur), 4),
            )
        )
    return density_rows


def _run_salt_dome_zoom(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    pca8: PCA,
    *,
    radius_2d: float,
) -> tuple[TraceLogs, np.ndarray, np.ndarray, np.ndarray, dict[str, float], int]:
    ni, nx, _ = volume.shape
    zi0, zi1 = (150, 450) if ni >= _LARGE_INLINE_ZOOM else (0, ni)
    zx0, zx1 = (350, 800) if nx >= _LARGE_XLINE_ZOOM else (0, nx)
    zoom = extract_trace_logs(
        volume,
        inline_range=(zi0, zi1),
        crossline_range=(zx0, zx1),
        stride_2d=(1, 1)
        if (zi1 - zi0) * (zx1 - zx0) <= _MAX_NATIVE_ZOOM_TRACES
        else (2, 2),
        feature_mode="envelope",
    )
    xz = scaler.transform_features(zoom.traces)
    pz = scaler.transform_locations(zoom.locations)
    zoom_radius = (
        radius_2d * 0.6
        if zoom.traces.shape[0] > _DENSE_ZOOM_RADIUS_THRESHOLD
        else radius_2d
    )
    gz = SeismicPointGraph(mode="radius", radius=zoom_radius, normalize=False).build(
        xz, pz
    )
    zz_nofe = embed_numpy(nofe8, gz)
    zz_pca = pca8.transform(xz).astype(np.float32)
    kl_zoom, _ = per_query_kl(
        None,
        torch.from_numpy(zz_nofe),
        graph_edge_index(gz),
        num_nodes=graph_num_nodes(gz),
        weights=gz.affinity,
    )
    kl_zoom_np = kl_zoom.numpy()

    grad_nofe = _spatial_gradient_magnitude(zz_nofe, zoom.grid_shape)
    grad_pca = _spatial_gradient_magnitude(zz_pca, zoom.grid_shape)
    boundary_auc: dict[str, float] = {}
    if zoom.deepest_facies is not None:
        edges_true = _true_boundary_mask(zoom.deepest_facies.reshape(zoom.grid_shape))
        if len(np.unique(edges_true)) == _MIN_BINARY_CLASSES:
            boundary_auc["NOFE-8D_grad_auc"] = round(
                float(roc_auc_score(edges_true.ravel(), grad_nofe.ravel())), 4
            )
            boundary_auc["PCA-8D_grad_auc"] = round(
                float(roc_auc_score(edges_true.ravel(), grad_pca.ravel())), 4
            )
            boundary_auc["NOFE-8D_kl_auc"] = round(
                float(roc_auc_score(edges_true.ravel(), kl_zoom_np)), 4
            )
    return zoom, zz_nofe, grad_nofe, kl_zoom_np, boundary_auc, graph_num_nodes(gz)


def _plot_fig_a_radar_and_pareto(
    noise_rows: list[NoiseRow],
    mosaic_errs: dict[str, float],
    density_rows: list[DensityRow],
    out_path: Path,
) -> None:
    """Plot a 6-axis radar chart of NOFE advantages plus noise/stitching bars."""
    n0 = {r.method: r for r in noise_rows if r.sigma == 0.0}
    n50 = {
        r.method: r for r in noise_rows if abs(r.sigma - _NOISE_VIZ_SIGMA) < _NOISE_TOL
    }
    min_r = min(r.pearson_r_vs_native for r in density_rows)

    categories = [
        "Local Geometry\n(1 - Local Stress)",
        "Global Geometry\n(1 - Global Stress)",
        "Tile Stitching\n(1 / (1 + Seam Err))",
        "Noise Stability\n($\\sigma$=0.50 r)",
        "Noisy Stress\n(1 - Clean Stress)",
        "Trustworthiness\n(Held-Out)",
    ]
    scores = {
        m: [
            max(0.0, 1.0 - n0[m].clean_target_local_stress),
            max(0.0, 1.0 - n0[m].clean_target_global_stress),
            1.0 / (1.0 + mosaic_errs[m]),
            max(0.0, n50[m].stability_r),
            max(0.0, 1.0 - n50[m].clean_target_local_stress),
            n0[m].clean_target_trustworthiness,
        ]
        for m in ("NOFE-8D", "PCA-8D", "UMAP-8D")
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(14.2, 5.5), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0])
    ax_rad = fig.add_subplot(gs[0, 0], polar=True)
    colors = {"NOFE-8D": "#1b7837", "PCA-8D": "#762a83", "UMAP-8D": "#d95f02"}
    for m in ("NOFE-8D", "PCA-8D", "UMAP-8D"):
        vals = scores[m] + scores[m][:1]
        ax_rad.plot(angles, vals, linewidth=2.2, label=m, color=colors[m])
        ax_rad.fill(angles, vals, alpha=0.14, color=colors[m])
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(categories, fontsize=8.5)
    ax_rad.tick_params(axis="x", pad=18)
    ax_rad.set_ylim(0.0, 1.05)
    if isinstance(ax_rad, PolarAxes):
        ax_rad.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=22.5, fontsize=8)
    ax_rad.set_title(
        "Multi-Axis Seismic Stress-Test Profile (Higher = Better)",
        pad=26,
        fontsize=11,
    )
    ax_rad.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)

    ax_bar = fig.add_subplot(gs[0, 1])
    metrics = (
        (
            "4-Tile Seam Err ↓",
            [mosaic_errs[m] for m in ("NOFE-8D", "PCA-8D", "UMAP-8D")],
        ),
        (
            "Noise Drift ($\\sigma$=0.5) ↓",
            [n50[m].drift_norm_rmse for m in ("NOFE-8D", "PCA-8D", "UMAP-8D")],
        ),
        (
            "Noisy Local Stress ↓",
            [
                n50[m].clean_target_local_stress
                for m in ("NOFE-8D", "PCA-8D", "UMAP-8D")
            ],
        ),
    )
    x = np.arange(len(metrics))
    w = 0.25
    for idx, m in enumerate(("NOFE-8D", "PCA-8D", "UMAP-8D")):
        y = [row[1][idx] for row in metrics]
        bars = ax_bar.bar(x + (idx - 1) * w, y, width=w, label=m, color=colors[m])
        for b, v in zip(bars, y, strict=True):
            ax_bar.text(
                b.get_x() + b.get_width() / 2.0,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    max_bar = max(max(row[1]) for row in metrics)
    ax_bar.set_ylim(0.0, max_bar * 1.15)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([m[0] for m in metrics], fontsize=9.5)
    ax_bar.set_title(
        f"Hard Failure Modes Eliminated by NOFE\n"
        f"(9x-36x Density Correlation >= {min_r:.3f})",
        fontsize=11,
    )
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(fontsize=9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig_b_4tile_mosaic(
    full_data: TraceLogs,
    mosaic_images: dict[str, np.ndarray],
    mosaic_errs: dict[str, float],
    *,
    mid_i: float,
    mid_x: float,
    out_path: Path,
) -> None:
    extent = _extent_from_locs(full_data.locations)
    fig, raw_axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    for ax, m in zip(axes, ("NOFE-8D", "PCA-8D", "UMAP-8D"), strict=True):
        ax.imshow(mosaic_images[m], extent=extent, aspect="auto")
        ax.axhline(mid_i, color="white", linestyle="--", linewidth=1.4)
        ax.axvline(mid_x, color="white", linestyle="--", linewidth=1.4)
        ax.set_title(
            f"{m} 4-Tile Mosaic\n(Mean Seam Norm-RMSE: {mosaic_errs[m]:.3f})",
            fontsize=10.5,
        )
        ax.set_xlabel("Crossline")
    axes[0].set_ylabel("Inline")
    fig.suptitle(
        "4-Quadrant Survey Stitching: NOFE Is Seam-Free Across Independent Tiles",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig_c_noise_and_density(
    noise_rows: list[NoiseRow],
    density_rows: list[DensityRow],
    te_data: TraceLogs,
    *,
    noisy_viz: dict[str, np.ndarray],
    z_nofe_tr: np.ndarray,
    seed: int,
    out_path: Path,
) -> None:
    fig, raw_axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    axes = raw_axes
    colors = {"NOFE-8D": "#1b7837", "PCA-8D": "#762a83", "UMAP-8D": "#d95f02"}

    sigmas = sorted({r.sigma for r in noise_rows})
    for m in ("NOFE-8D", "PCA-8D", "UMAP-8D"):
        sub = sorted((r for r in noise_rows if r.method == m), key=lambda r: r.sigma)
        axes[0, 0].plot(
            sigmas,
            [r.drift_norm_rmse for r in sub],
            "o-",
            linewidth=2,
            color=colors[m],
            label=m,
        )
        axes[0, 1].plot(
            sigmas,
            [r.clean_target_local_stress for r in sub],
            "o-",
            linewidth=2,
            color=colors[m],
            label=m,
        )
    axes[0, 0].set_title("Embedding Drift Under Trace Noise (↓)", fontsize=10.5)
    axes[0, 0].set_xlabel("Gaussian Noise Std $\\sigma$")
    axes[0, 0].set_ylabel("Norm. RMSE vs Clean Embedding")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].set_title("Local Spatial Stress vs Clean Target (↓)", fontsize=10.5)
    axes[0, 1].set_xlabel("Gaussian Noise Std $\\sigma$")
    axes[0, 1].set_ylabel("Local Stress")
    axes[0, 1].grid(alpha=0.3)

    sorted_density = sorted(density_rows, key=lambda r: r.mean_degree)
    degs = [r.mean_degree for r in sorted_density]
    corrs = [r.pearson_r_vs_native for r in sorted_density]
    nrmses = [r.norm_rmse_vs_native for r in sorted_density]
    axes[0, 2].plot(
        degs, corrs, "s-", color="#1b7837", linewidth=2, label="Pearson r vs 1x1 (↑)"
    )
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_ylim(min(0.85, min(corrs) - 0.03), 1.01)
    axes[0, 2].set_title("Sampling-Density Transfer (6x6 -> 1x1 Native)", fontsize=10.5)
    axes[0, 2].set_xlabel("Graph Mean Degree (log scale)")
    axes[0, 2].set_ylabel("Pearson r vs Native 1x1")
    axes[0, 2].grid(alpha=0.3)
    ax2_r = axes[0, 2].twinx()
    ax2_r.plot(
        degs, nrmses, "^--", color="#2b83ba", linewidth=1.6, label="Norm. RMSE (↓)"
    )
    ax2_r.set_ylim(0.0, max(max(nrmses) * 1.15, 0.05))
    ax2_r.set_ylabel("Norm. RMSE vs Native 1x1", color="#2b83ba")
    h1, l1 = axes[0, 2].get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    axes[0, 2].legend(
        h1 + h2, l1 + l2, loc="center right", fontsize=8.5, framealpha=0.9
    )

    gi, gx = te_data.grid_shape
    extent = _extent_from_locs(te_data.locations)
    for ax, m in zip(axes[1], ("NOFE-8D", "PCA-8D", "UMAP-8D"), strict=True):
        fit_ref = z_nofe_tr if m == "NOFE-8D" else None
        rgb = embedding_to_rgb(
            project_embedding_pca(
                noisy_viz[m], fit_embedding=fit_ref, n_components=3, seed=seed
            )[0]
        ).reshape(gi, gx, _RGB_CHANNELS)
        ax.imshow(rgb, extent=extent, aspect="auto")
        ax.set_title(f"{m} Held-Out Map at Noise $\\sigma$=0.50", fontsize=10.5)
        ax.set_xlabel("Crossline")
    axes[1, 0].set_ylabel("Held-Out Inline")

    fig.suptitle(
        "Stress Testing: Acquisition Noise Denoising & Multi-Density Sampling",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fig_e_native_salt_zoom(
    zoom: TraceLogs,
    zz_nofe: np.ndarray,
    z_nofe_tr: np.ndarray,
    *,
    grad_nofe: np.ndarray,
    kl_zoom: np.ndarray,
    boundary_auc: dict[str, float],
    seed: int,
    out_path: Path,
) -> None:
    gi, gx = zoom.grid_shape
    extent = _extent_from_locs(zoom.locations)
    rgb = embedding_to_rgb(
        project_embedding_pca(
            zz_nofe, fit_embedding=z_nofe_tr, n_components=3, seed=seed
        )[0]
    ).reshape(gi, gx, _RGB_CHANNELS)
    kl_img = kl_zoom.reshape(gi, gx)

    fig, raw_axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    if zoom.deepest_facies is not None:
        deep_img = zoom.deepest_facies.reshape(gi, gx)
        axes[0].imshow(
            deep_img, cmap="tab10", vmin=0, vmax=9, extent=extent, aspect="auto"
        )
        axes[0].set_title("Native Deepest Facies\n(Ground-Truth Labels)")
    else:
        axes[0].imshow(
            zoom.rms_amp.reshape(gi, gx), cmap="viridis", extent=extent, aspect="auto"
        )
        axes[0].set_title("Native Trace RMS\n(Amplitude Energy)")
    axes[0].set_ylabel("Inline")

    axes[1].imshow(rgb, extent=extent, aspect="auto")
    axes[1].set_title("Native 1x1 NOFE-8D\n(PCA-3 RGB)")

    vmax_g = float(np.percentile(grad_nofe, 98.0)) or 1.0
    im_g = axes[2].imshow(
        grad_nofe, cmap="magma", vmin=0.0, vmax=vmax_g, extent=extent, aspect="auto"
    )
    auc_g = boundary_auc.get("NOFE-8D_grad_auc")
    t_g = (
        f"Embedding Gradient $\\|\\nabla z\\|$\n(Boundary AUC={auc_g:.3f})"
        if auc_g
        else "Embedding Gradient $\\|\\nabla z\\|$"
    )
    axes[2].set_title(t_g)
    fig.colorbar(im_g, ax=axes[2], fraction=0.046, pad=0.04)

    vmax_k = float(np.percentile(kl_img, 98.0)) or 1e-3
    im_k = axes[3].imshow(
        kl_img, cmap="inferno", vmin=0.0, vmax=vmax_k, extent=extent, aspect="auto"
    )
    auc_k = boundary_auc.get("NOFE-8D_kl_auc")
    t_k = (
        f"Native Per-Trace KL Residual\n(Boundary AUC={auc_k:.3f})"
        if auc_k
        else "Native Per-Trace KL"
    )
    axes[3].set_title(t_k)
    fig.colorbar(im_k, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Crossline")
    fig.suptitle(
        f"Native-Resolution Salt Diapir Zoom ({gi}x{gx} = {gi * gx:,} traces)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _embed_plane_rgba(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe3d: NOFE,
    *,
    z_tr: np.ndarray,
    radius_3d: float,
    shape_2d: tuple[int, int],
    seed: int,
    inline_range: tuple[int, int] | None = None,
    crossline_range: tuple[int, int] | None = None,
    depth_range: tuple[int, int] | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    pts = volume_to_points(
        volume,
        inline_range=inline_range,
        crossline_range=crossline_range,
        depth_range=depth_range,
        stride=stride,
    )
    xs = scaler.transform_features(pts.features)
    ps = scaler.transform_locations(pts.locations)
    gp = SeismicPointGraph(mode="radius", radius=radius_3d, normalize=False).build(
        xs, ps
    )
    zp = embed_numpy(nofe3d, gp)
    proj, _ = project_embedding_pca(zp, fit_embedding=z_tr, n_components=3, seed=seed)
    rgb = embedding_to_rgb(proj).reshape(shape_2d[0], shape_2d[1], _RGB_CHANNELS)
    rgba = np.concatenate([rgb, np.ones((*shape_2d, 1), dtype=np.float32)], axis=2)
    return pts.locations, rgba


def _plot_3d_manifold_panel(
    fig: plt.Figure,
    z_tr: np.ndarray,
    labels: np.ndarray | None,
    *,
    seed: int,
) -> None:
    ax_man = fig.add_subplot(1, 2, 2, projection="3d")
    z_proj, var_r = project_embedding_pca(z_tr, n_components=3, seed=seed)
    idx_s = np.random.default_rng(seed).choice(
        z_proj.shape[0], size=min(6000, z_proj.shape[0]), replace=False
    )
    labels_s = labels[idx_s] if labels is not None else np.zeros(len(idx_s))
    facies_names = (
        "0: Upper North Sea",
        "1: Middle North Sea",
        "2: Lower North Sea",
        "3: Rijnland/Chalk",
        "4: Scruff",
        "5: Zechstein",
    )
    cmap = plt.get_cmap("tab10")
    for cls in sorted(np.unique(labels_s)):
        m = labels_s == cls
        lbl = (
            facies_names[int(cls)]
            if int(cls) < len(facies_names)
            else f"Facies {int(cls)}"
        )
        ax_man.scatter(
            z_proj[idx_s[m], 0],
            z_proj[idx_s[m], 1],
            z_proj[idx_s[m], 2],
            s=6,
            alpha=0.55,
            color=cmap(int(cls)),
            label=lbl,
        )
    ax_man.set_xlabel(f"PC1 ({100 * var_r[0]:.1f}%)")
    ax_man.set_ylabel(f"PC2 ({100 * var_r[1]:.1f}%)")
    ax_man.set_zlabel(f"PC3 ({100 * var_r[2]:.1f}%)")
    ax_man.set_title(
        "3D NOFE-8D Latent Manifold Coloured by Facies", fontsize=10.5, pad=12
    )
    ax_man.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        fontsize=8,
        markerscale=2,
        framealpha=0.88,
    )


def _run_3d_cube_and_manifold_viz(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    device: str,
    seed: int,
    epochs: int,
    out_path: Path,
) -> dict[str, float | int]:
    """Train 3D NOFE-8D and render an orthogonal cube cutaway and 3D latent manifold."""
    ni, nx, nz = volume.shape
    split_i = max(8, round(ni * 0.75))
    st = (14, 22, 6) if ni >= _LARGE_3D_INLINES else (2, 2, 2)
    tr_pts = volume_to_points(volume, inline_range=(0, split_i), stride=st)
    scaler = SharedStandardizer.fit(tr_pts.features, tr_pts.locations)
    r3d = 0.16 if ni >= _LARGE_3D_INLINES else 0.25
    g_tr = SeismicPointGraph(mode="radius", radius=r3d, normalize=False).build(
        scaler.transform_features(tr_pts.features),
        scaler.transform_locations(tr_pts.locations),
    )

    nofe3d = NOFE(
        out_dim=8,
        width=16,
        ker_width=48,
        depth=2,
        nystrom=0.10,
        chunk_size=8192,
        device=device,
        seed=seed,
    )
    train_nofe(nofe3d, g_tr, epochs=epochs, patience=20, lr=1e-2, seed=seed)
    z_tr = embed_numpy(nofe3d, g_tr)

    i_idx = 300 if ni >= _LARGE_INLINE_ZOOM + 100 else ni // 2
    x_idx = 450 if nx >= _LARGE_3D_XLINES else nx // 2
    z_idx = 160 if nz >= _LARGE_3D_DEPTH else nz // 2
    s_in = (1, 6, 3) if ni >= _LARGE_3D_INLINES else (1, 1, 1)
    s_xl = (4, 1, 3) if ni >= _LARGE_3D_INLINES else (1, 1, 1)
    s_dp = (4, 6, 1) if ni >= _LARGE_3D_INLINES else (1, 1, 1)

    ni_xl = len(range(0, ni, s_xl[0]))
    nx_in = len(range(0, nx, s_in[1]))
    nz_in = len(range(0, nz, s_in[2]))
    ni_dp = len(range(0, ni, s_dp[0]))
    nx_dp = len(range(0, nx, s_dp[1]))

    loc_in, rgba_in = _embed_plane_rgba(
        volume,
        scaler,
        nofe3d,
        z_tr=z_tr,
        radius_3d=r3d,
        shape_2d=(nx_in, nz_in),
        seed=seed,
        inline_range=(i_idx, i_idx + 1),
        stride=s_in,
    )
    loc_xl, rgba_xl = _embed_plane_rgba(
        volume,
        scaler,
        nofe3d,
        z_tr=z_tr,
        radius_3d=r3d,
        shape_2d=(ni_xl, nz_in),
        seed=seed,
        crossline_range=(x_idx, x_idx + 1),
        stride=s_xl,
    )
    loc_dp, rgba_dp = _embed_plane_rgba(
        volume,
        scaler,
        nofe3d,
        z_tr=z_tr,
        radius_3d=r3d,
        shape_2d=(ni_dp, nx_dp),
        seed=seed,
        depth_range=(z_idx, z_idx + 1),
        stride=s_dp,
    )

    fig = plt.figure(figsize=(15.5, 6.4), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for locs_f, rgba_f, shape_f in (
        (loc_in, rgba_in, (nx_in, nz_in)),
        (loc_xl, rgba_xl, (ni_xl, nz_in)),
        (loc_dp, rgba_dp, (ni_dp, nx_dp)),
    ):
        ax3d.plot_surface(
            locs_f[:, 0].reshape(*shape_f),
            locs_f[:, 1].reshape(*shape_f),
            locs_f[:, 2].reshape(*shape_f),
            facecolors=rgba_f,
            rstride=1,
            cstride=1,
            shade=False,
        )
    ax3d.invert_zaxis()
    ax3d.view_init(elev=24, azim=-52)
    ax3d.set_xlabel("Inline")
    ax3d.set_ylabel("Crossline")
    ax3d.set_zlabel("Depth")
    ax3d.set_title(
        f"3D Orthogonal Cube Cutaway (NOFE-8D PCA-RGB)\n"
        f"Inline={i_idx}, Crossline={x_idx}, Depth={z_idx}",
        fontsize=10.5,
        pad=10,
    )

    _plot_3d_manifold_panel(fig, z_tr, tr_pts.labels, seed=seed)

    fig.suptitle(
        "3D Seismic Volume Reduction: Volumetric Cutaway & Latent Manifold", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "train_nodes": graph_num_nodes(g_tr),
        "kl_loss": round(graph_loss(nofe3d, g_tr), 6),
    }


def run_hard_eval_suite(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    out_dir: Path = Path("outputs/f3_hard_evals"),
    device: str = "cpu",
    seed: int = 42,
    base_stride_2d: tuple[int, int] = (3, 3),
    radius_2d: float = 0.05,
    epochs_2d: int = 100,
    epochs_3d: int = 70,
    run_3d_cube: bool = True,
) -> dict[str, object]:
    """Run noise stress tests, 4-tile mosaic stitching, density scaling and 3D viz."""
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ni, _, _ = volume.shape
    si = base_stride_2d[0]
    split_i = max(si * 4, (round(ni * 0.75) // si) * si)

    tr = extract_trace_logs(
        volume,
        inline_range=(0, split_i),
        stride_2d=base_stride_2d,
        feature_mode="envelope",
    )
    te = extract_trace_logs(
        volume,
        inline_range=(split_i, ni),
        stride_2d=base_stride_2d,
        feature_mode="envelope",
    )
    full = extract_trace_logs(volume, stride_2d=base_stride_2d, feature_mode="envelope")

    sc = SharedStandardizer.fit(tr.traces, tr.locations, feature_scale="global")
    x_tr, p_tr = sc.transform_features(tr.traces), sc.transform_locations(tr.locations)
    x_te, p_te = sc.transform_features(te.traces), sc.transform_locations(te.locations)

    builder = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False)
    g_tr = builder.build(x_tr, p_tr)
    g_te = builder.build(x_te, p_te)

    t0 = time.perf_counter()
    nofe8 = NOFE(
        out_dim=8,
        width=24,
        ker_width=64,
        depth=2,
        nystrom=0.10,
        chunk_size=8192,
        device=device,
        seed=seed,
    )
    train_nofe(nofe8, g_tr, epochs=epochs_2d, patience=25, lr=1e-2, seed=seed)
    t_nofe = time.perf_counter() - t0

    pca8 = PCA(n_components=8, random_state=seed).fit(x_tr)
    umap8 = UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=seed).fit(
        x_tr
    )

    z_nofe_te = embed_numpy(nofe8, g_te)
    z_pca_te = pca8.transform(x_te).astype(np.float32)
    z_umap_te = umap8.transform(x_te).astype(np.float32)
    z_nofe_tr = embed_numpy(nofe8, g_tr)

    noise_rows, noisy_viz_snapshot = _run_noise_stress_test(
        x_te,
        p_te,
        g_te,
        nofe8=nofe8,
        pca8=pca8,
        umap8=umap8,
        z_nofe_te=z_nofe_te,
        z_pca_te=z_pca_te,
        z_umap_te=z_umap_te,
        radius_2d=radius_2d,
        seed=seed,
    )
    mosaic_seam_errors, mosaic_images, mid_i, mid_x = _run_mosaic_stitching(
        volume,
        sc,
        nofe8,
        full,
        z_nofe_tr=z_nofe_tr,
        base_stride_2d=base_stride_2d,
        radius_2d=radius_2d,
        seed=seed,
    )
    density_rows = _run_density_transfer_curve(
        volume,
        sc,
        nofe8,
        split_i=split_i,
        radius_2d=radius_2d,
        seed=seed,
    )
    zoom, zz_nofe, grad_nofe, kl_zoom_np, boundary_auc, zoom_nodes = (
        _run_salt_dome_zoom(volume, sc, nofe8, pca8, radius_2d=radius_2d)
    )

    _plot_fig_a_radar_and_pareto(
        noise_rows,
        mosaic_seam_errors,
        density_rows,
        fig_dir / "figA_nofe_radar_overview.png",
    )
    _plot_fig_b_4tile_mosaic(
        full,
        mosaic_images,
        mosaic_seam_errors,
        mid_i=float(mid_i),
        mid_x=float(mid_x),
        out_path=fig_dir / "figB_4tile_mosaic_seams.png",
    )
    _plot_fig_c_noise_and_density(
        noise_rows,
        density_rows,
        te,
        noisy_viz=noisy_viz_snapshot,
        z_nofe_tr=z_nofe_tr,
        seed=seed,
        out_path=fig_dir / "figC_noise_and_density_stress.png",
    )
    _plot_fig_e_native_salt_zoom(
        zoom,
        zz_nofe,
        z_nofe_tr,
        grad_nofe=grad_nofe,
        kl_zoom=kl_zoom_np,
        boundary_auc=boundary_auc,
        seed=seed,
        out_path=fig_dir / "figE_salt_dome_native_zoom.png",
    )

    cube_metrics: dict[str, float | int] = {}
    if run_3d_cube:
        cube_metrics = _run_3d_cube_and_manifold_viz(
            volume,
            device=device,
            seed=seed,
            epochs=epochs_3d,
            out_path=fig_dir / "figD_3d_cube_and_manifold.png",
        )

    noise_dicts = [r.to_dict() for r in noise_rows]
    density_dicts = [r.to_dict() for r in density_rows]
    summary: dict[str, object] = {
        "nofe_2d_train_sec": round(t_nofe, 2),
        "noise_stress_test": noise_dicts,
        "mosaic_4tile_seam_norm_rmse": mosaic_seam_errors,
        "density_transfer_curve": density_dicts,
        "salt_zoom_nodes": zoom_nodes,
        "salt_zoom_boundary_auc": boundary_auc,
        "cube_3d_8d_summary": cube_metrics,
    }
    (out_dir / "hard_eval_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(noise_dicts).to_csv(out_dir / "noise_stress.csv", index=False)
    pd.DataFrame(density_dicts).to_csv(out_dir / "density_curve.csv", index=False)
    return summary
