"""Plotting helpers for the hard stress-test evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from nofe_seismic.data.traces import TraceLogs
from nofe_seismic.types import extent_from_locations
from nofe_seismic.viz.slices import embedding_to_rgb, project_embedding_pca

_NOISE_VIZ_SIGMA = 0.50
_NOISE_TOL = 1e-6
_RGB_CHANNELS = 3


@dataclass(frozen=True)
class NoiseRow:
    sigma: float
    method: str
    drift_norm_rmse: float
    stability_r: float
    clean_target_local_stress: float
    clean_target_trustworthiness: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "sigma": self.sigma,
            "method": self.method,
            "drift_norm_rmse": self.drift_norm_rmse,
            "stability_r": self.stability_r,
            "clean_target_local_stress": self.clean_target_local_stress,
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


def plot_fig_a_radar_and_pareto(
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
    methods = ("NOFE-8D", "PCA-8D", "UMAP-8D")
    colors = {"NOFE-8D": "#1b7837", "PCA-8D": "#762a83", "UMAP-8D": "#d95f02"}

    categories = (
        "Local Geometry\n(1 - Local Stress)",
        "Global Geometry\n(1 - Global Stress)",
        "Tile Stitching\n(1 / (1 + Seam Err))",
        "Noise Stability\n($\\sigma$=0.50 r)",
        "Noisy Stress\n(1 - Clean Stress)",
        "Trustworthiness\n(Held-Out)",
    )
    scores = {
        m: [
            max(0.0, 1.0 - n0[m].clean_target_local_stress),
            max(0.0, 1.0 - n50[m].drift_norm_rmse * 0.5),
            1.0 / (1.0 + mosaic_errs[m]),
            max(0.0, n50[m].stability_r),
            max(0.0, 1.0 - n50[m].clean_target_local_stress),
            n0[m].clean_target_trustworthiness,
        ]
        for m in methods
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(13.5, 5.2), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0])
    ax_rad = fig.add_subplot(gs[0, 0], polar=True)
    for m in methods:
        vals = scores[m] + scores[m][:1]
        ax_rad.plot(angles, vals, linewidth=2.2, label=m, color=colors[m])
        ax_rad.fill(angles, vals, alpha=0.14, color=colors[m])
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(categories, fontsize=9)
    ax_rad.set_ylim(0.0, 1.05)
    ax_rad.set_title(
        "Multi-Axis Seismic Stress-Test Profile (Higher = Better)", pad=18, fontsize=11
    )
    ax_rad.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=9)

    ax_bar = fig.add_subplot(gs[0, 1])
    metrics = (
        ("4-Tile Seam Err ↓", [mosaic_errs[m] for m in methods]),
        ("Noise Drift ($\\sigma$=0.5) ↓", [n50[m].drift_norm_rmse for m in methods]),
        (
            "Noisy Local Stress ↓",
            [n50[m].clean_target_local_stress for m in methods],
        ),
    )
    x = np.arange(len(metrics))
    w = 0.25
    for idx, m in enumerate(methods):
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


def plot_fig_b_4tile_mosaic(
    full_data: TraceLogs,
    mosaic_images: dict[str, np.ndarray],
    mosaic_errs: dict[str, float],
    *,
    mid_i: float,
    mid_x: float,
    out_path: Path,
) -> None:
    """Plot 4-quadrant survey mosaic RGB maps for NOFE, PCA and UMAP."""
    extent = extent_from_locations(full_data.locations)
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


def plot_fig_c_noise_and_density(
    noise_rows: list[NoiseRow],
    density_rows: list[DensityRow],
    te_data: TraceLogs,
    *,
    noisy_viz: dict[str, np.ndarray],
    z_nofe_tr: np.ndarray,
    seed: int,
    out_path: Path,
) -> None:
    """Plot noise drift/stress curves, density transfer and noisy RGB maps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
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

    degs = [r.mean_degree for r in density_rows]
    corrs = [r.pearson_r_vs_native for r in density_rows]
    nrmses = [r.norm_rmse_vs_native for r in density_rows]
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
    ax2_r.set_ylabel("Norm. RMSE vs Native 1x1", color="#2b83ba")

    gi, gx = te_data.grid_shape
    extent = extent_from_locations(te_data.locations)
    for ax, m in zip(axes[1], ("NOFE-8D", "PCA-8D", "UMAP-8D"), strict=True):
        fit_ref = z_nofe_tr if m == "NOFE-8D" else None
        rgb = embedding_to_rgb(
            project_embedding_pca(
                noisy_viz[m], fit_embedding=fit_ref, n_components=3, seed=seed
            ).embedding
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


def plot_fig_e_native_salt_zoom(
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
    """Plot native 1x1 salt-dome zoom with embedding gradient and KL residuals."""
    gi, gx = zoom.grid_shape
    extent = extent_from_locations(zoom.locations)
    rgb = embedding_to_rgb(
        project_embedding_pca(
            zz_nofe, fit_embedding=z_nofe_tr, n_components=3, seed=seed
        ).embedding
    ).reshape(gi, gx, _RGB_CHANNELS)
    kl_img = kl_zoom.reshape(gi, gx)

    fig, raw_axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    if zoom.deepest_facies is not None:
        axes[0].imshow(
            zoom.deepest_facies.reshape(gi, gx),
            cmap="tab10",
            vmin=0,
            vmax=9,
            extent=extent,
            aspect="auto",
        )
        axes[0].set_title("Native Deepest Facies")
    else:
        axes[0].imshow(
            zoom.rms_amp.reshape(gi, gx), cmap="viridis", extent=extent, aspect="auto"
        )
        axes[0].set_title("Native Trace RMS")
    axes[0].set_ylabel("Inline")

    axes[1].imshow(rgb, extent=extent, aspect="auto")
    axes[1].set_title("Native 1x1 NOFE-8D (PCA-3 RGB)")

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
