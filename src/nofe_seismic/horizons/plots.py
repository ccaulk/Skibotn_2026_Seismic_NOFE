"""Diagnostic figures for NOFE horizon boundary reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nofe_seismic.types import Extent2D, RowDict
from nofe_seismic.viz.slices import embedding_to_rgb, project_embedding_pca

from .boundaries import extract_unit_top_depths
from .config import HorizonConfig
from .evals import (
    NativeSlice,
    ProbeMethodArt,
    UnsupervisedMethodArt,
)
from .robustness import Slab3DPlotData, TileSeamResult


def _slice_extent(u_coords: np.ndarray, z_coords: np.ndarray) -> Extent2D:
    return Extent2D(
        xmin=float(u_coords.min()),
        xmax=float(u_coords.max()),
        ymax=float(z_coords.max()),
        ymin=float(z_coords.min()),
    )


def plot_fig1_unsupervised(
    sl_data: NativeSlice,
    nofe_art: UnsupervisedMethodArt,
    z_tr_nofe: np.ndarray,
    *,
    slice_name: str,
    out_path: Path,
    seed: int,
) -> None:
    """Plot unsupervised amplitude, PCA-3 RGB, boundary contrast and RGT horizons."""
    u_coords = sl_data.u_coords
    z_coords = sl_data.z_coords
    extent = _slice_extent(u_coords, z_coords)
    proj = project_embedding_pca(
        sl_data.nofe.reshape(-1, sl_data.nofe.shape[2]),
        fit_embedding=z_tr_nofe,
        n_components=3,
        seed=seed,
    ).embedding
    rgb = embedding_to_rgb(proj).reshape(u_coords.size, z_coords.size, 3)

    n_panels = 5 if sl_data.labels is not None else 4
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.3 * n_panels, 4.5), constrained_layout=True
    )
    lim = float(np.percentile(np.abs(sl_data.amplitude), 99.0)) or 1.0
    p = 0

    axes[p].imshow(
        sl_data.amplitude.T,
        cmap="gray",
        vmin=-lim,
        vmax=lim,
        extent=extent,
        aspect="auto",
    )
    axes[p].set_title("Seismic Amplitude")
    axes[p].set_ylabel("Depth Sample")
    p += 1

    if sl_data.labels is not None:
        axes[p].imshow(
            sl_data.labels.T, cmap="tab10", vmin=0, vmax=9, extent=extent, aspect="auto"
        )
        axes[p].set_title("Ground-Truth Facies")
        p += 1

    axes[p].imshow(np.transpose(rgb, (1, 0, 2)), extent=extent, aspect="auto")
    axes[p].set_title("NOFE-8D (PCA-3 RGB)")
    p += 1

    vmax_c = float(np.percentile(nofe_art.contrast, 98.0)) or 1.0
    axes[p].imshow(
        nofe_art.contrast.T,
        cmap="magma",
        vmin=0.0,
        vmax=vmax_c,
        extent=extent,
        aspect="auto",
    )
    ru, rz = np.where(nofe_art.ridges)
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
        sl_data.amplitude.T,
        cmap="gray",
        vmin=-lim,
        vmax=lim,
        extent=extent,
        aspect="auto",
    )
    colors = ["#00e5ff", "#ffeb3b", "#ff5252", "#69f0ae"]
    for idx, item in enumerate(nofe_art.discovered):
        axes[p].plot(
            u_coords,
            item.depths,
            color=colors[idx % len(colors)],
            linewidth=1.8,
            label=f"RGT Horizon #{idx + 1}",
        )
    axes[p].set_title("Zero-Seed Auto-Discovered\nNOFE RGT Horizons")
    axes[p].legend(loc="lower left", fontsize=7.5)

    for ax in axes:
        ax.set_xlabel(sl_data.xlabel)
    fig.suptitle(
        f"Unsupervised Boundary Reconstruction from NOFE-8D ({slice_name})", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_horizon_panel(
    ax: Axes,
    sl: NativeSlice,
    curves: dict[int, np.ndarray],
    true_tops: dict[int, np.ndarray],
    *,
    title: str,
    colors: dict[int, str],
    show_legend: bool = False,
) -> None:
    extent = _slice_extent(sl.u_coords, sl.z_coords)
    lim = float(np.percentile(np.abs(sl.amplitude), 99.0)) or 1.0
    ax.imshow(
        sl.amplitude.T, cmap="gray", vmin=-lim, vmax=lim, extent=extent, aspect="auto"
    )
    for uid, curve in curves.items():
        if not np.isfinite(curve).any():
            continue
        if uid in true_tops and np.isfinite(true_tops[uid]).any():
            ax.plot(
                sl.u_coords,
                true_tops[uid],
                color="white",
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
            )
        ax.plot(
            sl.u_coords,
            curve,
            color=colors.get(uid, "cyan"),
            linewidth=1.8,
            label=f"Unit {uid}",
        )
    ax.set_title(title)
    ax.set_xlabel(sl.xlabel)
    if show_legend:
        ax.legend(loc="lower left", fontsize=7.5)


def plot_fig2_tracking_comparison(
    sl_inline: NativeSlice,
    sl_cross: NativeSlice,
    *,
    unsup_in: dict[str, UnsupervisedMethodArt],
    unsup_xl: dict[str, UnsupervisedMethodArt],
    probe_in: dict[str, ProbeMethodArt],
    probe_xl: dict[str, ProbeMethodArt],
    unit_ids: Sequence[int],
    out_path: Path,
) -> None:
    """Plot true vs seeded-RGT vs probe-DP tracked horizons across held-out slices."""
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)
    row_specs = [
        ("Held-Out Inline", sl_inline, unsup_in, probe_in),
        ("Crossline", sl_cross, unsup_xl, probe_xl),
    ]
    h_colors = {1: "#ffeb3b", 2: "#00e5ff", 3: "#ff5252", 4: "#e040fb", 5: "#69f0ae"}

    for r_idx, (r_title, sl, u_art, p_art) in enumerate(row_specs):
        true_tops = (
            extract_unit_top_depths(sl.labels, unit_ids, sl.z_coords)
            if sl.labels is not None
            else {}
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
            u_art["NOFE-8D"].seeded_curves,
            true_tops,
            title=f"{r_title}: NOFE-8D Seeded RGT",
            colors=h_colors,
        )
        _draw_horizon_panel(
            axes[r_idx, 2],
            sl,
            p_art["NOFE-8D"].pred_tops,
            true_tops,
            title=f"{r_title}: NOFE-8D Probe + Monotonic DP",
            colors=h_colors,
        )
        _draw_horizon_panel(
            axes[r_idx, 3],
            sl,
            p_art["PCA-8D"].pred_tops,
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


def plot_fig3_robustness(
    sl_inline: NativeSlice,
    seam_info: TileSeamResult,
    stress_rows: list[RowDict],
    cfg: HorizonConfig,
    out_path: Path,
) -> None:
    """Plot two-tile seam mosaics and noisy RGT depth error bars."""
    extent = _slice_extent(sl_inline.u_coords, sl_inline.z_coords)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    c_nofe = seam_info.contrast_mosaic["NOFE-8D"]
    vmax_n = float(np.percentile(c_nofe, 98.0)) or 1.0
    axes[0].imshow(
        c_nofe.T, cmap="magma", vmin=0.0, vmax=vmax_n, extent=extent, aspect="auto"
    )
    axes[0].axvline(seam_info.mid_x, color="cyan", linestyle="--", linewidth=1.3)
    axes[0].set_title(
        "NOFE-8D 2-Tile Boundary Contrast\n"
        f"(Seam NRMSE={seam_info.nofe_seam_norm_rmse:.3f})"
    )
    axes[0].set_xlabel(sl_inline.xlabel)
    axes[0].set_ylabel("Depth Sample")

    c_pca = seam_info.contrast_mosaic["PCA-8D (Per-Tile)"]
    vmax_p = float(np.percentile(c_pca, 98.0)) or 1.0
    axes[1].imshow(
        c_pca.T, cmap="magma", vmin=0.0, vmax=vmax_p, extent=extent, aspect="auto"
    )
    axes[1].axvline(seam_info.mid_x, color="cyan", linestyle="--", linewidth=1.3)
    axes[1].set_title(
        "Per-Tile PCA-8D Boundary Contrast\n"
        f"(Seam NRMSE={seam_info.pca_tile_seam_norm_rmse:.3f})"
    )
    axes[1].set_xlabel(sl_inline.xlabel)

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
    for bar, val in zip(bars, maes, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f}",
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


def plot_fig4_3d_surfaces(plot_data: Slab3DPlotData, out_path: Path) -> None:
    """Plot true vs reconstructed 3D horizon surfaces and absolute depth error."""
    ii, xx = np.meshgrid(plot_data.i_coords, plot_data.x_coords, indexing="ij")
    fig = plt.figure(figsize=(14.5, 4.8), constrained_layout=True)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(
        ii, xx, plot_data.true_surf, cmap="viridis", linewidth=0, antialiased=True
    )
    ax1.invert_zaxis()
    ax1.set_title("True 3D Horizon Surface (Unit 1 Top)")
    ax1.set_xlabel("Inline")
    ax1.set_ylabel("Crossline")
    ax1.set_zlabel("Depth")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(
        ii, xx, plot_data.pred_surf, cmap="viridis", linewidth=0, antialiased=True
    )
    ax2.invert_zaxis()
    ax2.set_title("NOFE-8D Reconstructed 3D Surface")
    ax2.set_xlabel("Inline")
    ax2.set_ylabel("Crossline")
    ax2.set_zlabel("Depth")

    ax3 = fig.add_subplot(1, 3, 3)
    diff = np.abs(plot_data.pred_surf - plot_data.true_surf)
    extent = Extent2D(
        xmin=float(plot_data.x_coords.min()),
        xmax=float(plot_data.x_coords.max()),
        ymax=float(plot_data.i_coords.max()),
        ymin=float(plot_data.i_coords.min()),
    )
    finite_diff = diff[np.isfinite(diff)]
    vmax = (float(np.percentile(finite_diff, 96.0)) or 1.0) if finite_diff.size else 1.0
    im = ax3.imshow(
        diff, cmap="inferno", vmin=0.0, vmax=vmax, extent=extent, aspect="auto"
    )
    ax3.set_title(f"Absolute Depth Error |$\\Delta z$|\n({plot_data.err_title})")
    ax3.set_xlabel("Crossline")
    ax3.set_ylabel("Inline")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(
        "3D Held-Out Slab Boundary Surface Reconstruction from NOFE-8D", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
