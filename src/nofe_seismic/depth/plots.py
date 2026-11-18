"""Visualisation helpers for 2D depth-log compression maps and reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nofe_seismic.types import GridShape2D, extent_from_locations
from nofe_seismic.viz.slices import embedding_to_rgb

_SAMPLE_TRACE_PANELS = 4


@dataclass(frozen=True)
class DepthMapData:
    """Arrays needed to plot 2D depth-log compression and diagnostic maps."""

    grid_shape: GridShape2D
    locations: np.ndarray
    rms_amp: np.ndarray
    deepest_facies: np.ndarray | None
    salt_fraction: np.ndarray | None
    split_inline: float


def plot_depth_rgb_maps(
    map_data: DepthMapData,
    embeddings: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Plot trace RMS, optional deepest facies and RGB depth-compressed maps."""
    gi, gx = map_data.grid_shape
    extent = extent_from_locations(map_data.locations)
    has_geo = map_data.salt_fraction is not None
    n_panels = 1 + (1 if has_geo else 0) + len(embeddings)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.4 * n_panels, 4.4), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    p = 0
    axes[p].imshow(
        map_data.rms_amp.reshape(gi, gx), cmap="viridis", extent=extent, aspect="auto"
    )
    axes[p].axhline(map_data.split_inline, color="white", linestyle="--", linewidth=1.2)
    axes[p].set_title("Trace RMS Energy")
    axes[p].set_xlabel("Crossline")
    axes[p].set_ylabel("Inline")
    p += 1

    if map_data.deepest_facies is not None:
        axes[p].imshow(
            map_data.deepest_facies.reshape(gi, gx),
            cmap="tab10",
            vmin=0,
            vmax=9,
            extent=extent,
            aspect="auto",
        )
        axes[p].axhline(
            map_data.split_inline, color="white", linestyle="--", linewidth=1.2
        )
        axes[p].set_title("Deepest Facies Province")
        axes[p].set_xlabel("Crossline")
        p += 1

    for name, emb in embeddings.items():
        rgb = embedding_to_rgb(emb).reshape(gi, gx, 3)
        axes[p].imshow(rgb, extent=extent, aspect="auto")
        axes[p].axhline(
            map_data.split_inline, color="white", linestyle="--", linewidth=1.2
        )
        axes[p].set_title(name)
        axes[p].set_xlabel("Crossline")
        p += 1

    fig.suptitle(
        "2D Depth-Log Compression Map (dashed line = train/held-out inline split)",
        fontsize=12,
    )
    fig.savefig(out_dir / "depth_compression_maps.png", dpi=180)
    plt.close(fig)


def plot_depth_diagnostics(
    map_data: DepthMapData,
    kl_map: np.ndarray,
    clusters_by_method: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Plot 2D per-trace KL residuals alongside salt fraction and clusters."""
    gi, gx = map_data.grid_shape
    extent = extent_from_locations(map_data.locations)
    n_c = 2 + len(clusters_by_method)
    fig, raw_axes = plt.subplots(
        1, n_c, figsize=(4.4 * n_c, 4.4), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    kl_img = kl_map.reshape(gi, gx)
    vmax = float(np.percentile(kl_img, 98.0)) or 1e-3
    im_kl = axes[0].imshow(
        kl_img, cmap="inferno", vmin=0.0, vmax=vmax, extent=extent, aspect="auto"
    )
    axes[0].axhline(map_data.split_inline, color="cyan", linestyle="--", linewidth=1.2)
    axes[0].set_title("Per-Trace KL Residual (NOFE-8D)")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Inline")
    fig.colorbar(im_kl, ax=axes[0], fraction=0.046, pad=0.04)

    if map_data.salt_fraction is not None:
        im_s = axes[1].imshow(
            map_data.salt_fraction.reshape(gi, gx),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        axes[1].set_title("Deep Unit / Salt Fraction")
        axes[1].set_xlabel("Crossline")
        fig.colorbar(im_s, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].imshow(
            map_data.rms_amp.reshape(gi, gx),
            cmap="viridis",
            extent=extent,
            aspect="auto",
        )
        axes[1].set_title("Trace RMS Energy")
        axes[1].set_xlabel("Crossline")

    for idx, (m_name, cl) in enumerate(clusters_by_method.items(), start=2):
        axes[idx].imshow(
            cl.reshape(gi, gx),
            cmap="tab10",
            vmin=0,
            vmax=9,
            extent=extent,
            aspect="auto",
        )
        axes[idx].axhline(
            map_data.split_inline, color="white", linestyle="--", linewidth=1.2
        )
        axes[idx].set_title(f"{m_name} Clusters")
        axes[idx].set_xlabel("Crossline")

    fig.suptitle("2D Depth-Log KL Residuals & Spatial Province Clustering", fontsize=12)
    fig.savefig(out_dir / "depth_compression_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_sample_reconstructions(
    traces_true: np.ndarray,
    traces_pred_by_method: dict[str, np.ndarray],
    locations: np.ndarray,
    deepest_facies: np.ndarray | None,
    out_path: Path,
) -> None:
    """Plot example depth logs vs probe reconstructions from compressed 2D codes."""
    rng = np.random.default_rng(0)
    chosen: list[int] = []
    if deepest_facies is not None:
        for cls in np.unique(deepest_facies)[:_SAMPLE_TRACE_PANELS]:
            idxs = np.flatnonzero(deepest_facies == cls)
            if idxs.size > 0:
                chosen.append(int(idxs[len(idxs) // 2]))
    while len(chosen) < _SAMPLE_TRACE_PANELS:
        chosen.append(int(rng.integers(0, traces_true.shape[0])))
    chosen = chosen[:_SAMPLE_TRACE_PANELS]

    z_axis = np.arange(traces_true.shape[1])
    fig, raw_axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    for ax, idx in zip(axes, chosen, strict=True):
        ax.plot(
            z_axis,
            traces_true[idx],
            color="black",
            linewidth=1.2,
            label="True depth log",
        )
        for m_name, pred in traces_pred_by_method.items():
            ax.plot(z_axis, pred[idx], linewidth=1.0, alpha=0.85, label=m_name)
        i_val, x_val = int(locations[idx, 0]), int(locations[idx, 1])
        suffix = (
            f", deep_facies={int(deepest_facies[idx])}"
            if deepest_facies is not None
            else ""
        )
        ax.set_title(f"Trace (inline={i_val}, crossline={x_val}{suffix})", fontsize=10)
        ax.set_xlabel("Depth / Vertical Sample")
        ax.set_ylabel("Std. Amplitude")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Depth-Log Reconstruction Probe from 2D Compressed Embeddings", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
