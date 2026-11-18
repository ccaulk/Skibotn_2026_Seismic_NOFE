"""Visualisation helpers for seismic slices, RGB embeddings and cluster maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from sklearn.decomposition import PCA

from nofe_seismic.types import Extent2D, GridShape2D

_RGB_CHANNELS = 3
_MIN_CHANNEL_SPAN = 1e-8


class PcaProjection(NamedTuple):
    """PCA-projected embedding and per-component explained variance ratios."""

    embedding: np.ndarray
    explained_variance: np.ndarray


@dataclass(frozen=True)
class SliceFrame:
    """Shared coordinate grid and axis metadata for 2D slice plotting."""

    coords: np.ndarray
    xlabel: str
    ylabel: str
    invert_y: bool = True


@dataclass(frozen=True)
class FieldStyle:
    """Colour mapping and value range options for 2D scalar fields."""

    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    rgb: bool = False


def project_embedding_pca(
    embedding: np.ndarray,
    *,
    fit_embedding: np.ndarray | None = None,
    n_components: int = 3,
    seed: int = 42,
) -> PcaProjection:
    """Project an embedding onto its top PCA components for RGB visualisation."""
    emb = np.asarray(embedding, dtype=np.float32)
    ref = emb if fit_embedding is None else np.asarray(fit_embedding, dtype=np.float32)
    k = min(n_components, emb.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(ref)
    return PcaProjection(
        embedding=pca.transform(emb).astype(np.float32),
        explained_variance=pca.explained_variance_ratio_.astype(np.float32),
    )


def embedding_to_rgb(
    embedding: np.ndarray,
    *,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """Normalise a 3-D embedding (or its first 3 dims) to ``[0, 1]`` RGB colours."""
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape[1] < _RGB_CHANNELS:
        pad = np.zeros((emb.shape[0], _RGB_CHANNELS - emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)
    else:
        emb = emb[:, :_RGB_CHANNELS]

    rgb = np.empty_like(emb)
    for ch in range(_RGB_CHANNELS):
        lo = float(np.percentile(emb[:, ch], low_pct))
        hi = float(np.percentile(emb[:, ch], high_pct))
        if hi - lo < _MIN_CHANNEL_SPAN:
            rgb[:, ch] = 0.5
        else:
            rgb[:, ch] = np.clip((emb[:, ch] - lo) / (hi - lo), 0.0, 1.0)
    return rgb


def draw_2d_field(
    ax: Axes,
    grid_coords: np.ndarray,
    values: np.ndarray,
    *,
    style: FieldStyle | None = None,
    invert_y: bool = True,
) -> AxesImage | PathCollection:
    """Render a 2D regular grid with ``imshow`` or scattered points fallback."""
    cfg = style if style is not None else FieldStyle()
    u_vals = np.unique(grid_coords[:, 0])
    v_vals = np.unique(grid_coords[:, 1])
    if u_vals.size * v_vals.size == grid_coords.shape[0]:
        order = np.lexsort((grid_coords[:, 0], grid_coords[:, 1]))
        y_top = float(v_vals.max()) if invert_y else float(v_vals.min())
        y_bot = float(v_vals.min()) if invert_y else float(v_vals.max())
        extent = Extent2D(float(u_vals.min()), float(u_vals.max()), y_top, y_bot)
        origin = "upper" if invert_y else "lower"
        if cfg.rgb:
            img = values[order].reshape(v_vals.size, u_vals.size, _RGB_CHANNELS)
            return ax.imshow(img, aspect="auto", extent=extent, origin=origin)
        img = values[order].reshape(v_vals.size, u_vals.size)
        return ax.imshow(
            img,
            cmap=cfg.cmap,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            aspect="auto",
            extent=extent,
            origin=origin,
        )

    if cfg.rgb:
        sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1], c=values, s=6, marker="s")
    else:
        sc = ax.scatter(
            grid_coords[:, 0],
            grid_coords[:, 1],
            c=values,
            cmap=cfg.cmap,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            s=6,
            marker="s",
        )
    if invert_y:
        ax.invert_yaxis()
    return sc


def _draw_amplitude_and_labels(
    axes: list[Axes],
    frame: SliceFrame,
    seismic_amp: np.ndarray,
    labels: np.ndarray | None,
) -> int:
    panel = 0
    lim = float(np.percentile(np.abs(seismic_amp), 99.0)) or 1.0
    draw_2d_field(
        axes[panel],
        frame.coords,
        seismic_amp,
        style=FieldStyle(cmap="gray", vmin=-lim, vmax=lim),
        invert_y=frame.invert_y,
    )
    axes[panel].set_title("Seismic Amplitude")
    axes[panel].set_xlabel(frame.xlabel)
    axes[panel].set_ylabel(frame.ylabel)
    panel += 1

    if labels is not None:
        draw_2d_field(
            axes[panel],
            frame.coords,
            labels,
            style=FieldStyle(cmap="tab10", vmin=0, vmax=9),
            invert_y=frame.invert_y,
        )
        axes[panel].set_title("Facies Labels")
        axes[panel].set_xlabel(frame.xlabel)
        panel += 1
    return panel


def plot_slice_comparison(
    frame: SliceFrame,
    seismic_amp: np.ndarray,
    embeddings_by_method: dict[str, np.ndarray],
    *,
    title: str,
    out_path: Path,
    labels: np.ndarray | None = None,
) -> None:
    """Plot seismic amplitude, optional facies labels and RGB embeddings on a slice."""
    n_panels = 1 + (1 if labels is not None else 0) + len(embeddings_by_method)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4.5), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    panel = _draw_amplitude_and_labels(axes, frame, seismic_amp, labels)

    for name, emb in embeddings_by_method.items():
        rgb = embedding_to_rgb(emb)
        draw_2d_field(
            axes[panel],
            frame.coords,
            rgb,
            style=FieldStyle(rgb=True),
            invert_y=frame.invert_y,
        )
        axes[panel].set_title(f"{name} (RGB)")
        axes[panel].set_xlabel(frame.xlabel)
        panel += 1

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_kl_residual_slice(
    frame: SliceFrame,
    seismic_amp: np.ndarray,
    per_node_kl: np.ndarray,
    *,
    title: str,
    out_path: Path,
    labels: np.ndarray | None = None,
) -> None:
    """Plot seismic amplitude, optional facies labels and per-node KL residuals."""
    n_panels = 2 + (1 if labels is not None else 0)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.8 * n_panels, 4.5), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    panel = _draw_amplitude_and_labels(axes, frame, seismic_amp, labels)

    vmax = float(np.percentile(per_node_kl, 98.0)) or 1e-3
    im = draw_2d_field(
        axes[panel],
        frame.coords,
        per_node_kl,
        style=FieldStyle(cmap="inferno", vmin=0.0, vmax=vmax),
        invert_y=frame.invert_y,
    )
    axes[panel].set_title("Per-Node KL Divergence")
    axes[panel].set_xlabel(frame.xlabel)
    fig.colorbar(im, ax=axes[panel], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cluster_slices(
    frame: SliceFrame,
    clusters_by_method: dict[str, np.ndarray],
    *,
    title: str,
    out_path: Path,
    labels: np.ndarray | None = None,
) -> None:
    """Plot ground-truth facies alongside K-Means cluster maps for each method."""
    n_panels = (1 if labels is not None else 0) + len(clusters_by_method)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4.5), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    panel = 0
    if labels is not None:
        draw_2d_field(
            axes[panel],
            frame.coords,
            labels,
            style=FieldStyle(cmap="tab10", vmin=0, vmax=9),
            invert_y=frame.invert_y,
        )
        axes[panel].set_title("Ground Truth Facies")
        axes[panel].set_xlabel(frame.xlabel)
        axes[panel].set_ylabel(frame.ylabel)
        panel += 1

    for name, clust in clusters_by_method.items():
        draw_2d_field(
            axes[panel],
            frame.coords,
            clust,
            style=FieldStyle(cmap="tab10", vmin=0, vmax=9),
            invert_y=frame.invert_y,
        )
        axes[panel].set_title(name)
        axes[panel].set_xlabel(frame.xlabel)
        if panel == 0:
            axes[panel].set_ylabel(frame.ylabel)
        panel += 1

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def resolve_grid_shape(shape: tuple[int, int]) -> GridShape2D:
    return GridShape2D(int(shape[0]), int(shape[1]))
