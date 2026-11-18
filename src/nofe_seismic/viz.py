"""Visualisation helpers for seismic embeddings, RGB composites and metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from sklearn.decomposition import PCA

_RGB_CHANNELS = 3
_MIN_CHANNEL_SPAN = 1e-8


def project_embedding_pca(
    embedding: np.ndarray,
    *,
    fit_embedding: np.ndarray | None = None,
    n_components: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Project an embedding onto its top PCA components for RGB visualisation."""
    emb = np.asarray(embedding, dtype=np.float32)
    ref = emb if fit_embedding is None else np.asarray(fit_embedding, dtype=np.float32)
    k = min(n_components, emb.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(ref)
    proj = pca.transform(emb).astype(np.float32)
    return proj, pca.explained_variance_ratio_.astype(np.float32)


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


def _draw_2d_field(
    ax: Axes,
    grid_coords: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    rgb: bool = False,
    invert_y: bool = True,
) -> AxesImage | PathCollection:
    u_vals = np.unique(grid_coords[:, 0])
    v_vals = np.unique(grid_coords[:, 1])
    is_regular = u_vals.size * v_vals.size == grid_coords.shape[0]
    if is_regular:
        order = np.lexsort((grid_coords[:, 0], grid_coords[:, 1]))
        y_top = float(v_vals.max()) if invert_y else float(v_vals.min())
        y_bot = float(v_vals.min()) if invert_y else float(v_vals.max())
        extent = (float(u_vals.min()), float(u_vals.max()), y_top, y_bot)
        origin = "upper" if invert_y else "lower"
        if rgb:
            img = values[order].reshape(v_vals.size, u_vals.size, _RGB_CHANNELS)
            return ax.imshow(img, aspect="auto", extent=extent, origin=origin)
        img = values[order].reshape(v_vals.size, u_vals.size)
        return ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=extent,
            origin=origin,
        )

    if rgb:
        sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1], c=values, s=6, marker="s")
    else:
        sc = ax.scatter(
            grid_coords[:, 0],
            grid_coords[:, 1],
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=6,
            marker="s",
        )
    if invert_y:
        ax.invert_yaxis()
    return sc


def plot_slice_comparison(
    grid_coords: np.ndarray,
    seismic_amp: np.ndarray,
    embeddings_by_method: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    labels: np.ndarray | None = None,
    invert_y: bool = True,
) -> None:
    """Plot seismic amplitude, optional facies labels and RGB embeddings on a slice."""
    n_panels = 1 + (1 if labels is not None else 0) + len(embeddings_by_method)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4.5), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    panel = 0
    lim = float(np.percentile(np.abs(seismic_amp), 99.0)) or 1.0
    _draw_2d_field(
        axes[panel],
        grid_coords,
        seismic_amp,
        cmap="gray",
        vmin=-lim,
        vmax=lim,
        invert_y=invert_y,
    )
    axes[panel].set_title("Seismic Amplitude")
    axes[panel].set_xlabel(xlabel)
    axes[panel].set_ylabel(ylabel)
    panel += 1

    if labels is not None:
        _draw_2d_field(
            axes[panel],
            grid_coords,
            labels,
            cmap="tab10",
            vmin=0,
            vmax=9,
            invert_y=invert_y,
        )
        axes[panel].set_title("Facies Labels")
        axes[panel].set_xlabel(xlabel)
        panel += 1

    for name, emb in embeddings_by_method.items():
        rgb = embedding_to_rgb(emb)
        _draw_2d_field(axes[panel], grid_coords, rgb, rgb=True, invert_y=invert_y)
        axes[panel].set_title(f"{name} (RGB)")
        axes[panel].set_xlabel(xlabel)
        panel += 1

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_kl_residual_slice(
    grid_coords: np.ndarray,
    seismic_amp: np.ndarray,
    per_node_kl: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    labels: np.ndarray | None = None,
    invert_y: bool = True,
) -> None:
    """Plot seismic amplitude, optional facies labels and per-node KL residuals."""
    n_panels = 2 + (1 if labels is not None else 0)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.8 * n_panels, 4.5), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    panel = 0
    lim = float(np.percentile(np.abs(seismic_amp), 99.0)) or 1.0
    _draw_2d_field(
        axes[panel],
        grid_coords,
        seismic_amp,
        cmap="gray",
        vmin=-lim,
        vmax=lim,
        invert_y=invert_y,
    )
    axes[panel].set_title("Seismic Amplitude")
    axes[panel].set_xlabel(xlabel)
    axes[panel].set_ylabel(ylabel)
    panel += 1

    if labels is not None:
        _draw_2d_field(
            axes[panel],
            grid_coords,
            labels,
            cmap="tab10",
            vmin=0,
            vmax=9,
            invert_y=invert_y,
        )
        axes[panel].set_title("Facies Labels")
        axes[panel].set_xlabel(xlabel)
        panel += 1

    vmax = float(np.percentile(per_node_kl, 98.0)) or 1e-3
    im = _draw_2d_field(
        axes[panel],
        grid_coords,
        per_node_kl,
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        invert_y=invert_y,
    )
    axes[panel].set_title("Per-Node KL Divergence")
    axes[panel].set_xlabel(xlabel)
    fig.colorbar(im, ax=axes[panel], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cluster_slices(
    grid_coords: np.ndarray,
    clusters_by_method: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    labels: np.ndarray | None = None,
    invert_y: bool = True,
) -> None:
    """Plot ground-truth facies alongside K-Means cluster maps for each method."""
    n_panels = (1 if labels is not None else 0) + len(clusters_by_method)
    fig, raw_axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 4.5), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    panel = 0
    if labels is not None:
        _draw_2d_field(
            axes[panel],
            grid_coords,
            labels,
            cmap="tab10",
            vmin=0,
            vmax=9,
            invert_y=invert_y,
        )
        axes[panel].set_title("Ground Truth Facies")
        axes[panel].set_xlabel(xlabel)
        axes[panel].set_ylabel(ylabel)
        panel += 1

    for name, clust in clusters_by_method.items():
        _draw_2d_field(
            axes[panel],
            grid_coords,
            clust,
            cmap="tab10",
            vmin=0,
            vmax=9,
            invert_y=invert_y,
        )
        axes[panel].set_title(name)
        axes[panel].set_xlabel(xlabel)
        if panel == 0:
            axes[panel].set_ylabel(ylabel)
        panel += 1

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(
    histories: dict[str, list[dict[str, float]]],
    out_path: Path,
    *,
    title: str = "NOFE Training & Validation KL Loss",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, hist in histories.items():
        epochs = [row["epoch"] for row in hist]
        ax.plot(epochs, [row["train_loss"] for row in hist], label=f"{label} train")
        ax.plot(
            epochs,
            [row["val_loss"] for row in hist],
            linestyle="--",
            label=f"{label} val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Neighbourhood KL Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _float_cell(row: Mapping[str, float | int | str | None], key: str) -> float:
    val = row[key]
    if not isinstance(val, (int, float)):
        raise TypeError(f"Expected numeric value for {key!r}, got {type(val).__name__}")
    return float(val)


def plot_benchmark_bars(
    rows: Sequence[Mapping[str, float | int | str | None]],
    out_path: Path,
    *,
    title: str = "Method Comparison on 3D Seismic Data",
) -> None:
    """Bar chart comparing methods across structure and spatial metrics."""
    methods = [str(r["method"]) for r in rows]
    metric_keys = (
        ("trustworthiness", "Trustworthiness (higher better)"),
        ("continuity", "Continuity (higher better)"),
        ("local_stress", "Local Stress (lower better)"),
        ("global_stress", "Global Stress (lower better)"),
    )
    fig, raw_axes = plt.subplots(
        1, len(metric_keys), figsize=(15, 4.0), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    for ax, (key, subtitle) in zip(axes, metric_keys, strict=True):
        vals = [_float_cell(r, key) for r in rows]
        bars = ax.bar(methods, vals, color=colors[: len(methods)])
        ax.set_title(subtitle, fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_8d_projection_map(
    emb_8d: np.ndarray,
    grid_shape: tuple[int, int],
    locations: np.ndarray,
    out_path: Path,
    *,
    fit_embedding: np.ndarray | None = None,
    split_inline: float | None = None,
    title: str = "Visualising NOFE-8D: First 3 Dims vs. PCA-3 Projection",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare raw dims 0..2 RGB against PCA-3 RGB and top-3 PCs of an 8D embedding."""
    gi, gx = grid_shape
    extent = (
        float(locations[:, 1].min()),
        float(locations[:, 1].max()),
        float(locations[:, 0].max()),
        float(locations[:, 0].min()),
    )
    pca3, var_ratio = project_embedding_pca(
        emb_8d, fit_embedding=fit_embedding, n_components=3, seed=seed
    )
    rgb_raw3 = embedding_to_rgb(emb_8d[:, :_RGB_CHANNELS]).reshape(
        gi, gx, _RGB_CHANNELS
    )
    rgb_pca3 = embedding_to_rgb(pca3).reshape(gi, gx, _RGB_CHANNELS)

    fig, raw_axes = plt.subplots(1, 5, figsize=(22, 4.3), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    axes[0].imshow(rgb_raw3, extent=extent, aspect="auto")
    axes[0].set_title("NOFE-8D Dims 0-2 (RGB)")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Inline")

    var_pct = 100.0 * float(np.sum(var_ratio))
    axes[1].imshow(rgb_pca3, extent=extent, aspect="auto")
    axes[1].set_title(f"NOFE-8D PCA-3 (RGB, {var_pct:.1f}% var)")
    axes[1].set_xlabel("Crossline")

    for k in range(_RGB_CHANNELS):
        ax = axes[2 + k]
        comp = pca3[:, k].reshape(gi, gx)
        lo, hi = float(np.percentile(comp, 2.0)), float(np.percentile(comp, 98.0))
        im = ax.imshow(
            comp, cmap="coolwarm", vmin=lo, vmax=hi, extent=extent, aspect="auto"
        )
        ax.set_title(f"PC{k + 1} ({100.0 * float(var_ratio[k]):.1f}% var)")
        ax.set_xlabel("Crossline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if split_inline is not None:
        for ax in axes:
            ax.axhline(split_inline, color="white", linestyle="--", linewidth=1.1)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return pca3, var_ratio


def plot_ablation_summary(
    rows: Sequence[Mapping[str, float | int | str | None]],
    out_path: Path,
    *,
    title: str = "NOFE Hyperparameter & Nystrom Ablation Summary",
) -> None:
    """Plot KL loss, Local Stress and Trustworthiness across ablation configs."""
    labels = [str(r["config"]) for r in rows]
    metrics = (
        ("kl_loss", "Final KL Loss (lower better)"),
        ("local_stress", "Local Stress (lower better)"),
        ("trustworthiness", "Trustworthiness (higher better)"),
    )
    fig, raw_axes = plt.subplots(
        1, len(metrics), figsize=(14, 4.2), constrained_layout=True
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    for ax, (key, subtitle) in zip(axes, metrics, strict=True):
        vals = [_float_cell(r, key) for r in rows]
        bars = ax.bar(labels, vals, color="#2a7b9b")
        ax.set_title(subtitle, fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
