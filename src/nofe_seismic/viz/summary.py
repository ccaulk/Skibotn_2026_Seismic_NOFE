"""Summary bar, curve and 8D-projection visualisations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nofe_seismic.types import extent_from_locations

from .slices import (
    PcaProjection,
    embedding_to_rgb,
    project_embedding_pca,
    resolve_grid_shape,
)

_RGB_CHANNELS = 3


def _float_cell(row: Mapping[str, float | int | str | None], key: str) -> float:
    val = row[key]
    if not isinstance(val, (int, float)):
        raise TypeError(f"Expected numeric value for {key!r}, got {type(val).__name__}")
    return float(val)


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
) -> PcaProjection:
    """Compare raw dims 0..2 RGB against PCA-3 RGB and top-3 PCs."""
    gs = resolve_grid_shape(grid_shape)
    extent = extent_from_locations(locations)
    proj = project_embedding_pca(
        emb_8d, fit_embedding=fit_embedding, n_components=3, seed=seed
    )
    rgb_raw3 = embedding_to_rgb(emb_8d[:, :_RGB_CHANNELS]).reshape(
        gs.nu, gs.nv, _RGB_CHANNELS
    )
    rgb_pca3 = embedding_to_rgb(proj.embedding).reshape(gs.nu, gs.nv, _RGB_CHANNELS)

    fig, raw_axes = plt.subplots(1, 5, figsize=(22, 4.3), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    axes[0].imshow(rgb_raw3, extent=extent, aspect="auto")
    axes[0].set_title("NOFE-8D Dims 0-2 (RGB)")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Inline")

    var_pct = 100.0 * float(np.sum(proj.explained_variance))
    axes[1].imshow(rgb_pca3, extent=extent, aspect="auto")
    axes[1].set_title(f"NOFE-8D PCA-3 (RGB, {var_pct:.1f}% var)")
    axes[1].set_xlabel("Crossline")

    for k in range(_RGB_CHANNELS):
        ax = axes[2 + k]
        comp = proj.embedding[:, k].reshape(gs.nu, gs.nv)
        lo, hi = float(np.percentile(comp, 2.0)), float(np.percentile(comp, 98.0))
        im = ax.imshow(
            comp, cmap="coolwarm", vmin=lo, vmax=hi, extent=extent, aspect="auto"
        )
        ax.set_title(
            f"PC{k + 1} ({100.0 * float(proj.explained_variance[k]):.1f}% var)"
        )
        ax.set_xlabel("Crossline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if split_inline is not None:
        for ax in axes:
            ax.axhline(split_inline, color="white", linestyle="--", linewidth=1.1)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return proj


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
