"""Metric curves and 3D facies separation plots for ablation studies."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from umap import UMAP

from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume
from nofe_seismic.types import RowDict

from .evals import Ablation3DResult, balanced_sample_indices

F3_FACIES_NAMES: tuple[str, ...] = (
    "0: Upper North Sea",
    "1: Middle North Sea",
    "2: Lower North Sea",
    "3: Rijnland / Chalk",
    "4: Scruff",
    "5: Zechstein",
)
F3_FACIES_SHORT: tuple[str, ...] = (
    "U. North Sea",
    "M. North Sea",
    "L. North Sea",
    "Chalk",
    "Scruff",
    "Zechstein",
)
FACIES_COLORS: tuple[str, ...] = (
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#984ea3",
    "#8c564b",
)

_UMAP_MIN_NODES = 3
_F3_PCA_REF_DIM = 8
_CM_HIGHLIGHT_THRESH = 0.5


class FaciesStyle(NamedTuple):
    """Discrete Matplotlib colour map and boundary norm for geological facies."""

    cmap: ListedColormap
    norm: BoundaryNorm


def facies_cmap_and_norm(classes: np.ndarray) -> FaciesStyle:
    """Build discrete colour map and boundary norm for geological facies."""
    max_cls = int(classes.max())
    cmap = ListedColormap(
        [FACIES_COLORS[c % len(FACIES_COLORS)] for c in range(max_cls + 1)]
    )
    norm = BoundaryNorm(np.arange(-0.5, max_cls + 1.5, 1.0), cmap.N)
    return FaciesStyle(cmap=cmap, norm=norm)


def _style_dim_axis(
    ax: Axes, dims: list[int], *, xlabel: str, ylabel: str, title: str
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _plot_2d_ablation_row(axes_row: np.ndarray, df2: pd.DataFrame) -> None:
    colors = {"NOFE": "#1b7837", "PCA": "#762a83", "UMAP": "#d95f02"}
    dims2 = sorted(df2["dim"].unique())
    y_col = (
        "facies_fraction_r2"
        if "facies_fraction_r2" in df2.columns
        else "trustworthiness"
    )
    for method, col in colors.items():
        sub = df2[df2["method"] == method].sort_values("dim")
        axes_row[0].plot(
            sub["dim"], sub["local_stress"], marker="o", lw=2.2, color=col, label=method
        )
        axes_row[1].plot(
            sub["dim"],
            sub["depth_recon_r2"],
            marker="s",
            lw=2.2,
            color=col,
            label=method,
        )
        axes_row[2].plot(
            sub["dim"], sub[y_col], marker="^", lw=2.2, color=col, label=method
        )
    _style_dim_axis(
        axes_row[0],
        dims2,
        xlabel="Latent dimension d",
        ylabel="Local spatial stress (lower better)",
        title="(a) 2D depth-log local geometry stress",
    )
    _style_dim_axis(
        axes_row[1],
        dims2,
        xlabel="Latent dimension d",
        ylabel="MLP depth-log reconstruction R^2 (higher better)",
        title="(b) Full trace depth-log reconstruction",
    )
    _style_dim_axis(
        axes_row[2],
        dims2,
        xlabel="Latent dimension d",
        ylabel="Stratigraphic fraction R^2"
        if "facies_fraction_r2" in df2.columns
        else "Trustworthiness",
        title="(c) 2D formation thickness preservation",
    )


def _plot_3d_pareto_row(
    axes_row: np.ndarray, df2: pd.DataFrame, df3: pd.DataFrame
) -> None:
    colors = {"NOFE": "#1b7837", "PCA": "#762a83", "UMAP": "#d95f02"}
    colors3 = {
        "NOFE-Tuned (d=1, aniso)": "#1b7837",
        "PCA (tex7+z)": "#762a83",
        "UMAP (tex7+z)": "#d95f02",
    }
    dims2 = sorted(df2["dim"].unique())
    dims3 = sorted(df3["dim"].unique())

    for method, col in colors3.items():
        sub = df3[df3["method"] == method].sort_values("dim")
        if len(sub) > 0:
            axes_row[0].plot(
                sub["dim"],
                sub["lr_bal_acc"],
                marker="o",
                lw=2.2,
                color=col,
                label=method,
            )
    naive = df3[df3["method"] == "NOFE-Naive (d=2, iso)"]
    if len(naive) > 0:
        axes_row[0].scatter(
            naive["dim"],
            naive["lr_bal_acc"],
            s=110,
            marker="X",
            color="#e41a1c",
            zorder=5,
            label="NOFE-naive (d=2, iso)",
        )
    _style_dim_axis(
        axes_row[0],
        dims3,
        xlabel="Latent dimension d",
        ylabel="Balanced accuracy (higher better)",
        title="(d) 3D facies linear separation vs d",
    )

    sub_n2 = df2[df2["method"] == "NOFE"].sort_values("dim")
    sub_n3 = df3[df3["method"] == "NOFE-Tuned (d=1, aniso)"].sort_values("dim")
    axes_row[1].plot(
        sub_n2["dim"],
        sub_n2["heldout_kl"],
        marker="o",
        lw=2.2,
        color="#1b7837",
        label="2D NOFE KL",
    )
    axes_row[1].plot(
        sub_n3["dim"],
        sub_n3["heldout_kl"],
        marker="s",
        lw=2.2,
        ls="--",
        color="#2166ac",
        label="3D tuned NOFE KL",
    )
    axes_row[1].set_yscale("log")
    _style_dim_axis(
        axes_row[1],
        dims2,
        xlabel="Latent dimension d",
        ylabel="Held-out affinity KL loss (log)",
        title="(e) Affinity KL decay across latent dims",
    )

    for method, col in colors.items():
        sub = df2[df2["method"] == method].sort_values("compression_ratio")
        joint = 0.5 * (1.0 - sub["local_stress"]) + 0.5 * np.clip(
            sub["depth_recon_r2"], 0.0, 1.0
        )
        axes_row[2].plot(
            sub["compression_ratio"], joint, marker="o", lw=2.2, color=col, label=method
        )
        if method == "NOFE":
            for _, row in sub.iterrows():
                y_pt = 0.5 * (1.0 - row["local_stress"]) + 0.5 * max(
                    0.0, row["depth_recon_r2"]
                )
                axes_row[2].annotate(
                    f"d={int(row['dim'])}",
                    (row["compression_ratio"], y_pt),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                    color="#1b7837",
                    fontweight="bold",
                )
    axes_row[2].set_xscale("log")
    axes_row[2].set_xlabel("Depth compression ratio (253 / d, log)")
    axes_row[2].set_ylabel("Joint geometry + reconstruction score")
    axes_row[2].set_title("(f) Rate-distortion Pareto frontier", fontweight="bold")
    axes_row[2].grid(alpha=0.25)
    axes_row[2].legend(fontsize=8)


def plot_fig1_latent_ablation(
    rows_2d: list[RowDict],
    rows_3d: list[RowDict],
    out_path: Path,
) -> None:
    """Plot 2x3 Pareto and latent-dimension ablation curves for 2D and 3D NOFE."""
    df2, df3 = pd.DataFrame(rows_2d), pd.DataFrame(rows_3d)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
    _plot_2d_ablation_row(axes[0], df2)
    _plot_3d_pareto_row(axes[1], df2, df3)
    fig.suptitle(
        "Latent-dimension ablations (d = 2..32): "
        "geometry, reconstruction & facies separation",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _draw_confusion(
    ax: Axes, cm_list: list[list[float]], short_names: list[str], title: str
) -> None:
    cm = np.asarray(cm_list, dtype=np.float32)
    ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for r in range(cm.shape[0]):
        for c_idx in range(cm.shape[1]):
            val = cm[r, c_idx]
            ax.text(
                c_idx,
                r,
                f"{100 * val:.0f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if val > _CM_HIGHLIGHT_THRESH else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")


def plot_fig2_facies_breakthrough(
    volume: SeismicVolume | LazySeismicVolume,
    res3d: Ablation3DResult,
    *,
    seed: int,
    out_path: Path,
) -> None:
    """Plot 2x3 before-vs-after 3D facies separation manifolds and metrics."""
    if volume.labels is None:
        raise ValueError("3D facies figure requires labeled seismic volume.")
    classes = np.asarray(res3d.tuned_fac_hero.classes, dtype=int)
    style = facies_cmap_and_norm(classes)
    short_names = [
        F3_FACIES_SHORT[c] if c < len(F3_FACIES_SHORT) else f"F{c}" for c in classes
    ]

    bal_idx = balanced_sample_indices(res3d.y_te, max_per_class=750, seed=seed)
    y_bal = res3d.y_te[bal_idx]
    n_nbrs = min(15, max(_UMAP_MIN_NODES, len(bal_idx) - 2))
    um_naive = UMAP(
        n_components=2, n_neighbors=n_nbrs, min_dist=0.15, random_state=seed
    ).fit_transform(res3d.zn_te[bal_idx])
    um_tuned = UMAP(
        n_components=2, n_neighbors=n_nbrs, min_dist=0.15, random_state=seed
    ).fit_transform(res3d.zt_te_hero[bal_idx])

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.8), constrained_layout=True)
    i_mid = volume.shape[0] // 2
    lbl_slice = np.asarray(volume.labels[i_mid, :, :], dtype=np.int64).T
    im = axes[0, 0].imshow(
        lbl_slice,
        aspect="auto",
        cmap=style.cmap,
        norm=style.norm,
        interpolation="nearest",
    )
    axes[0, 0].set_xlabel("Crossline")
    axes[0, 0].set_ylabel("Depth")
    axes[0, 0].set_title(f"(a) Ground-truth facies (inline {i_mid})", fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0, 0], ticks=classes, fraction=0.045, pad=0.03)
    cbar.ax.set_yticklabels(short_names, fontsize=7.5)

    manifold_panels = (
        (
            axes[0, 1],
            um_naive,
            (
                "(b) Naive NOFE-8D (depth=2, iso)\n"
                f"BalAcc={res3d.naive_fac.lr_bal_acc:.3f}, "
                f"MacroF1={res3d.naive_fac.lr_macro_f1:.3f}"
            ),
        ),
        (
            axes[0, 2],
            um_tuned,
            (
                f"(c) Tuned NOFE-{res3d.hero_dim}D (depth=1, aniso-Z)\n"
                f"BalAcc={res3d.tuned_fac_hero.lr_bal_acc:.3f}, "
                f"MacroF1={res3d.tuned_fac_hero.lr_macro_f1:.3f}"
            ),
        ),
    )
    for ax, um, title in manifold_panels:
        for c, sname in zip(classes, short_names, strict=True):
            m = y_bal == c
            ax.scatter(
                um[m, 0],
                um[m, 1],
                s=8,
                alpha=0.65,
                color=FACIES_COLORS[c % len(FACIES_COLORS)],
                label=sname,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(fontsize=7.5, markerscale=1.8, loc="best")

    _draw_confusion(
        axes[1, 0],
        res3d.naive_fac.confusion_matrix,
        short_names,
        "(d) Naive NOFE-8D confusion matrix",
    )
    _draw_confusion(
        axes[1, 1],
        res3d.tuned_fac_hero.confusion_matrix,
        short_names,
        f"(e) Tuned NOFE-{res3d.hero_dim}D confusion matrix",
    )

    pca_f1 = res3d.pca_ref_f1
    x_pos = np.arange(len(classes))
    w = 0.27
    axes[1, 2].bar(
        x_pos - w,
        [res3d.naive_fac.per_class_f1.get(int(c), 0.0) for c in classes],
        w,
        label="Naive NOFE-8D",
        color="#d95f02",
        alpha=0.85,
    )
    axes[1, 2].bar(
        x_pos,
        [pca_f1.get(int(c), 0.0) for c in classes],
        w,
        label="PCA-8D (tex7+z)",
        color="#762a83",
        alpha=0.85,
    )
    axes[1, 2].bar(
        x_pos + w,
        [res3d.tuned_fac_hero.per_class_f1.get(int(c), 0.0) for c in classes],
        w,
        label=f"Tuned NOFE-{res3d.hero_dim}D",
        color="#1b7837",
        alpha=0.95,
    )
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].set_ylabel("Per-formation F1 score")
    axes[1, 2].set_title(
        "(f) Per-class separation gain across F3 units", fontweight="bold"
    )
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        "Facies separation: anisotropic 1-hop texture graphs vs naive NOFE",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
