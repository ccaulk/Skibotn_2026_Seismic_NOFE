"""Spatial gallery, 3D cutaway cube and trace-level plots for ablation studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neural_network import MLPRegressor

from nofe_seismic.data.traces import TraceLogs
from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume, volume_to_points
from nofe_seismic.graph import SeismicPointGraph, build_scaled_graph, embed_numpy
from nofe_seismic.types import GridShape2D, PlaneSurface, Stride3D
from nofe_seismic.viz.slices import embedding_to_rgb, project_embedding_pca

from .evals import (
    Ablation2DResult,
    Ablation3DResult,
    AblationFaciesConfig,
    balanced_sample_indices,
    build_3d_texture_features,
    interleaved_inline_mask,
)
from .metrics_plots import (
    F3_FACIES_NAMES,
    FACIES_COLORS,
    facies_cmap_and_norm,
)

_F3_INLINE_THRESHOLD = 400
_F3_CUT_I = 300
_F3_CUT_X = 450
_F3_CUT_Z = 160
_F3_PCA_REF_DIM = 8
_F3_HERO_DIM = 16
_GALLERY_MAP_PANELS = 3
_PCA_RGB_DIMS = 3
_FEATURE_SCALE_TUNED_3D = 1.5
_UNIT_STRIDE_3D = Stride3D(1, 1, 1)


@dataclass(frozen=True)
class _TunedPlaneQuery:
    shape_2d: GridShape2D
    inline_range: tuple[int, int] | None = None
    crossline_range: tuple[int, int] | None = None
    depth_range: tuple[int, int] | None = None
    stride: Stride3D = _UNIT_STRIDE_3D


def _contrast_stretch_rgb(proj: np.ndarray) -> np.ndarray:
    return embedding_to_rgb(proj, low_pct=2.0, high_pct=98.0)


def _plot_gallery_cross_section_row(
    axes_row: np.ndarray,
    volume: SeismicVolume | LazySeismicVolume,
    res3d: Ablation3DResult,
    *,
    seed: int,
) -> None:
    ni, nx, nz = volume.shape
    i_mid = ni // 2
    st_sl = Stride3D(1, 2, 2) if ni >= _F3_INLINE_THRESHOLD else _UNIT_STRIDE_3D
    sl_pts = volume_to_points(volume, inline_range=(i_mid, i_mid + 1), stride=st_sl)
    nx_sl = len(range(0, nx, st_sl.dx))
    nz_sl = len(range(0, nz, st_sl.dz))

    amp_sl = sl_pts.amplitude.reshape(nx_sl, nz_sl).T
    vlim = float(np.percentile(np.abs(amp_sl), 98)) or 1.0
    axes_row[0].imshow(amp_sl, cmap="seismic", vmin=-vlim, vmax=vlim, aspect="auto")
    axes_row[0].set_title(f"(e) Raw seismic inline {i_mid}", fontweight="bold")
    axes_row[0].set_xlabel("Crossline")
    axes_row[0].set_ylabel("Depth sample")

    if sl_pts.labels is not None:
        style = facies_cmap_and_norm(np.unique(sl_pts.labels).astype(int))
        axes_row[1].imshow(
            sl_pts.labels.reshape(nx_sl, nz_sl).T,
            cmap=style.cmap,
            norm=style.norm,
            aspect="auto",
            interpolation="nearest",
        )
        axes_row[1].set_title("(f) True facies cross-section", fontweight="bold")
    axes_row[1].set_xlabel("Crossline")

    builder = SeismicPointGraph(mode="radius", radius=res3d.radius_3d, normalize=False)
    gn_sl = build_scaled_graph(
        builder, res3d.sc_naive, sl_pts.features, sl_pts.locations
    ).graph
    zn_sl = embed_numpy(res3d.naive_model, gn_sl)
    pn_sl = project_embedding_pca(
        zn_sl, fit_embedding=res3d.zn_te, n_components=_PCA_RGB_DIMS, seed=seed
    ).embedding
    axes_row[2].imshow(
        np.transpose(_contrast_stretch_rgb(pn_sl).reshape(nx_sl, nz_sl, 3), (1, 0, 2)),
        aspect="auto",
    )
    axes_row[2].set_title("(g) Naive 3D NOFE-8D (blurred)", fontweight="bold")
    axes_row[2].set_xlabel("Crossline")

    f_tex_sl = build_3d_texture_features(sl_pts, depth_max=nz)
    gt_sl = builder.build(
        (res3d.sc_tuned.transform_features(f_tex_sl) * _FEATURE_SCALE_TUNED_3D).astype(
            np.float32
        ),
        res3d.sc_tuned.transform_locations(sl_pts.locations),
    )
    zt_sl = embed_numpy(res3d.tuned_models[res3d.hero_dim], gt_sl)
    pt_sl = project_embedding_pca(
        zt_sl, fit_embedding=res3d.zt_te_hero, n_components=_PCA_RGB_DIMS, seed=seed
    ).embedding
    axes_row[3].imshow(
        np.transpose(_contrast_stretch_rgb(pt_sl).reshape(nx_sl, nz_sl, 3), (1, 0, 2)),
        aspect="auto",
    )
    axes_row[3].set_title(
        f"(h) Tuned 3D NOFE-{res3d.hero_dim}D (crisp beds)", fontweight="bold"
    )
    axes_row[3].set_xlabel("Crossline")


def plot_fig3_gallery(
    volume: SeismicVolume | LazySeismicVolume,
    ab2d: Ablation2DResult,
    res3d: Ablation3DResult,
    cfg: AblationFaciesConfig,
    out_path: Path,
) -> None:
    """Plot 2x4 multi-scale 2D depth maps (d=3,8,16) and inline cross-sections."""
    gi, gx = ab2d.full2d.grid_shape
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius_2d, normalize=False)
    g_full = build_scaled_graph(
        builder, ab2d.scaler, ab2d.full2d.traces, ab2d.full2d.locations
    ).graph

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)
    if ab2d.full2d.deepest_facies is not None:
        style = facies_cmap_and_norm(np.unique(ab2d.full2d.deepest_facies).astype(int))
        axes[0, 0].imshow(
            ab2d.full2d.deepest_facies.reshape(gi, gx),
            cmap=style.cmap,
            norm=style.norm,
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, 0].set_title("(a) Ground-truth deepest facies", fontweight="bold")
    else:
        axes[0, 0].imshow(
            ab2d.full2d.rms_amp.reshape(gi, gx), cmap="magma", aspect="auto"
        )
        axes[0, 0].set_title("(a) Mean trace RMS energy", fontweight="bold")
    axes[0, 0].set_xlabel("Crossline index")
    axes[0, 0].set_ylabel("Inline index")

    show_dims = [d for d in (3, 8, 16) if d in ab2d.nofe_models]
    while len(show_dims) < _GALLERY_MAP_PANELS:
        show_dims.append(max(ab2d.nofe_models.keys()))
    for idx, (col_idx, d) in enumerate(zip((1, 2, 3), show_dims, strict=True)):
        z_map = embed_numpy(ab2d.nofe_models[d], g_full)
        proj = project_embedding_pca(
            z_map, n_components=_PCA_RGB_DIMS, seed=cfg.seed
        ).embedding
        axes[0, col_idx].imshow(
            _contrast_stretch_rgb(proj).reshape(gi, gx, 3), aspect="auto"
        )
        ratio = ab2d.full2d.traces.shape[1] / float(d)
        tag = chr(ord("b") + idx)
        axes[0, col_idx].set_title(
            f"({tag}) 2D NOFE-{d}D map ({ratio:.1f}x compression)", fontweight="bold"
        )
        axes[0, col_idx].set_xlabel("Crossline index")

    _plot_gallery_cross_section_row(axes[1], volume, res3d, seed=cfg.seed)
    fig.suptitle(
        "Multi-scale 2D depth-log maps & 3D stratigraphic cross-section alignment",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _embed_tuned_plane(
    volume: SeismicVolume | LazySeismicVolume,
    res3d: Ablation3DResult,
    query: _TunedPlaneQuery,
    *,
    seed: int,
) -> PlaneSurface:
    _, _, nz = volume.shape
    pts = volume_to_points(
        volume,
        inline_range=query.inline_range,
        crossline_range=query.crossline_range,
        depth_range=query.depth_range,
        stride=query.stride,
    )
    f_tex = build_3d_texture_features(pts, depth_max=nz)
    gp = SeismicPointGraph(
        mode="radius", radius=res3d.radius_3d, normalize=False
    ).build(
        (res3d.sc_tuned.transform_features(f_tex) * _FEATURE_SCALE_TUNED_3D).astype(
            np.float32
        ),
        res3d.sc_tuned.transform_locations(pts.locations),
    )
    zp = embed_numpy(res3d.tuned_models[res3d.hero_dim], gp)
    proj = project_embedding_pca(
        zp, fit_embedding=res3d.zt_te_hero, n_components=_PCA_RGB_DIMS, seed=seed
    ).embedding
    rgb = _contrast_stretch_rgb(proj).reshape(query.shape_2d.nu, query.shape_2d.nv, 3)
    rgba = np.concatenate(
        [rgb, np.ones((*query.shape_2d, 1), dtype=np.float32)], axis=2
    )
    return PlaneSurface(locations=pts.locations, rgba=rgba)


def plot_fig4_hero_3d_cube(
    volume: SeismicVolume | LazySeismicVolume,
    res3d: Ablation3DResult,
    *,
    seed: int,
    out_path: Path,
) -> None:
    """Render high-contrast 3D chair-cut cube and class-balanced 3D manifold."""
    ni, nx, nz = volume.shape
    is_f3 = ni >= _F3_INLINE_THRESHOLD
    i_idx = _F3_CUT_I if is_f3 else ni // 2
    x_idx = _F3_CUT_X if is_f3 else nx // 2
    z_idx = _F3_CUT_Z if is_f3 else nz // 2
    s_in = Stride3D(1, 6, 3) if is_f3 else _UNIT_STRIDE_3D
    s_xl = Stride3D(4, 1, 3) if is_f3 else _UNIT_STRIDE_3D
    s_dp = Stride3D(4, 6, 1) if is_f3 else _UNIT_STRIDE_3D

    sh_in = GridShape2D(len(range(0, nx, s_in.dx)), len(range(0, nz, s_in.dz)))
    sh_xl = GridShape2D(len(range(0, ni, s_xl.di)), len(range(0, nz, s_in.dz)))
    sh_dp = GridShape2D(len(range(0, ni, s_dp.di)), len(range(0, nx, s_dp.dx)))

    planes = (
        (
            _embed_tuned_plane(
                volume,
                res3d,
                _TunedPlaneQuery(sh_in, inline_range=(i_idx, i_idx + 1), stride=s_in),
                seed=seed,
            ),
            sh_in,
        ),
        (
            _embed_tuned_plane(
                volume,
                res3d,
                _TunedPlaneQuery(
                    sh_xl, crossline_range=(x_idx, x_idx + 1), stride=s_xl
                ),
                seed=seed,
            ),
            sh_xl,
        ),
        (
            _embed_tuned_plane(
                volume,
                res3d,
                _TunedPlaneQuery(sh_dp, depth_range=(z_idx, z_idx + 1), stride=s_dp),
                seed=seed,
            ),
            sh_dp,
        ),
    )

    fig = plt.figure(figsize=(15.8, 6.8), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1.2, 1.0], figure=fig)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    for surf, sh in planes:
        ax3d.plot_surface(
            surf.locations[:, 0].reshape(*sh),
            surf.locations[:, 1].reshape(*sh),
            surf.locations[:, 2].reshape(*sh),
            facecolors=surf.rgba,
            rstride=1,
            cstride=1,
            shade=False,
        )
    ax3d.invert_zaxis()
    ax3d.view_init(elev=25, azim=-50)
    ax3d.set_xlabel("Inline")
    ax3d.set_ylabel("Crossline")
    ax3d.set_zlabel("Depth")
    ax3d.set_title(
        f"Facies-tuned 3D NOFE-{res3d.hero_dim}D orthogonal cutaway\n"
        f"(depth=1, aniso-Z, inline={i_idx}, crossline={x_idx}, depth={z_idx})",
        fontsize=10.5,
        fontweight="bold",
    )

    ax_man = fig.add_subplot(gs[0, 1], projection="3d")
    bal_idx = balanced_sample_indices(res3d.y_te, max_per_class=700, seed=seed)
    proj = project_embedding_pca(
        res3d.zt_te_hero, n_components=_PCA_RGB_DIMS, seed=seed
    )
    z_bal, y_bal = proj.embedding[bal_idx], res3d.y_te[bal_idx]
    for cls in sorted(np.unique(y_bal)):
        m = y_bal == cls
        c_int = int(cls)
        col = FACIES_COLORS[c_int % len(FACIES_COLORS)]
        lbl = (
            F3_FACIES_NAMES[c_int]
            if c_int < len(F3_FACIES_NAMES)
            else f"Facies {c_int}"
        )
        ax_man.scatter(
            z_bal[m, 0], z_bal[m, 1], z_bal[m, 2], s=7, alpha=0.45, color=col, label=lbl
        )
        cen = z_bal[m].mean(axis=0)
        ax_man.scatter(
            [cen[0]],
            [cen[1]],
            [cen[2]],
            s=110,
            marker="*",
            color=col,
            edgecolors="k",
            linewidths=0.8,
        )

    var_r = proj.explained_variance
    tf = res3d.tuned_fac_hero
    ax_man.set_xlabel(f"PC1 ({100 * var_r[0]:.1f}%)")
    ax_man.set_ylabel(f"PC2 ({100 * var_r[1]:.1f}%)")
    ax_man.set_zlabel(f"PC3 ({100 * var_r[2]:.1f}%)")
    ax_man.set_title(
        f"Class-balanced 3D latent manifold (stars = centroids)\n"
        f"BalAcc={tf.lr_bal_acc:.3f} | MacroF1={tf.lr_macro_f1:.3f} | "
        f"7-NN={tf.knn_acc:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_man.legend(loc="upper left", fontsize=7.5, markerscale=1.8)

    fig.suptitle(
        "Facies-aware 3D volumetric compression & geological manifold structure",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _shade_trace_facies(ax: Axes, depth_labels: np.ndarray | None) -> None:
    if depth_labels is None:
        return
    nz = len(depth_labels)
    start = 0
    for z in range(1, nz + 1):
        if z == nz or depth_labels[z] != depth_labels[start]:
            cls = int(depth_labels[start])
            ax.axvspan(
                start,
                z,
                color=FACIES_COLORS[cls % len(FACIES_COLORS)],
                alpha=0.14,
                lw=0,
            )
            start = z


def _predict_full_traces(
    z_all: np.ndarray, x_full: np.ndarray, train_mask: np.ndarray, *, seed: int
) -> np.ndarray:
    return (
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=150,
            early_stopping=True,
            random_state=seed,
        )
        .fit(z_all[train_mask], x_full[train_mask])
        .predict(z_all)
    )


def _plot_fig5_cluster_row(
    fig: Figure,
    axes_row: np.ndarray,
    full2d: TraceLogs,
    z_high: np.ndarray,
    trace_rmse: np.ndarray,
    *,
    d_high: int,
    seed: int,
) -> dict[str, float]:
    gi, gx = full2d.grid_shape
    km_ari, km_nmi = 0.0, 0.0
    if full2d.deepest_facies is not None:
        classes = np.unique(full2d.deepest_facies).astype(int)
        style = facies_cmap_and_norm(classes)
        axes_row[0].imshow(
            full2d.deepest_facies.reshape(gi, gx),
            cmap=style.cmap,
            norm=style.norm,
            aspect="auto",
            interpolation="nearest",
        )
        axes_row[0].set_title("(d) Ground-truth deepest formation", fontweight="bold")
        km = KMeans(n_clusters=max(2, len(classes)), n_init=10, random_state=seed).fit(
            z_high
        )
        km_ari = float(adjusted_rand_score(full2d.deepest_facies, km.labels_))
        km_nmi = float(normalized_mutual_info_score(full2d.deepest_facies, km.labels_))
        axes_row[1].imshow(
            km.labels_.reshape(gi, gx),
            cmap="tab10",
            aspect="auto",
            interpolation="nearest",
        )
        axes_row[1].set_title(
            f"(e) Unsupervised k-means (NOFE-{d_high}D, NMI={km_nmi:.3f})",
            fontweight="bold",
        )
    else:
        axes_row[0].imshow(full2d.rms_amp.reshape(gi, gx), cmap="magma", aspect="auto")
        axes_row[0].set_title("(d) Mean trace RMS energy", fontweight="bold")
        km = KMeans(n_clusters=6, n_init=10, random_state=seed).fit(z_high)
        axes_row[1].imshow(
            km.labels_.reshape(gi, gx),
            cmap="tab10",
            aspect="auto",
            interpolation="nearest",
        )
        axes_row[1].set_title(
            f"(e) Unsupervised k-means (NOFE-{d_high}D)", fontweight="bold"
        )

    axes_row[0].set_xlabel("Crossline index")
    axes_row[0].set_ylabel("Inline index")
    axes_row[1].set_xlabel("Crossline index")

    im_err = axes_row[2].imshow(trace_rmse, cmap="inferno", aspect="auto")
    axes_row[2].set_title(
        f"(f) Per-trace reconstruction RMSE (NOFE-{d_high}D)", fontweight="bold"
    )
    axes_row[2].set_xlabel("Crossline index")
    fig.colorbar(im_err, ax=axes_row[2], fraction=0.045, pad=0.03)
    return {"kmeans_ari": round(km_ari, 4), "kmeans_nmi": round(km_nmi, 4)}


def plot_fig5_trace_profiles_and_clusters(
    volume: SeismicVolume | LazySeismicVolume,
    ab2d: Ablation2DResult,
    cfg: AblationFaciesConfig,
    out_path: Path,
) -> dict[str, float]:
    """Plot reconstructed seismic depth-log profiles and 2D clusters."""
    full2d = ab2d.full2d
    gi, gx = full2d.grid_shape
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius_2d, normalize=False)
    g_full = build_scaled_graph(builder, ab2d.scaler, full2d.traces, full2d.locations)
    m_tr = interleaved_inline_mask(full2d.locations, holdout_mod=4)

    d_low = (
        _F3_PCA_REF_DIM
        if _F3_PCA_REF_DIM in ab2d.nofe_models
        else min(ab2d.nofe_models)
    )
    d_high = _F3_HERO_DIM if _F3_HERO_DIM in ab2d.nofe_models else max(ab2d.nofe_models)
    z_low = embed_numpy(ab2d.nofe_models[d_low], g_full.graph)
    z_high = embed_numpy(ab2d.nofe_models[d_high], g_full.graph)

    pca_d = min(d_low, g_full.features.shape[1], int(m_tr.sum()))
    zp_full = (
        PCA(n_components=pca_d, random_state=cfg.seed)
        .fit(g_full.features[m_tr])
        .transform(g_full.features)
        .astype(np.float32)
    )
    rec_low = _predict_full_traces(z_low, g_full.features, m_tr, seed=cfg.seed)
    rec_high = _predict_full_traces(z_high, g_full.features, m_tr, seed=cfg.seed)
    rec_pca = _predict_full_traces(zp_full, g_full.features, m_tr, seed=cfg.seed)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), constrained_layout=True)
    specs = (
        ("(a) Shelf parallel beds", int(0.18 * gi), int(0.22 * gx)),
        ("(b) Chalk dipping flank", int(0.50 * gi), int(0.42 * gx)),
        ("(c) Zechstein salt diapir", int(0.45 * gi), int(0.62 * gx)),
    )
    z_axis = np.arange(g_full.features.shape[1])
    for ax, (title, ri, rx) in zip(axes[0], specs, strict=True):
        idx = ri * gx + rx
        loc_i, loc_x = int(full2d.locations[idx, 0]), int(full2d.locations[idx, 1])
        lbl_col = (
            np.asarray(volume.labels[loc_i, loc_x, :], dtype=np.int64)
            if volume.labels is not None
            else None
        )
        _shade_trace_facies(ax, lbl_col)
        ax.plot(
            z_axis, g_full.features[idx], color="#111111", lw=1.7, label="True envelope"
        )
        ax.plot(
            z_axis,
            rec_high[idx],
            color="#1b7837",
            lw=1.5,
            label=f"NOFE-{d_high}D ({g_full.features.shape[1] / d_high:.1f}x)",
        )
        ax.plot(
            z_axis,
            rec_low[idx],
            color="#2166ac",
            lw=1.2,
            ls="--",
            label=f"NOFE-{d_low}D ({g_full.features.shape[1] / d_low:.1f}x)",
        )
        ax.plot(
            z_axis,
            rec_pca[idx],
            color="#762a83",
            lw=1.1,
            ls=":",
            label=f"PCA-{pca_d}D ({g_full.features.shape[1] / pca_d:.1f}x)",
        )
        ax.set_title(f"{title} (inline {loc_i}, xline {loc_x})", fontweight="bold")
        ax.set_xlabel("Depth sample")
        ax.set_ylabel("Normalised trace envelope")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7.5, loc="upper right")

    trace_rmse = np.sqrt(np.mean((g_full.features - rec_high) ** 2, axis=1)).reshape(
        gi, gx
    )
    clust_stats = _plot_fig5_cluster_row(
        fig, axes[1], full2d, z_high, trace_rmse, d_high=d_high, seed=cfg.seed
    )
    fig.suptitle(
        "Trace-level depth-log reconstructions & unsupervised facies clustering",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return clust_stats
