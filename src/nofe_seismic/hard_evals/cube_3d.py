"""3D cutaway cube and latent manifold visualisation for hard evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from nofe import NOFE

from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume, volume_to_points
from nofe_seismic.graph import (
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    embed_numpy,
    fit_and_embed_nofe,
    graph_loss,
    graph_num_nodes,
)
from nofe_seismic.types import GridShape2D, NofeHyperparams, PlaneSurface, Stride3D
from nofe_seismic.viz.slices import embedding_to_rgb, project_embedding_pca

_LARGE_3D_INLINES = 400
_LARGE_3D_XLINES = 900
_LARGE_3D_DEPTH = 200
_LARGE_INLINE_CUT = 600
_RGB_CHANNELS = 3
_UNIT_STRIDE_3D = Stride3D(1, 1, 1)


@dataclass(frozen=True)
class _PlaneQuery:
    shape_2d: GridShape2D
    inline_range: tuple[int, int] | None = None
    crossline_range: tuple[int, int] | None = None
    depth_range: tuple[int, int] | None = None
    stride: Stride3D = _UNIT_STRIDE_3D


def _embed_plane_rgba(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe3d: NOFE,
    query: _PlaneQuery,
    *,
    z_tr: np.ndarray,
    radius_3d: float,
    seed: int,
) -> PlaneSurface:
    pts = volume_to_points(
        volume,
        inline_range=query.inline_range,
        crossline_range=query.crossline_range,
        depth_range=query.depth_range,
        stride=query.stride,
    )
    builder = SeismicPointGraph(mode="radius", radius=radius_3d, normalize=False)
    scaled = build_scaled_graph(builder, scaler, pts.features, pts.locations)
    zp = embed_numpy(nofe3d, scaled.graph)
    proj = project_embedding_pca(
        zp, fit_embedding=z_tr, n_components=3, seed=seed
    ).embedding
    rgb = embedding_to_rgb(proj).reshape(
        query.shape_2d.nu, query.shape_2d.nv, _RGB_CHANNELS
    )
    rgba = np.concatenate(
        [rgb, np.ones((*query.shape_2d, 1), dtype=np.float32)], axis=2
    )
    return PlaneSurface(locations=pts.locations, rgba=rgba)


def _plot_3d_manifold_panel(
    fig: Figure,
    z_tr: np.ndarray,
    labels: np.ndarray | None,
    *,
    seed: int,
) -> None:
    ax_man = fig.add_subplot(1, 2, 2, projection="3d")
    proj = project_embedding_pca(z_tr, n_components=3, seed=seed)
    z_proj, var_r = proj.embedding, proj.explained_variance
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
    ax_man.set_title("3D NOFE-8D Latent Manifold Coloured by Facies", fontsize=10.5)
    ax_man.legend(loc="upper left", fontsize=8, markerscale=2)


def run_3d_cube_and_manifold_viz(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    device: str,
    seed: int,
    epochs: int,
    out_path: Path,
) -> dict[str, float | int]:
    """Train 3D NOFE-8D and render an orthogonal cube cutaway and 3D manifold."""
    ni, nx, nz = volume.shape
    split_i = max(8, round(ni * 0.75))
    st = Stride3D(14, 22, 6) if ni >= _LARGE_3D_INLINES else Stride3D(2, 2, 2)
    tr_pts = volume_to_points(volume, inline_range=(0, split_i), stride=st)
    scaler = SharedStandardizer.fit(tr_pts.features, tr_pts.locations)
    r3d = 0.16 if ni >= _LARGE_3D_INLINES else 0.25
    builder = SeismicPointGraph(mode="radius", radius=r3d, normalize=False)
    g_tr = build_scaled_graph(builder, scaler, tr_pts.features, tr_pts.locations)

    trained = fit_and_embed_nofe(
        g_tr.graph,
        NofeHyperparams(
            out_dim=8,
            width=16,
            ker_width=48,
            depth=2,
            nystrom=0.10,
            chunk_size=8192,
            device=device,
            seed=seed,
            epochs=epochs,
            patience=20,
            lr=1e-2,
        ),
    )

    i_idx = 300 if ni >= _LARGE_INLINE_CUT else ni // 2
    x_idx = 450 if nx >= _LARGE_3D_XLINES else nx // 2
    z_idx = 160 if nz >= _LARGE_3D_DEPTH else nz // 2
    s_in = Stride3D(1, 6, 3) if ni >= _LARGE_3D_INLINES else Stride3D(1, 1, 1)
    s_xl = Stride3D(4, 1, 3) if ni >= _LARGE_3D_INLINES else Stride3D(1, 1, 1)
    s_dp = Stride3D(4, 6, 1) if ni >= _LARGE_3D_INLINES else Stride3D(1, 1, 1)

    shape_in = GridShape2D(len(range(0, nx, s_in.dx)), len(range(0, nz, s_in.dz)))
    shape_xl = GridShape2D(len(range(0, ni, s_xl.di)), len(range(0, nz, s_in.dz)))
    shape_dp = GridShape2D(len(range(0, ni, s_dp.di)), len(range(0, nx, s_dp.dx)))

    planes = (
        (
            _embed_plane_rgba(
                volume,
                scaler,
                trained.model,
                _PlaneQuery(shape_in, inline_range=(i_idx, i_idx + 1), stride=s_in),
                z_tr=trained.train_embedding,
                radius_3d=r3d,
                seed=seed,
            ),
            shape_in,
        ),
        (
            _embed_plane_rgba(
                volume,
                scaler,
                trained.model,
                _PlaneQuery(shape_xl, crossline_range=(x_idx, x_idx + 1), stride=s_xl),
                z_tr=trained.train_embedding,
                radius_3d=r3d,
                seed=seed,
            ),
            shape_xl,
        ),
        (
            _embed_plane_rgba(
                volume,
                scaler,
                trained.model,
                _PlaneQuery(shape_dp, depth_range=(z_idx, z_idx + 1), stride=s_dp),
                z_tr=trained.train_embedding,
                radius_3d=r3d,
                seed=seed,
            ),
            shape_dp,
        ),
    )

    fig = plt.figure(figsize=(15, 6.2), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for surf, shape_f in planes:
        ax3d.plot_surface(
            surf.locations[:, 0].reshape(*shape_f),
            surf.locations[:, 1].reshape(*shape_f),
            surf.locations[:, 2].reshape(*shape_f),
            facecolors=surf.rgba,
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
    )

    _plot_3d_manifold_panel(fig, trained.train_embedding, tr_pts.labels, seed=seed)

    fig.suptitle(
        "3D Seismic Volume Reduction: Volumetric Cutaway & Latent Manifold", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "train_nodes": graph_num_nodes(g_tr.graph),
        "kl_loss": round(graph_loss(trained.model, g_tr.graph), 6),
    }
