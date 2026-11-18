"""Harder stress-test evaluations and visualisations for NOFE on F3 data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nofe import NOFE
from nofe.affinity import per_query_kl
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from umap import UMAP

from nofe_seismic.data.traces import TraceLogs, extract_trace_logs
from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume
from nofe_seismic.graph import (
    ScaledGraph,
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    coord_map_2d,
    embed_numpy,
    fit_and_embed_nofe,
    graph_edge_index,
    graph_loss,
    graph_num_nodes,
    graph_summary,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics, norm_rmse
from nofe_seismic.types import NofeHyperparams, Stride2D
from nofe_seismic.viz.slices import embedding_to_rgb, project_embedding_pca

from .cube_3d import run_3d_cube_and_manifold_viz
from .plots import (
    DensityRow,
    NoiseRow,
    plot_fig_a_radar_and_pareto,
    plot_fig_b_4tile_mosaic,
    plot_fig_c_noise_and_density,
    plot_fig_e_native_salt_zoom,
)

_NOISE_VIZ_SIGMA = 0.50
_NOISE_TOL = 1e-6
_NEIGHBOR_K = 15
_NOISE_EVAL_SAMPLE = 2000
_MIN_BINARY_CLASSES = 2
_MAX_NATIVE_ZOOM_TRACES = 150_000
_DENSE_ZOOM_RADIUS_THRESHOLD = 80_000
_LARGE_INLINE_ZOOM = 500
_LARGE_XLINE_ZOOM = 850
_RGB_CHANNELS = 3


@dataclass(frozen=True)
class _NoiseStressResult:
    rows: list[NoiseRow]
    viz_snapshot: dict[str, np.ndarray]


@dataclass(frozen=True)
class _MosaicResult:
    seam_errors: dict[str, float]
    images: dict[str, np.ndarray]
    mid_i: int
    mid_x: int


@dataclass(frozen=True)
class _SaltZoomResult:
    zoom: TraceLogs
    zz_nofe: np.ndarray
    grad_nofe: np.ndarray
    kl_zoom: np.ndarray
    boundary_auc: dict[str, float]
    nodes: int


@dataclass(frozen=True)
class _FittedModels2D:
    nofe8: NOFE
    z_nofe_tr: np.ndarray
    z_nofe_te: np.ndarray
    pca8: PCA
    z_pca_te: np.ndarray
    z_umap_te: np.ndarray
    train_sec: float


def _spatial_gradient_magnitude(
    emb: np.ndarray, grid_shape: tuple[int, int]
) -> np.ndarray:
    gi, gx = grid_shape
    grid = emb.reshape(gi, gx, emb.shape[1])
    di = np.asarray(np.gradient(grid, axis=0), dtype=np.float32)
    dx = np.asarray(np.gradient(grid, axis=1), dtype=np.float32)
    return np.sqrt(np.sum(di * di + dx * dx, axis=2)).astype(np.float32)


def _true_boundary_mask(labels_2d: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(labels_2d, dtype=bool)
    edge[:-1, :] |= labels_2d[:-1, :] != labels_2d[1:, :]
    edge[1:, :] |= labels_2d[:-1, :] != labels_2d[1:, :]
    edge[:, :-1] |= labels_2d[:, :-1] != labels_2d[:, 1:]
    edge[:, 1:] |= labels_2d[:, :-1] != labels_2d[:, 1:]
    return edge


def _fit_2d_models(
    g_tr: ScaledGraph,
    g_te: ScaledGraph,
    *,
    device: str,
    seed: int,
    epochs_2d: int,
) -> _FittedModels2D:
    trained = fit_and_embed_nofe(
        g_tr.graph,
        NofeHyperparams(
            out_dim=8,
            width=24,
            ker_width=64,
            depth=2,
            nystrom=0.10,
            chunk_size=8192,
            device=device,
            seed=seed,
            epochs=epochs_2d,
            patience=25,
            lr=1e-2,
        ),
    )
    pca8 = PCA(n_components=8, random_state=seed).fit(g_tr.features)
    umap8 = UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=seed).fit(
        g_tr.features
    )
    return _FittedModels2D(
        nofe8=trained.model,
        z_nofe_tr=trained.train_embedding,
        z_nofe_te=embed_numpy(trained.model, g_te.graph),
        pca8=pca8,
        z_pca_te=pca8.transform(g_te.features).astype(np.float32),
        z_umap_te=umap8.transform(g_te.features).astype(np.float32),
        train_sec=trained.elapsed_sec,
    )


def _run_noise_stress_test(
    g_te: ScaledGraph,
    models: _FittedModels2D,
    umap8: UMAP,
    *,
    radius_2d: float,
    seed: int,
) -> _NoiseStressResult:
    rng = np.random.default_rng(seed)
    noise_rows: list[NoiseRow] = []
    viz_snapshot: dict[str, np.ndarray] = {}
    te_ei = graph_edge_index(g_te.graph)
    builder = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False)

    for sigma in (0.0, 0.25, 0.50, 1.00):
        x_noisy = (
            g_te.features
            if sigma == 0.0
            else (
                g_te.features + rng.normal(scale=sigma, size=g_te.features.shape)
            ).astype(np.float32)
        )
        g_noisy = builder.build(x_noisy, g_te.locations)
        zn = embed_numpy(models.nofe8, g_noisy)
        zp = models.pca8.transform(x_noisy).astype(np.float32)
        zu = umap8.transform(x_noisy).astype(np.float32)

        for m_name, z_cur, z_clean in (
            ("NOFE-8D", zn, models.z_nofe_te),
            ("PCA-8D", zp, models.z_pca_te),
            ("UMAP-8D", zu, models.z_umap_te),
        ):
            nm = neighborhood_metrics(
                g_te.features,
                z_cur,
                k=_NEIGHBOR_K,
                sample_size=_NOISE_EVAL_SAMPLE,
                seed=seed,
            )
            lm = local_graph_metrics(g_te.features, z_cur, te_ei)
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
                    clean_target_trustworthiness=round(nm["trustworthiness"], 4),
                )
            )
            if abs(sigma - _NOISE_VIZ_SIGMA) < _NOISE_TOL:
                viz_snapshot[m_name] = z_cur
        if abs(sigma - _NOISE_VIZ_SIGMA) < _NOISE_TOL:
            viz_snapshot["noisy_rms"] = np.sqrt(np.mean(x_noisy * x_noisy, axis=1))
    return _NoiseStressResult(rows=noise_rows, viz_snapshot=viz_snapshot)


def _run_mosaic_stitching(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    models: _FittedModels2D,
    full: TraceLogs,
    *,
    base_stride_2d: Stride2D,
    radius_2d: float,
    seed: int,
) -> _MosaicResult:
    ni, nx, _ = volume.shape
    si, sx = base_stride_2d
    mid_i, mid_x = (ni // 2 // si) * si, (nx // 2 // sx) * sx
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
    ] = {"NOFE-8D": {}, "PCA-8D": {}, "UMAP-8D": {}}
    builder = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False)
    for tname, (ir, xr) in tiles.items():
        td = extract_trace_logs(
            volume,
            inline_range=ir,
            crossline_range=xr,
            stride_2d=base_stride_2d,
            feature_mode="envelope",
        )
        scaled = build_scaled_graph(builder, scaler, td.traces, td.locations)
        cmap = coord_map_2d(td.locations)
        tile_embeds["NOFE-8D"][tname] = (
            cmap,
            embed_numpy(models.nofe8, scaled.graph),
        )
        tile_embeds["PCA-8D"][tname] = (
            cmap,
            PCA(n_components=8, random_state=seed)
            .fit_transform(scaled.features)
            .astype(np.float32),
        )
        tile_embeds["UMAP-8D"][tname] = (
            cmap,
            UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=seed)
            .fit_transform(scaled.features)
            .astype(np.float32),
        )

    pairs = (("NW", "NE"), ("SW", "SE"), ("NW", "SW"), ("NE", "SE"))
    seam_errors: dict[str, float] = {}
    images: dict[str, np.ndarray] = {}
    gi_f, gx_f = full.grid_shape

    for m_name in ("NOFE-8D", "PCA-8D", "UMAP-8D"):
        errs = [
            norm_rmse(
                tile_embeds[m_name][a][1][
                    [tile_embeds[m_name][a][0][k] for k in shared]
                ],
                tile_embeds[m_name][b][1][
                    [tile_embeds[m_name][b][0][k] for k in shared]
                ],
            )
            for a, b in pairs
            if (
                shared := [
                    k
                    for k in tile_embeds[m_name][a][0]
                    if k in tile_embeds[m_name][b][0]
                ]
            )
        ]
        seam_errors[m_name] = round(float(np.mean(errs)), 4)

        mosaic_z = np.zeros((full.locations.shape[0], 8), dtype=np.float32)
        for row_idx, loc in enumerate(full.locations):
            key = (float(loc[0]), float(loc[1]))
            q = ("N" if loc[0] < mid_i else "S") + ("W" if loc[1] < mid_x else "E")
            qmap, qz = tile_embeds[m_name][q]
            mosaic_z[row_idx] = qz[qmap[key]]
        fit_ref = models.z_nofe_tr if m_name == "NOFE-8D" else None
        mosaic_rgb = embedding_to_rgb(
            project_embedding_pca(
                mosaic_z, fit_embedding=fit_ref, n_components=3, seed=seed
            ).embedding
        )
        images[m_name] = mosaic_rgb.reshape(gi_f, gx_f, _RGB_CHANNELS)

    return _MosaicResult(
        seam_errors=seam_errors, images=images, mid_i=mid_i, mid_x=mid_x
    )


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
        stride_2d=Stride2D(1, 1),
        feature_mode="envelope",
    )
    gn = build_scaled_graph(builder, scaler, nat.traces, nat.locations).graph
    zn_nat = embed_numpy(nofe8, gn)
    mnat = coord_map_2d(nat.locations)

    density_rows: list[DensityRow] = []
    regimes = (
        ("stride 6x6", Stride2D(6, 6), 0.0),
        ("stride 4x4", Stride2D(4, 4), 0.0),
        ("stride 3x3 + 50% drop", Stride2D(3, 3), 0.50),
        ("stride 3x3", Stride2D(3, 3), 0.0),
        ("stride 2x2", Stride2D(2, 2), 0.0),
        ("native 1x1", Stride2D(1, 1), 0.0),
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
            scaled = build_scaled_graph(builder, scaler, d_pts.traces, d_pts.locations)
            if drop_frac > 0.0:
                n_pts = scaled.features.shape[0]
                keep_n = min(n_pts, max(4, round(n_pts * (1.0 - drop_frac))))
                keep_idx = np.sort(rng.choice(n_pts, size=keep_n, replace=False))
                g_cur = builder.build(
                    scaled.features[keep_idx], scaled.locations[keep_idx]
                )
                cur_locs = d_pts.locations[keep_idx]
            else:
                g_cur, cur_locs = scaled.graph, d_pts.locations
            z_cur = embed_numpy(nofe8, g_cur)

        m_cur = coord_map_2d(cur_locs)
        shared_d = [k for k in m_cur if k in mnat]
        ic, in_ = [m_cur[k] for k in shared_d], [mnat[k] for k in shared_d]
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
    models: _FittedModels2D,
    *,
    radius_2d: float,
) -> _SaltZoomResult:
    ni, nx, _ = volume.shape
    zi0, zi1 = (150, 450) if ni >= _LARGE_INLINE_ZOOM else (0, ni)
    zx0, zx1 = (350, 800) if nx >= _LARGE_XLINE_ZOOM else (0, nx)
    zoom = extract_trace_logs(
        volume,
        inline_range=(zi0, zi1),
        crossline_range=(zx0, zx1),
        stride_2d=Stride2D(1, 1)
        if (zi1 - zi0) * (zx1 - zx0) <= _MAX_NATIVE_ZOOM_TRACES
        else Stride2D(2, 2),
        feature_mode="envelope",
    )
    zoom_radius = (
        radius_2d * 0.6
        if zoom.traces.shape[0] > _DENSE_ZOOM_RADIUS_THRESHOLD
        else radius_2d
    )
    builder = SeismicPointGraph(mode="radius", radius=zoom_radius, normalize=False)
    gz = build_scaled_graph(builder, scaler, zoom.traces, zoom.locations)
    zz_nofe = embed_numpy(models.nofe8, gz.graph)
    zz_pca = models.pca8.transform(gz.features).astype(np.float32)
    kl_zoom, _ = per_query_kl(
        None,
        torch.from_numpy(zz_nofe),
        graph_edge_index(gz.graph),
        num_nodes=graph_num_nodes(gz.graph),
        weights=gz.graph.affinity,
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
    return _SaltZoomResult(
        zoom=zoom,
        zz_nofe=zz_nofe,
        grad_nofe=grad_nofe,
        kl_zoom=kl_zoom_np,
        boundary_auc=boundary_auc,
        nodes=graph_num_nodes(gz.graph),
    )


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

    stride2 = Stride2D(int(base_stride_2d[0]), int(base_stride_2d[1]))
    ni, _, _ = volume.shape
    split_i = max(stride2.di * 4, (round(ni * 0.75) // stride2.di) * stride2.di)

    tr = extract_trace_logs(
        volume,
        inline_range=(0, split_i),
        stride_2d=stride2,
        feature_mode="envelope",
    )
    te = extract_trace_logs(
        volume,
        inline_range=(split_i, ni),
        stride_2d=stride2,
        feature_mode="envelope",
    )
    full = extract_trace_logs(volume, stride_2d=stride2, feature_mode="envelope")

    sc = SharedStandardizer.fit(tr.traces, tr.locations, feature_scale="global")
    builder = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False)
    g_tr = build_scaled_graph(builder, sc, tr.traces, tr.locations)
    g_te = build_scaled_graph(builder, sc, te.traces, te.locations)

    models = _fit_2d_models(g_tr, g_te, device=device, seed=seed, epochs_2d=epochs_2d)
    umap8 = UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=seed).fit(
        g_tr.features
    )

    noise = _run_noise_stress_test(g_te, models, umap8, radius_2d=radius_2d, seed=seed)
    mosaic = _run_mosaic_stitching(
        volume,
        sc,
        models,
        full,
        base_stride_2d=stride2,
        radius_2d=radius_2d,
        seed=seed,
    )
    density_rows = _run_density_transfer_curve(
        volume, sc, models.nofe8, split_i=split_i, radius_2d=radius_2d, seed=seed
    )
    salt = _run_salt_dome_zoom(volume, sc, models, radius_2d=radius_2d)

    plot_fig_a_radar_and_pareto(
        noise.rows,
        mosaic.seam_errors,
        density_rows,
        fig_dir / "figA_nofe_radar_overview.png",
    )
    plot_fig_b_4tile_mosaic(
        full,
        mosaic.images,
        mosaic.seam_errors,
        mid_i=float(mosaic.mid_i),
        mid_x=float(mosaic.mid_x),
        out_path=fig_dir / "figB_4tile_mosaic_seams.png",
    )
    plot_fig_c_noise_and_density(
        noise.rows,
        density_rows,
        te,
        noisy_viz=noise.viz_snapshot,
        z_nofe_tr=models.z_nofe_tr,
        seed=seed,
        out_path=fig_dir / "figC_noise_and_density_stress.png",
    )
    plot_fig_e_native_salt_zoom(
        salt.zoom,
        salt.zz_nofe,
        models.z_nofe_tr,
        grad_nofe=salt.grad_nofe,
        kl_zoom=salt.kl_zoom,
        boundary_auc=salt.boundary_auc,
        seed=seed,
        out_path=fig_dir / "figE_salt_dome_native_zoom.png",
    )

    cube_metrics: dict[str, float | int] = {}
    if run_3d_cube:
        cube_metrics = run_3d_cube_and_manifold_viz(
            volume,
            device=device,
            seed=seed,
            epochs=epochs_3d,
            out_path=fig_dir / "figD_3d_cube_and_manifold.png",
        )

    noise_dicts = [r.to_dict() for r in noise.rows]
    density_dicts = [r.to_dict() for r in density_rows]
    summary: dict[str, object] = {
        "nofe_2d_train_sec": round(models.train_sec, 2),
        "noise_stress_test": noise_dicts,
        "mosaic_4tile_seam_norm_rmse": mosaic.seam_errors,
        "density_transfer_curve": density_dicts,
        "salt_zoom_nodes": salt.nodes,
        "salt_zoom_boundary_auc": salt.boundary_auc,
        "cube_3d_8d_summary": cube_metrics,
    }
    (out_dir / "hard_eval_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(noise_dicts).to_csv(out_dir / "noise_stress.csv", index=False)
    pd.DataFrame(density_dicts).to_csv(out_dir / "density_curve.csv", index=False)
    return summary
