"""Benchmark, generalisation, ablation and clustering helpers for 3D experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from nofe import NOFE
from nofe.affinity import per_query_kl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch_geometric.data import Data
from umap import UMAP

from nofe_seismic.baselines import run_tsne
from nofe_seismic.data.volumes import (
    LazySeismicVolume,
    PointCloud,
    SeismicVolume,
    volume_to_points,
)
from nofe_seismic.graph import (
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    coord_map_3d,
    embed_graph,
    embed_numpy,
    fit_and_embed_nofe,
    graph_edge_index,
    graph_loss,
    graph_num_nodes,
    graph_summary,
)
from nofe_seismic.metrics import (
    evaluate_clustering,
    local_graph_metrics,
    neighborhood_metrics,
    norm_rmse,
)
from nofe_seismic.types import NofeHyperparams, RowDict, Stride3D, TrainedNofe
from nofe_seismic.viz.slices import (
    SliceFrame,
    plot_cluster_slices,
    plot_kl_residual_slice,
    plot_slice_comparison,
)
from nofe_seismic.viz.summary import (
    plot_ablation_summary,
    plot_benchmark_bars,
    plot_training_curves,
)

_NEIGHBOR_K = 15
_BENCHMARK_SAMPLE = 3000
_SLICE_SAMPLE = 2500
_SILHOUETTE_SAMPLE = 5000
_BAR_METHOD_COUNT = 4
_LARGE_INLINE = 600
_LARGE_CROSSLINE = 900
_LARGE_DEPTH = 200


@dataclass(frozen=True)
class ExperimentConfig:
    out_dir: Path = Path("outputs/synthetic")
    device: str = "cpu"
    seed: int = 42
    train_stride: tuple[int, int, int] = Stride3D(2, 2, 2)
    train_sample: int | None = None
    slice_stride: tuple[int, int, int] = Stride3D(1, 1, 1)
    slab_half_width: int = 2
    train_inline_frac: float = 0.75
    radius: float = 0.25
    width: int = 8
    ker_width: int = 32
    depth: int = 3
    out_dim: int = 3
    nystrom: float | None = 0.25
    chunk_size: int = 8192
    epochs: int = 120
    patience: int = 20
    lr: float = 1e-2
    eval_k: int | None = None
    include_dense_ablation: bool = True


@dataclass(frozen=True)
class SliceEvaluation:
    frame: SliceFrame
    amplitude: np.ndarray
    labels: np.ndarray | None
    x_scaled: np.ndarray
    nofe: np.ndarray
    pca: np.ndarray
    umap: np.ndarray
    kl_per_node: np.ndarray
    mean_kl: float
    graph_stats: dict[str, float | int]


@dataclass(frozen=True)
class AblationSpec:
    tag: str
    radius: float
    depth: int
    width: int
    nystrom: float | None
    out_dim: int


@dataclass(frozen=True)
class BenchmarkResult:
    nofe_main: TrainedNofe
    nofe_8d: TrainedNofe
    pca_model: PCA
    umap_model: UMAP
    emb_pca_train: np.ndarray
    emb_umap_train: np.ndarray
    benchmark_rows: list[RowDict]


@dataclass(frozen=True)
class SliceGeneralizationResult:
    rows: list[RowDict]
    slices: dict[str, SliceEvaluation]
    inline_heldout_idx: int


@dataclass(frozen=True)
class ClusteringSweepResult:
    train_rows: list[RowDict]
    heldout_rows: list[RowDict]


@dataclass(frozen=True)
class _SliceRequest:
    key: str
    axis: Literal["inline", "crossline", "depth"]
    index: int
    title: str


def _nofe_params(
    cfg: ExperimentConfig,
    *,
    out_dim: int | None = None,
    width: int | None = None,
    depth: int | None = None,
    nystrom: float | None = None,
    use_cfg_nystrom: bool = True,
) -> NofeHyperparams:
    return NofeHyperparams(
        out_dim=cfg.out_dim if out_dim is None else out_dim,
        width=cfg.width if width is None else width,
        ker_width=cfg.ker_width,
        depth=cfg.depth if depth is None else depth,
        nystrom=cfg.nystrom if use_cfg_nystrom else nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
    )


def _eval_embedding_row(
    method: str,
    features: np.ndarray,
    embedding: np.ndarray,
    edge_index: torch.Tensor,
    *,
    elapsed_sec: float,
    kl_loss: float | None = None,
    labels: np.ndarray | None = None,
    eval_k: int = 5,
    seed: int = 42,
) -> RowDict:
    nm = neighborhood_metrics(
        features, embedding, k=_NEIGHBOR_K, sample_size=_BENCHMARK_SAMPLE, seed=seed
    )
    lm = local_graph_metrics(features, embedding, edge_index)
    row: RowDict = {
        "method": method,
        "elapsed_sec": round(elapsed_sec, 3),
        "kl_loss": round(kl_loss, 6) if kl_loss is not None else None,
        "trustworthiness": round(nm["trustworthiness"], 4),
        "continuity": round(nm["continuity"], 4),
        "knn_recall": round(nm["knn_recall"], 4),
        "global_stress": round(nm["global_stress"], 4),
        "local_stress": round(lm["local_stress"], 4),
        "relative_edge_variation": round(lm["relative_edge_variation"], 4),
    }
    if labels is not None:
        clust = evaluate_clustering(
            embedding, k_values=(eval_k,), labels=labels, seed=seed
        )[0]
        row[f"silhouette_k{eval_k}"] = round(float(clust["silhouette"]), 4)
        row[f"ari_k{eval_k}"] = round(float(clust.get("ari", np.nan)), 4)
        row[f"nmi_k{eval_k}"] = round(float(clust.get("nmi", np.nan)), 4)
    return row


def run_main_benchmark(
    train_pts: PointCloud,
    train_graph: Data,
    x_train_s: np.ndarray,
    *,
    eval_k: int,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> BenchmarkResult:
    """Train NOFE-3D, NOFE-8D, PCA, UMAP and t-SNE and score training metrics."""
    nofe_main = fit_and_embed_nofe(train_graph, _nofe_params(cfg))
    nofe_8d = fit_and_embed_nofe(train_graph, _nofe_params(cfg, out_dim=8))

    t0 = time.perf_counter()
    pca_model = PCA(n_components=cfg.out_dim, random_state=cfg.seed)
    emb_pca_train = pca_model.fit_transform(x_train_s).astype(np.float32)
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    umap_model = UMAP(
        n_components=cfg.out_dim, n_neighbors=15, min_dist=0.1, random_state=cfg.seed
    )
    emb_umap_train = umap_model.fit_transform(x_train_s).astype(np.float32)
    t_umap = time.perf_counter() - t0

    tsne_res = run_tsne(x_train_s, out_dim=cfg.out_dim, max_nodes=8000, seed=cfg.seed)
    edge_index = graph_edge_index(train_graph)

    method_specs: list[tuple[str, np.ndarray, float, float | None]] = [
        (
            f"NOFE-{cfg.out_dim}D",
            nofe_main.train_embedding,
            nofe_main.elapsed_sec,
            graph_loss(nofe_main.model, train_graph),
        ),
        (
            "NOFE-8D",
            nofe_8d.train_embedding,
            nofe_8d.elapsed_sec,
            graph_loss(nofe_8d.model, train_graph),
        ),
        (f"PCA-{cfg.out_dim}D", emb_pca_train, t_pca, None),
        (f"UMAP-{cfg.out_dim}D", emb_umap_train, t_umap, None),
    ]
    benchmark_rows = [
        _eval_embedding_row(
            name,
            x_train_s,
            emb,
            edge_index,
            elapsed_sec=dt,
            kl_loss=kl,
            labels=train_pts.labels,
            eval_k=eval_k,
            seed=cfg.seed,
        )
        for name, emb, dt, kl in method_specs
    ]
    tsne_nm = neighborhood_metrics(
        x_train_s[tsne_res.indices],
        tsne_res.embedding,
        k=_NEIGHBOR_K,
        sample_size=_BENCHMARK_SAMPLE,
        seed=cfg.seed,
    )
    benchmark_rows.append(
        {
            "method": f"t-SNE-{cfg.out_dim}D (N={len(tsne_res.indices)})",
            "elapsed_sec": round(tsne_res.elapsed_sec, 3),
            "kl_loss": None,
            "trustworthiness": round(tsne_nm["trustworthiness"], 4),
            "continuity": round(tsne_nm["continuity"], 4),
            "knn_recall": round(tsne_nm["knn_recall"], 4),
            "global_stress": round(tsne_nm["global_stress"], 4),
            "local_stress": float("nan"),
            "relative_edge_variation": float("nan"),
        }
    )
    plot_benchmark_bars(
        benchmark_rows[:_BAR_METHOD_COUNT], fig_dir / "benchmark_bars.png"
    )
    return BenchmarkResult(
        nofe_main=nofe_main,
        nofe_8d=nofe_8d,
        pca_model=pca_model,
        umap_model=umap_model,
        emb_pca_train=emb_pca_train,
        emb_umap_train=emb_umap_train,
        benchmark_rows=benchmark_rows,
    )


def _embed_slice_in_context(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    bench: BenchmarkResult,
    req: _SliceRequest,
    cfg: ExperimentConfig,
) -> SliceEvaluation:
    ni, nx, nz = volume.shape
    si, sx, sz = cfg.slice_stride
    hw = cfg.slab_half_width
    if req.axis == "inline":
        pts = volume_to_points(
            volume,
            inline_range=(max(0, req.index - hw), min(ni, req.index + hw + 1)),
            stride=(1, sx, sz),
        )
        mask = pts.locations[:, 0] == float(req.index)
        coords_2d = pts.locations[mask][:, [1, 2]]
        xlabel, ylabel = "Crossline", "Depth / Time Sample"
    elif req.axis == "crossline":
        pts = volume_to_points(
            volume,
            crossline_range=(max(0, req.index - hw), min(nx, req.index + hw + 1)),
            stride=(si, 1, sz),
        )
        mask = pts.locations[:, 1] == float(req.index)
        coords_2d = pts.locations[mask][:, [0, 2]]
        xlabel, ylabel = "Inline", "Depth / Time Sample"
    else:
        pts = volume_to_points(
            volume,
            depth_range=(max(0, req.index - hw), min(nz, req.index + hw + 1)),
            stride=(si, sx, 1),
        )
        mask = pts.locations[:, 2] == float(req.index)
        coords_2d = pts.locations[mask][:, [0, 1]]
        xlabel, ylabel = "Inline", "Crossline"

    builder = SeismicPointGraph(mode="radius", radius=cfg.radius, normalize=False)
    scaled = build_scaled_graph(builder, scaler, pts.features, pts.locations)
    emb_slab = embed_graph(bench.nofe_main.model, scaled.graph)
    kl_nodes, valid = per_query_kl(
        None,
        emb_slab,
        graph_edge_index(scaled.graph),
        num_nodes=graph_num_nodes(scaled.graph),
        weights=scaled.graph.affinity,
    )
    kl_slice = kl_nodes.numpy()[mask]
    valid_slice = valid.numpy()[mask]
    mean_kl = (
        float(kl_slice[valid_slice].mean()) if np.any(valid_slice) else float("nan")
    )
    x_slice = scaled.features[mask]
    return SliceEvaluation(
        frame=SliceFrame(
            coords=coords_2d,
            xlabel=xlabel,
            ylabel=ylabel,
            invert_y=(req.axis != "depth"),
        ),
        amplitude=pts.amplitude[mask],
        labels=pts.labels[mask] if pts.labels is not None else None,
        x_scaled=x_slice,
        nofe=emb_slab.numpy()[mask],
        pca=bench.pca_model.transform(x_slice).astype(np.float32),
        umap=bench.umap_model.transform(x_slice).astype(np.float32),
        kl_per_node=kl_slice,
        mean_kl=mean_kl,
        graph_stats=graph_summary(scaled.graph),
    )


def run_slice_generalization(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    bench: BenchmarkResult,
    *,
    split_i: int,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> SliceGeneralizationResult:
    """Evaluate and plot 2D slices in 3D context slabs."""
    ni, nx, nz = volume.shape
    inline_train_idx = 300 if ni >= _LARGE_INLINE else split_i // 2
    inline_heldout_idx = (
        540
        if ni >= _LARGE_INLINE
        else min(ni - 3, split_i + max(2, (ni - split_i) // 2))
    )
    crossline_idx = 450 if nx >= _LARGE_CROSSLINE else nx // 3
    depth_idx = 160 if nz >= _LARGE_DEPTH else int(0.42 * nz)

    reqs = (
        _SliceRequest(
            "inline_train",
            "inline",
            inline_train_idx,
            f"Training Region Inline {inline_train_idx}",
        ),
        _SliceRequest(
            "inline_heldout",
            "inline",
            inline_heldout_idx,
            f"Held-Out Unseen Inline {inline_heldout_idx}",
        ),
        _SliceRequest(
            "crossline", "crossline", crossline_idx, f"Crossline {crossline_idx}"
        ),
        _SliceRequest("depth", "depth", depth_idx, f"Time/Depth Slice {depth_idx}"),
    )

    rows: list[RowDict] = []
    slices: dict[str, SliceEvaluation] = {}
    for req in reqs:
        res = _embed_slice_in_context(volume, scaler, bench, req, cfg)
        slices[req.key] = res
        plot_slice_comparison(
            res.frame,
            res.amplitude,
            {"NOFE": res.nofe, "PCA": res.pca, "UMAP": res.umap},
            title=req.title,
            labels=res.labels,
            out_path=fig_dir / f"slice_{req.key}.png",
        )
        plot_kl_residual_slice(
            res.frame,
            res.amplitude,
            res.kl_per_node,
            title=f"Local Affinity KL Residuals - {req.title}",
            labels=res.labels,
            out_path=fig_dir / f"kl_residual_{req.key}.png",
        )
        for m_name, emb_m in (
            ("NOFE", res.nofe),
            ("PCA", res.pca),
            ("UMAP", res.umap),
        ):
            nm = neighborhood_metrics(
                res.x_scaled,
                emb_m,
                k=_NEIGHBOR_K,
                sample_size=_SLICE_SAMPLE,
                seed=cfg.seed,
            )
            rows.append(
                {
                    "slice": req.key,
                    "method": m_name,
                    "nodes_on_slice": int(res.x_scaled.shape[0]),
                    "nofe_mean_kl": round(res.mean_kl, 6) if m_name == "NOFE" else None,
                    "trustworthiness": round(nm["trustworthiness"], 4),
                    "continuity": round(nm["continuity"], 4),
                    "global_stress": round(nm["global_stress"], 4),
                }
            )
    return SliceGeneralizationResult(
        rows=rows, slices=slices, inline_heldout_idx=inline_heldout_idx
    )


def run_patch_stitching(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe_main: NOFE,
    *,
    split_i: int,
    cfg: ExperimentConfig,
) -> dict[str, float | int]:
    """Compare overlapping 3D half-slab embeddings for NOFE vs independent UMAP."""
    ni, nx, _ = volume.shape
    sx = cfg.train_stride[1]
    ov_lo = (int(0.35 * nx) // sx) * sx
    ov_hi = max(ov_lo + sx, int(0.65 * nx))
    pts_full = volume_to_points(
        volume, inline_range=(split_i, ni), stride=cfg.train_stride
    )
    pts_left = volume_to_points(
        volume,
        inline_range=(split_i, ni),
        crossline_range=(0, ov_hi),
        stride=cfg.train_stride,
    )
    pts_right = volume_to_points(
        volume,
        inline_range=(split_i, ni),
        crossline_range=(ov_lo, nx),
        stride=cfg.train_stride,
    )
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius, normalize=False)
    g_full = build_scaled_graph(builder, scaler, pts_full.features, pts_full.locations)
    g_left = build_scaled_graph(builder, scaler, pts_left.features, pts_left.locations)
    g_right = build_scaled_graph(
        builder, scaler, pts_right.features, pts_right.locations
    )

    emb_full = embed_numpy(nofe_main, g_full.graph)
    emb_left = embed_numpy(nofe_main, g_left.graph)
    emb_right = embed_numpy(nofe_main, g_right.graph)

    margin_1hop = max(4, round(cfg.radius * nx / np.sqrt(12)))
    inner_lo, inner_hi = ov_lo + margin_1hop, ov_hi - margin_1hop
    map_full = coord_map_3d(pts_full.locations)
    map_left = coord_map_3d(pts_left.locations)
    map_right = coord_map_3d(pts_right.locations)
    shared = [
        k
        for k in map_left
        if k in map_right and k in map_full and inner_lo <= k[1] <= inner_hi
    ] or [k for k in map_left if k in map_right and k in map_full]

    if not shared:
        return {
            "shared_interior_nodes": 0,
            "nofe_patch_stitch_rmse": float("nan"),
            "nofe_patch_stitch_norm_rmse": float("nan"),
            "nofe_vs_full_norm_rmse": float("nan"),
            "umap_independent_stitch_rmse": float("nan"),
            "umap_independent_norm_rmse": float("nan"),
        }

    il = [map_left[k] for k in shared]
    ir = [map_right[k] for k in shared]
    inf = [map_full[k] for k in shared]
    nofe_rmse = float(np.sqrt(np.mean((emb_left[il] - emb_right[ir]) ** 2)))
    u_left = UMAP(n_components=cfg.out_dim, random_state=cfg.seed).fit_transform(
        g_left.features
    )
    u_right = UMAP(n_components=cfg.out_dim, random_state=cfg.seed).fit_transform(
        g_right.features
    )
    umap_rmse = float(np.sqrt(np.mean((u_left[il] - u_right[ir]) ** 2)))
    return {
        "shared_interior_nodes": len(shared),
        "nofe_patch_stitch_rmse": round(nofe_rmse, 6),
        "nofe_patch_stitch_norm_rmse": round(norm_rmse(emb_left[il], emb_right[ir]), 6),
        "nofe_vs_full_norm_rmse": round(norm_rmse(emb_left[il], emb_full[inf]), 6),
        "umap_independent_stitch_rmse": round(umap_rmse, 6),
        "umap_independent_norm_rmse": round(norm_rmse(u_left[il], u_right[ir]), 6),
    }


def run_resolution_consistency(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe_main: NOFE,
    *,
    test_pts_matched: PointCloud,
    emb_test_matched: np.ndarray,
    split_i: int,
    cfg: ExperimentConfig,
) -> dict[str, float | int]:
    """Compare held-out sparse embeddings against a 2x-denser graph."""
    ni, _, _ = volume.shape
    si, sx, sz = cfg.train_stride
    res_stride = Stride3D(max(1, si // 2), max(1, sx // 2), max(1, sz // 2))
    pts_dense = volume_to_points(
        volume,
        inline_range=(split_i, min(ni, split_i + max(6, 2 * si))),
        stride=res_stride,
    )
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius, normalize=False)
    g_dense = build_scaled_graph(
        builder, scaler, pts_dense.features, pts_dense.locations
    )
    emb_dense = embed_numpy(nofe_main, g_dense.graph)
    map_dense = coord_map_3d(pts_dense.locations)
    map_matched = coord_map_3d(test_pts_matched.locations)
    res_keys = [k for k in map_matched if k in map_dense]
    if res_keys:
        idx_m = [map_matched[k] for k in res_keys]
        idx_d = [map_dense[k] for k in res_keys]
        nrmse = norm_rmse(emb_test_matched[idx_m], emb_dense[idx_d])
        res_corr = float(
            np.corrcoef(emb_test_matched[idx_m].ravel(), emb_dense[idx_d].ravel())[0, 1]
        )
    else:
        nrmse, res_corr = float("nan"), float("nan")
    return {
        "shared_nodes": len(res_keys),
        "dense_nodes": graph_num_nodes(g_dense.graph),
        "dense_mean_degree": round(
            float(graph_summary(g_dense.graph)["mean_degree"]), 1
        ),
        "norm_rmse_sparse_vs_dense": round(nrmse, 6),
        "pearson_r_sparse_vs_dense": round(res_corr, 4),
    }


def run_ablations(
    train_pts: PointCloud,
    train_graph: Data,
    x_train_s: np.ndarray,
    scaler: SharedStandardizer,
    *,
    bench: BenchmarkResult,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> list[RowDict]:
    """Run radius, depth, width and Nystrom ablations on the training graph."""
    r_lo = round(0.72 * cfg.radius, 3)
    r_mid = round(cfg.radius, 3)
    r_hi = round(1.40 * cfg.radius, 3)
    last_nys_tag = "nystrom=dense" if cfg.include_dense_ablation else "nystrom=0.50"
    last_nys_val = None if cfg.include_dense_ablation else 0.50

    specs = (
        AblationSpec(f"radius={r_lo}", r_lo, 3, 8, cfg.nystrom, cfg.out_dim),
        AblationSpec(f"radius={r_mid}", r_mid, 3, 8, cfg.nystrom, cfg.out_dim),
        AblationSpec(f"radius={r_hi}", r_hi, 3, 8, cfg.nystrom, cfg.out_dim),
        AblationSpec("depth=2", r_mid, 2, 8, cfg.nystrom, cfg.out_dim),
        AblationSpec("depth=4", r_mid, 4, 8, cfg.nystrom, cfg.out_dim),
        AblationSpec("width=16", r_mid, 3, 16, cfg.nystrom, cfg.out_dim),
        AblationSpec("nystrom=0.10", r_mid, 3, 8, 0.10, cfg.out_dim),
        AblationSpec(last_nys_tag, r_mid, 3, 8, last_nys_val, cfg.out_dim),
    )

    ablation_rows: list[RowDict] = []
    histories = {
        "main_3d": bench.nofe_main.history,
        "main_8d": bench.nofe_8d.history,
    }
    graph_by_radius: dict[float, Data] = {r_mid: train_graph}

    for spec in specs:
        if spec.radius not in graph_by_radius:
            builder = SeismicPointGraph(
                mode="radius", radius=spec.radius, normalize=False
            )
            graph_by_radius[spec.radius] = build_scaled_graph(
                builder, scaler, train_pts.features, train_pts.locations
            ).graph
        g_cur = graph_by_radius[spec.radius]
        trained = fit_and_embed_nofe(
            g_cur,
            _nofe_params(
                cfg,
                out_dim=spec.out_dim,
                width=spec.width,
                depth=spec.depth,
                nystrom=spec.nystrom,
                use_cfg_nystrom=False,
            ),
        )
        kl_a = graph_loss(trained.model, g_cur)
        nm_a = neighborhood_metrics(
            x_train_s,
            trained.train_embedding,
            k=_NEIGHBOR_K,
            sample_size=_SLICE_SAMPLE,
            seed=cfg.seed,
        )
        lm_a = local_graph_metrics(
            x_train_s, trained.train_embedding, graph_edge_index(g_cur)
        )
        histories[spec.tag] = trained.history
        ablation_rows.append(
            {
                "config": spec.tag,
                "epochs_run": len(trained.history),
                "mean_degree": round(float(graph_summary(g_cur)["mean_degree"]), 1),
                "elapsed_sec": round(trained.elapsed_sec, 2),
                "kl_loss": round(kl_a, 6),
                "trustworthiness": round(nm_a["trustworthiness"], 4),
                "continuity": round(nm_a["continuity"], 4),
                "local_stress": round(lm_a["local_stress"], 4),
                "global_stress": round(nm_a["global_stress"], 4),
            }
        )

    plot_training_curves(
        {k: histories[k] for k in ("main_3d", "nystrom=0.10", last_nys_tag, "depth=4")},
        fig_dir / "training_curves.png",
    )
    plot_ablation_summary(ablation_rows, fig_dir / "ablation_summary.png")
    return ablation_rows


def run_clustering_sweep(
    x_train_s: np.ndarray,
    train_labels: np.ndarray | None,
    bench: BenchmarkResult,
    slice_gen: SliceGeneralizationResult,
    *,
    eval_k: int,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> ClusteringSweepResult:
    """Evaluate K-Means clustering on training volumes and held-out inline slices."""
    cluster_inputs = {
        "Raw_11D": x_train_s,
        f"NOFE-{cfg.out_dim}D": bench.nofe_main.train_embedding,
        "NOFE-8D": bench.nofe_8d.train_embedding,
        f"PCA-{cfg.out_dim}D": bench.emb_pca_train,
        f"UMAP-{cfg.out_dim}D": bench.emb_umap_train,
    }
    train_rows: list[RowDict] = [
        {"method": m_name, **crow}
        for m_name, rep in cluster_inputs.items()
        for crow in evaluate_clustering(
            rep, k_values=(3, 4, 5, 6, 8), labels=train_labels, seed=cfg.seed
        )
    ]

    heldout = slice_gen.slices["inline_heldout"]
    slice_clusters: dict[str, np.ndarray] = {}
    heldout_rows: list[RowDict] = []
    for m_name, train_rep, slice_rep in (
        (f"NOFE-{cfg.out_dim}D", bench.nofe_main.train_embedding, heldout.nofe),
        (f"PCA-{cfg.out_dim}D", bench.emb_pca_train, heldout.pca),
        (f"UMAP-{cfg.out_dim}D", bench.emb_umap_train, heldout.umap),
    ):
        pred_slice = (
            KMeans(n_clusters=eval_k, n_init=10, random_state=cfg.seed)
            .fit(train_rep)
            .predict(slice_rep)
        )
        slice_clusters[m_name] = pred_slice
        sil = float(
            silhouette_score(
                slice_rep,
                pred_slice,
                sample_size=min(_SILHOUETTE_SAMPLE, len(slice_rep)),
                random_state=cfg.seed,
            )
        )
        row_hc: RowDict = {"method": m_name, "k": eval_k, "silhouette": round(sil, 4)}
        if heldout.labels is not None:
            valid_lbl = heldout.labels >= 0
            if np.any(valid_lbl):
                row_hc["ari"] = round(
                    float(
                        adjusted_rand_score(
                            heldout.labels[valid_lbl], pred_slice[valid_lbl]
                        )
                    ),
                    4,
                )
                row_hc["nmi"] = round(
                    float(
                        normalized_mutual_info_score(
                            heldout.labels[valid_lbl], pred_slice[valid_lbl]
                        )
                    ),
                    4,
                )
        heldout_rows.append(row_hc)

    plot_cluster_slices(
        heldout.frame,
        slice_clusters,
        title=(
            f"K-Means (K={eval_k}) Facies Clustering on "
            f"Held-Out Inline {slice_gen.inline_heldout_idx}"
        ),
        labels=heldout.labels,
        out_path=fig_dir / "clustering_inline_heldout.png",
    )
    return ClusteringSweepResult(train_rows=train_rows, heldout_rows=heldout_rows)
