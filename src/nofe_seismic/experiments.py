"""End-to-end experiment suite for NOFE on 3D seismic data."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
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

from .baselines import run_tsne
from .data import LazySeismicVolume, PointCloud, SeismicVolume, volume_to_points
from .graph import (
    SeismicPointGraph,
    SharedStandardizer,
    coord_map_3d,
    embed_graph,
    embed_numpy,
    graph_edge_index,
    graph_loss,
    graph_num_nodes,
    graph_summary,
    train_nofe,
)
from .metrics import (
    evaluate_clustering,
    local_graph_metrics,
    neighborhood_metrics,
    norm_rmse,
)
from .viz import (
    plot_ablation_summary,
    plot_benchmark_bars,
    plot_cluster_slices,
    plot_kl_residual_slice,
    plot_slice_comparison,
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
_MIN_SPLIT_INLINES = 8

MetricRow: TypeAlias = dict[str, float | int | str | None]


@dataclass(frozen=True)
class ExperimentConfig:
    out_dir: Path = Path("outputs/synthetic")
    device: str = "cpu"
    seed: int = 42
    train_stride: tuple[int, int, int] = (2, 2, 2)
    train_sample: int | None = None
    slice_stride: tuple[int, int, int] = (1, 1, 1)
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
    coords_2d: np.ndarray
    xlabel: str
    ylabel: str
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
    nofe_main: NOFE
    pca_model: PCA
    umap_model: UMAP
    hist_main: list[dict[str, float]]
    hist_8d: list[dict[str, float]]
    emb_nofe_train: np.ndarray
    emb_nofe8_train: np.ndarray
    emb_pca_train: np.ndarray
    emb_umap_train: np.ndarray
    benchmark_rows: list[MetricRow]


def _build_scaled_graph(
    pts: PointCloud,
    scaler: SharedStandardizer,
    *,
    mode: Literal["radius", "knn"] = "radius",
    radius: float = 0.25,
    k: int = 16,
    sample: int | None = None,
    seed: int = 0,
) -> tuple[Data, np.ndarray, np.ndarray]:
    x_s = scaler.transform_features(pts.features)
    p_s = scaler.transform_locations(pts.locations)
    graph = SeismicPointGraph(
        mode=mode,
        radius=radius,
        k=k,
        normalize=False,
        sample=sample,
        seed=seed,
    ).build(x_s, p_s)
    return graph, x_s, p_s


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
) -> MetricRow:
    nm = neighborhood_metrics(
        features, embedding, k=_NEIGHBOR_K, sample_size=_BENCHMARK_SAMPLE, seed=seed
    )
    lm = local_graph_metrics(features, embedding, edge_index)
    row: MetricRow = {
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


def _embed_slice_in_context(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    model: NOFE,
    pca: PCA,
    umap_model: UMAP,
    *,
    axis: Literal["inline", "crossline", "depth"],
    index: int,
    radius: float,
    slab_half_width: int = 2,
    slice_stride: tuple[int, int, int] = (1, 1, 1),
) -> SliceEvaluation:
    """Embed a 2D slice (inside a thin 3D context slab) with frozen models."""
    ni, nx, nz = volume.shape
    si, sx, sz = slice_stride
    if axis == "inline":
        lo, hi = max(0, index - slab_half_width), min(ni, index + slab_half_width + 1)
        pts = volume_to_points(volume, inline_range=(lo, hi), stride=(1, sx, sz))
        mask = pts.locations[:, 0] == float(index)
        coords_2d = pts.locations[mask][:, [1, 2]]
        xlabel, ylabel = "Crossline", "Depth / Time Sample"
    elif axis == "crossline":
        lo, hi = max(0, index - slab_half_width), min(nx, index + slab_half_width + 1)
        pts = volume_to_points(volume, crossline_range=(lo, hi), stride=(si, 1, sz))
        mask = pts.locations[:, 1] == float(index)
        coords_2d = pts.locations[mask][:, [0, 2]]
        xlabel, ylabel = "Inline", "Depth / Time Sample"
    elif axis == "depth":
        lo, hi = max(0, index - slab_half_width), min(nz, index + slab_half_width + 1)
        pts = volume_to_points(volume, depth_range=(lo, hi), stride=(si, sx, 1))
        mask = pts.locations[:, 2] == float(index)
        coords_2d = pts.locations[mask][:, [0, 1]]
        xlabel, ylabel = "Inline", "Crossline"
    else:
        raise ValueError(f"Unsupported axis {axis!r}")

    graph, x_s, _ = _build_scaled_graph(pts, scaler, mode="radius", radius=radius)
    emb_slab = embed_graph(model, graph)
    kl_nodes, valid = per_query_kl(
        None,
        emb_slab,
        graph_edge_index(graph),
        num_nodes=graph_num_nodes(graph),
        weights=graph.affinity,
    )
    emb_nofe = emb_slab.numpy()[mask]
    kl_slice = kl_nodes.numpy()[mask]
    valid_slice = valid.numpy()[mask]
    mean_kl = (
        float(kl_slice[valid_slice].mean()) if np.any(valid_slice) else float("nan")
    )

    x_slice = x_s[mask]
    return SliceEvaluation(
        coords_2d=coords_2d,
        xlabel=xlabel,
        ylabel=ylabel,
        amplitude=pts.amplitude[mask],
        labels=pts.labels[mask] if pts.labels is not None else None,
        x_scaled=x_slice,
        nofe=emb_nofe,
        pca=pca.transform(x_slice).astype(np.float32),
        umap=umap_model.transform(x_slice).astype(np.float32),
        kl_per_node=kl_slice,
        mean_kl=mean_kl,
        graph_stats=graph_summary(graph),
    )


def _train_nofe_model(
    graph: Data,
    *,
    out_dim: int,
    width: int,
    ker_width: int,
    depth: int,
    nystrom: float | None,
    cfg: ExperimentConfig,
) -> tuple[NOFE, list[dict[str, float]], np.ndarray, float]:
    t0 = time.perf_counter()
    model = NOFE(
        out_dim=out_dim,
        width=width,
        ker_width=ker_width,
        depth=depth,
        nystrom=nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
    )
    hist = train_nofe(
        model,
        graph,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
        seed=cfg.seed,
    )
    emb = embed_numpy(model, graph)
    return model, hist, emb, time.perf_counter() - t0


def _run_main_benchmark(
    train_pts: PointCloud,
    train_graph: Data,
    x_train_s: np.ndarray,
    *,
    eval_k: int,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> BenchmarkResult:
    nofe_main, hist_main, emb_nofe_train, t_nofe = _train_nofe_model(
        train_graph,
        out_dim=cfg.out_dim,
        width=cfg.width,
        ker_width=cfg.ker_width,
        depth=cfg.depth,
        nystrom=cfg.nystrom,
        cfg=cfg,
    )
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
    nofe_8d, hist_8d, emb_nofe8_train, t_nofe8 = _train_nofe_model(
        train_graph,
        out_dim=8,
        width=cfg.width,
        ker_width=cfg.ker_width,
        depth=cfg.depth,
        nystrom=cfg.nystrom,
        cfg=cfg,
    )

    edge_index = graph_edge_index(train_graph)
    benchmark_rows = [
        _eval_embedding_row(
            f"NOFE-{cfg.out_dim}D",
            x_train_s,
            emb_nofe_train,
            edge_index,
            elapsed_sec=t_nofe,
            kl_loss=graph_loss(nofe_main, train_graph),
            labels=train_pts.labels,
            eval_k=eval_k,
            seed=cfg.seed,
        ),
        _eval_embedding_row(
            "NOFE-8D",
            x_train_s,
            emb_nofe8_train,
            edge_index,
            elapsed_sec=t_nofe8,
            kl_loss=graph_loss(nofe_8d, train_graph),
            labels=train_pts.labels,
            eval_k=eval_k,
            seed=cfg.seed,
        ),
        _eval_embedding_row(
            f"PCA-{cfg.out_dim}D",
            x_train_s,
            emb_pca_train,
            edge_index,
            elapsed_sec=t_pca,
            labels=train_pts.labels,
            eval_k=eval_k,
            seed=cfg.seed,
        ),
        _eval_embedding_row(
            f"UMAP-{cfg.out_dim}D",
            x_train_s,
            emb_umap_train,
            edge_index,
            elapsed_sec=t_umap,
            labels=train_pts.labels,
            eval_k=eval_k,
            seed=cfg.seed,
        ),
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
        pca_model=pca_model,
        umap_model=umap_model,
        hist_main=hist_main,
        hist_8d=hist_8d,
        emb_nofe_train=emb_nofe_train,
        emb_nofe8_train=emb_nofe8_train,
        emb_pca_train=emb_pca_train,
        emb_umap_train=emb_umap_train,
        benchmark_rows=benchmark_rows,
    )


def _run_slice_generalization(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe_main: NOFE,
    pca_model: PCA,
    umap_model: UMAP,
    *,
    split_i: int,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> tuple[list[MetricRow], dict[str, SliceEvaluation], int]:
    ni, nx, nz = volume.shape
    inline_train_idx = 300 if ni >= _LARGE_INLINE else split_i // 2
    inline_heldout_idx = (
        540
        if ni >= _LARGE_INLINE
        else min(ni - 3, split_i + max(2, (ni - split_i) // 2))
    )
    crossline_idx = 450 if nx >= _LARGE_CROSSLINE else nx // 3
    depth_idx = 160 if nz >= _LARGE_DEPTH else int(0.42 * nz)

    slice_specs: tuple[
        tuple[str, Literal["inline", "crossline", "depth"], int, str], ...
    ] = (
        (
            "inline_train",
            "inline",
            inline_train_idx,
            f"Training Region Inline {inline_train_idx}",
        ),
        (
            "inline_heldout",
            "inline",
            inline_heldout_idx,
            f"Held-Out Unseen Inline {inline_heldout_idx}",
        ),
        ("crossline", "crossline", crossline_idx, f"Crossline {crossline_idx}"),
        ("depth", "depth", depth_idx, f"Time/Depth Slice {depth_idx}"),
    )

    generalization_rows: list[MetricRow] = []
    slice_cache: dict[str, SliceEvaluation] = {}
    for key, axis, idx, title in slice_specs:
        res = _embed_slice_in_context(
            volume,
            scaler,
            nofe_main,
            pca_model,
            umap_model,
            axis=axis,
            index=idx,
            radius=cfg.radius,
            slab_half_width=cfg.slab_half_width,
            slice_stride=cfg.slice_stride,
        )
        slice_cache[key] = res
        plot_slice_comparison(
            res.coords_2d,
            res.amplitude,
            {"NOFE": res.nofe, "PCA": res.pca, "UMAP": res.umap},
            title=title,
            xlabel=res.xlabel,
            ylabel=res.ylabel,
            labels=res.labels,
            invert_y=(axis != "depth"),
            out_path=fig_dir / f"slice_{key}.png",
        )
        plot_kl_residual_slice(
            res.coords_2d,
            res.amplitude,
            res.kl_per_node,
            title=f"Local Affinity KL Residuals - {title}",
            xlabel=res.xlabel,
            ylabel=res.ylabel,
            labels=res.labels,
            invert_y=(axis != "depth"),
            out_path=fig_dir / f"kl_residual_{key}.png",
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
            generalization_rows.append(
                {
                    "slice": key,
                    "method": m_name,
                    "nodes_on_slice": int(res.x_scaled.shape[0]),
                    "nofe_mean_kl": round(res.mean_kl, 6) if m_name == "NOFE" else None,
                    "trustworthiness": round(nm["trustworthiness"], 4),
                    "continuity": round(nm["continuity"], 4),
                    "global_stress": round(nm["global_stress"], 4),
                }
            )
    return generalization_rows, slice_cache, inline_heldout_idx


def _run_patch_stitching(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe_main: NOFE,
    *,
    split_i: int,
    cfg: ExperimentConfig,
) -> dict[str, float | int]:
    ni, nx, _ = volume.shape
    sx = cfg.train_stride[1]
    ov_lo = (int(0.35 * nx) // sx) * sx
    ov_hi = max(ov_lo + sx, int(0.65 * nx))
    pts_full_slab = volume_to_points(
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
    g_full, _, _ = _build_scaled_graph(
        pts_full_slab, scaler, mode="radius", radius=cfg.radius
    )
    g_left, x_left_s, _ = _build_scaled_graph(
        pts_left, scaler, mode="radius", radius=cfg.radius
    )
    g_right, x_right_s, _ = _build_scaled_graph(
        pts_right, scaler, mode="radius", radius=cfg.radius
    )

    emb_full_slab = embed_numpy(nofe_main, g_full)
    emb_left = embed_numpy(nofe_main, g_left)
    emb_right = embed_numpy(nofe_main, g_right)

    margin_1hop = max(4, round(cfg.radius * nx / np.sqrt(12)))
    inner_lo, inner_hi = ov_lo + margin_1hop, ov_hi - margin_1hop
    map_full = coord_map_3d(pts_full_slab.locations)
    map_left = coord_map_3d(pts_left.locations)
    map_right = coord_map_3d(pts_right.locations)
    shared_keys = [
        k
        for k in map_left
        if k in map_right and k in map_full and inner_lo <= k[1] <= inner_hi
    ]
    if not shared_keys:
        shared_keys = [k for k in map_left if k in map_right and k in map_full]

    if not shared_keys:
        return {
            "shared_interior_nodes": 0,
            "nofe_patch_stitch_rmse": float("nan"),
            "nofe_patch_stitch_norm_rmse": float("nan"),
            "nofe_vs_full_norm_rmse": float("nan"),
            "umap_independent_stitch_rmse": float("nan"),
            "umap_independent_norm_rmse": float("nan"),
        }

    idx_l = [map_left[k] for k in shared_keys]
    idx_r = [map_right[k] for k in shared_keys]
    idx_f = [map_full[k] for k in shared_keys]
    nofe_stitch_rmse = float(
        np.sqrt(np.mean((emb_left[idx_l] - emb_right[idx_r]) ** 2))
    )
    nofe_vs_full_rmse = float(
        np.sqrt(np.mean((emb_left[idx_l] - emb_full_slab[idx_f]) ** 2))
    )
    emb_scale = float(np.std(emb_full_slab[idx_f])) or 1.0

    u_left = UMAP(n_components=cfg.out_dim, random_state=cfg.seed).fit_transform(
        x_left_s
    )
    u_right = UMAP(n_components=cfg.out_dim, random_state=cfg.seed).fit_transform(
        x_right_s
    )
    umap_stitch_rmse = float(np.sqrt(np.mean((u_left[idx_l] - u_right[idx_r]) ** 2)))
    umap_scale = float(np.std(u_left[idx_l])) or 1.0
    return {
        "shared_interior_nodes": len(shared_keys),
        "nofe_patch_stitch_rmse": round(nofe_stitch_rmse, 6),
        "nofe_patch_stitch_norm_rmse": round(nofe_stitch_rmse / emb_scale, 6),
        "nofe_vs_full_norm_rmse": round(nofe_vs_full_rmse / emb_scale, 6),
        "umap_independent_stitch_rmse": round(umap_stitch_rmse, 6),
        "umap_independent_norm_rmse": round(umap_stitch_rmse / umap_scale, 6),
    }


def _run_resolution_consistency(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe_main: NOFE,
    test_pts_matched: PointCloud,
    emb_test_matched: np.ndarray,
    *,
    split_i: int,
    cfg: ExperimentConfig,
) -> dict[str, float | int]:
    ni, _, _ = volume.shape
    si, sx, sz = cfg.train_stride
    res_stride = (max(1, si // 2), max(1, sx // 2), max(1, sz // 2))
    pts_dense = volume_to_points(
        volume,
        inline_range=(split_i, min(ni, split_i + max(6, 2 * si))),
        stride=res_stride,
    )
    g_dense, _, _ = _build_scaled_graph(
        pts_dense, scaler, mode="radius", radius=cfg.radius
    )
    emb_dense = embed_numpy(nofe_main, g_dense)
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
        "dense_nodes": graph_num_nodes(g_dense),
        "dense_mean_degree": round(float(graph_summary(g_dense)["mean_degree"]), 1),
        "norm_rmse_sparse_vs_dense": round(nrmse, 6),
        "pearson_r_sparse_vs_dense": round(res_corr, 4),
    }


def _run_ablations(
    train_pts: PointCloud,
    train_graph: Data,
    x_train_s: np.ndarray,
    scaler: SharedStandardizer,
    *,
    hist_main: list[dict[str, float]],
    hist_8d: list[dict[str, float]],
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> list[MetricRow]:
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

    ablation_rows: list[MetricRow] = []
    ablation_histories: dict[str, list[dict[str, float]]] = {
        "main_3d": hist_main,
        "main_8d": hist_8d,
    }
    graph_by_radius: dict[float, Data] = {r_mid: train_graph}

    for spec in specs:
        if spec.radius not in graph_by_radius:
            g_r, _, _ = _build_scaled_graph(
                train_pts, scaler, mode="radius", radius=spec.radius
            )
            graph_by_radius[spec.radius] = g_r
        g_cur = graph_by_radius[spec.radius]
        g_stat = graph_summary(g_cur)
        model, hist, emb_a, dt = _train_nofe_model(
            g_cur,
            out_dim=spec.out_dim,
            width=spec.width,
            ker_width=cfg.ker_width,
            depth=spec.depth,
            nystrom=spec.nystrom,
            cfg=cfg,
        )
        kl_a = graph_loss(model, g_cur)
        nm_a = neighborhood_metrics(
            x_train_s, emb_a, k=_NEIGHBOR_K, sample_size=_SLICE_SAMPLE, seed=cfg.seed
        )
        lm_a = local_graph_metrics(x_train_s, emb_a, graph_edge_index(g_cur))
        ablation_histories[spec.tag] = hist
        ablation_rows.append(
            {
                "config": spec.tag,
                "epochs_run": len(hist),
                "mean_degree": round(float(g_stat["mean_degree"]), 1),
                "elapsed_sec": round(dt, 2),
                "kl_loss": round(kl_a, 6),
                "trustworthiness": round(nm_a["trustworthiness"], 4),
                "continuity": round(nm_a["continuity"], 4),
                "local_stress": round(lm_a["local_stress"], 4),
                "global_stress": round(nm_a["global_stress"], 4),
            }
        )

    plot_training_curves(
        {
            k: ablation_histories[k]
            for k in ("main_3d", "nystrom=0.10", last_nys_tag, "depth=4")
        },
        fig_dir / "training_curves.png",
    )
    plot_ablation_summary(ablation_rows, fig_dir / "ablation_summary.png")
    return ablation_rows


def _run_clustering_sweep(
    x_train_s: np.ndarray,
    train_labels: np.ndarray | None,
    heldout: SliceEvaluation,
    *,
    emb_nofe_train: np.ndarray,
    emb_nofe8_train: np.ndarray,
    emb_pca_train: np.ndarray,
    emb_umap_train: np.ndarray,
    eval_k: int,
    inline_heldout_idx: int,
    cfg: ExperimentConfig,
    fig_dir: Path,
) -> tuple[list[MetricRow], list[MetricRow]]:
    cluster_inputs = {
        "Raw_11D": x_train_s,
        f"NOFE-{cfg.out_dim}D": emb_nofe_train,
        "NOFE-8D": emb_nofe8_train,
        f"PCA-{cfg.out_dim}D": emb_pca_train,
        f"UMAP-{cfg.out_dim}D": emb_umap_train,
    }
    clustering_rows: list[MetricRow] = [
        {"method": m_name, **crow}
        for m_name, rep in cluster_inputs.items()
        for crow in evaluate_clustering(
            rep, k_values=(3, 4, 5, 6, 8), labels=train_labels, seed=cfg.seed
        )
    ]

    slice_clusters: dict[str, np.ndarray] = {}
    heldout_rows: list[MetricRow] = []
    for m_name, train_rep, slice_rep in (
        (f"NOFE-{cfg.out_dim}D", emb_nofe_train, heldout.nofe),
        (f"PCA-{cfg.out_dim}D", emb_pca_train, heldout.pca),
        (f"UMAP-{cfg.out_dim}D", emb_umap_train, heldout.umap),
    ):
        km = KMeans(n_clusters=eval_k, n_init=10, random_state=cfg.seed).fit(train_rep)
        pred_slice = km.predict(slice_rep)
        slice_clusters[m_name] = pred_slice
        sil = float(
            silhouette_score(
                slice_rep,
                pred_slice,
                sample_size=min(_SILHOUETTE_SAMPLE, len(slice_rep)),
                random_state=cfg.seed,
            )
        )
        row_hc: MetricRow = {"method": m_name, "k": eval_k, "silhouette": round(sil, 4)}
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
        heldout.coords_2d,
        slice_clusters,
        title=(
            f"K-Means (K={eval_k}) Facies Clustering on "
            f"Held-Out Inline {inline_heldout_idx}"
        ),
        xlabel=heldout.xlabel,
        ylabel=heldout.ylabel,
        labels=heldout.labels,
        invert_y=True,
        out_path=fig_dir / "clustering_inline_heldout.png",
    )
    return clustering_rows, heldout_rows


def run_all_experiments(
    volume: SeismicVolume | LazySeismicVolume, cfg: ExperimentConfig
) -> dict[str, object]:
    """Run baseline benchmark, ablations, generalisation and clustering sweep."""
    out_dir = cfg.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ni, _, _ = volume.shape
    split_i = max(_MIN_SPLIT_INLINES, round(ni * cfg.train_inline_frac))
    train_pts = volume_to_points(
        volume, inline_range=(0, split_i), stride=cfg.train_stride
    )
    test_pts_matched = volume_to_points(
        volume, inline_range=(split_i, ni), stride=cfg.train_stride
    )

    if cfg.eval_k is not None:
        eval_k = cfg.eval_k
    elif train_pts.labels is not None:
        eval_k = len(np.unique(train_pts.labels[train_pts.labels >= 0]))
    else:
        eval_k = 5

    scaler = SharedStandardizer.fit(train_pts.features, train_pts.locations)
    train_graph, x_train_s, _ = _build_scaled_graph(
        train_pts, scaler, mode="radius", radius=cfg.radius
    )
    test_graph_matched, x_test_s, _ = _build_scaled_graph(
        test_pts_matched, scaler, mode="radius", radius=cfg.radius
    )

    bench = _run_main_benchmark(
        train_pts,
        train_graph,
        x_train_s,
        eval_k=eval_k,
        cfg=cfg,
        fig_dir=fig_dir,
    )

    generalization_rows, slice_cache, inline_heldout_idx = _run_slice_generalization(
        volume,
        scaler,
        bench.nofe_main,
        bench.pca_model,
        bench.umap_model,
        split_i=split_i,
        cfg=cfg,
        fig_dir=fig_dir,
    )

    emb_test_matched = embed_numpy(bench.nofe_main, test_graph_matched)
    kl_test_matched = graph_loss(bench.nofe_main, test_graph_matched)
    test_matched_nm = neighborhood_metrics(
        x_test_s,
        emb_test_matched,
        k=_NEIGHBOR_K,
        sample_size=_BENCHMARK_SAMPLE,
        seed=cfg.seed,
    )
    test_matched_lm = local_graph_metrics(
        x_test_s, emb_test_matched, graph_edge_index(test_graph_matched)
    )

    patch_stitching = _run_patch_stitching(
        volume, scaler, bench.nofe_main, split_i=split_i, cfg=cfg
    )
    resolution_consistency = _run_resolution_consistency(
        volume,
        scaler,
        bench.nofe_main,
        test_pts_matched,
        emb_test_matched,
        split_i=split_i,
        cfg=cfg,
    )

    ablation_rows = _run_ablations(
        train_pts,
        train_graph,
        x_train_s,
        scaler,
        hist_main=bench.hist_main,
        hist_8d=bench.hist_8d,
        cfg=cfg,
        fig_dir=fig_dir,
    )
    clustering_rows, heldout_clustering_rows = _run_clustering_sweep(
        x_train_s,
        train_pts.labels,
        slice_cache["inline_heldout"],
        emb_nofe_train=bench.emb_nofe_train,
        emb_nofe8_train=bench.emb_nofe8_train,
        emb_pca_train=bench.emb_pca_train,
        emb_umap_train=bench.emb_umap_train,
        eval_k=eval_k,
        inline_heldout_idx=inline_heldout_idx,
        cfg=cfg,
        fig_dir=fig_dir,
    )

    benchmark_rows = bench.benchmark_rows
    pd.DataFrame(benchmark_rows).to_csv(out_dir / "benchmark.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "ablations.csv", index=False)
    pd.DataFrame(generalization_rows).to_csv(
        out_dir / "generalization.csv", index=False
    )
    pd.DataFrame(clustering_rows).to_csv(out_dir / "clustering.csv", index=False)
    pd.DataFrame(heldout_clustering_rows).to_csv(
        out_dir / "heldout_clustering.csv", index=False
    )

    summary: dict[str, object] = {
        "dataset": volume.name,
        "volume_shape": list(volume.shape),
        "feature_names": list(volume.feature_names),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        },
        "train_graph": graph_summary(train_graph),
        "test_matched_graph": graph_summary(test_graph_matched),
        "heldout_3d_matched": {
            "kl_loss": round(kl_test_matched, 6),
            "trustworthiness": round(test_matched_nm["trustworthiness"], 4),
            "continuity": round(test_matched_nm["continuity"], 4),
            "knn_recall": round(test_matched_nm["knn_recall"], 4),
            "global_stress": round(test_matched_nm["global_stress"], 4),
            "local_stress": round(test_matched_lm["local_stress"], 4),
        },
        "patch_stitching": patch_stitching,
        "resolution_consistency": resolution_consistency,
        "heldout_slice_clustering": heldout_clustering_rows,
        "benchmark": benchmark_rows,
        "ablations": ablation_rows,
        "generalization": generalization_rows,
        "clustering": clustering_rows,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
