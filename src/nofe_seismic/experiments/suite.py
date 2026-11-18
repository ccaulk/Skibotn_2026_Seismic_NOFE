"""End-to-end experiment suite orchestrator for NOFE on 3D seismic data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from nofe import NOFE

from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume, volume_to_points
from nofe_seismic.graph import (
    ScaledGraph,
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    embed_numpy,
    graph_edge_index,
    graph_loss,
    graph_summary,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics
from nofe_seismic.types import IndexRange, RowDict

from .evals import (
    ExperimentConfig,
    run_ablations,
    run_clustering_sweep,
    run_main_benchmark,
    run_patch_stitching,
    run_resolution_consistency,
    run_slice_generalization,
)

_NEIGHBOR_K = 15
_BENCHMARK_SAMPLE = 3000
_MIN_SPLIT_INLINES = 8
_DEFAULT_EVAL_K = 5


def _resolve_eval_k(labels: np.ndarray | None, eval_k: int | None) -> int:
    if eval_k is not None:
        return eval_k
    if labels is not None:
        return len(np.unique(labels[labels >= 0]))
    return _DEFAULT_EVAL_K


def _heldout_matched_metrics(
    bench_model: NOFE,
    g_test: ScaledGraph,
    seed: int,
) -> dict[str, float]:
    emb_test = embed_numpy(bench_model, g_test.graph)
    kl_test = graph_loss(bench_model, g_test.graph)
    test_nm = neighborhood_metrics(
        g_test.features,
        emb_test,
        k=_NEIGHBOR_K,
        sample_size=_BENCHMARK_SAMPLE,
        seed=seed,
    )
    test_lm = local_graph_metrics(
        g_test.features, emb_test, graph_edge_index(g_test.graph)
    )
    return {
        "kl_loss": round(kl_test, 6),
        "trustworthiness": round(test_nm["trustworthiness"], 4),
        "continuity": round(test_nm["continuity"], 4),
        "knn_recall": round(test_nm["knn_recall"], 4),
        "global_stress": round(test_nm["global_stress"], 4),
        "local_stress": round(test_lm["local_stress"], 4),
    }


def _write_experiment_outputs(
    out_dir: Path,
    tables: dict[str, list[RowDict]],
    summary: dict[str, object],
) -> None:
    for fname, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_dir / fname, index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))


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
        volume, inline_range=IndexRange(0, split_i), stride=cfg.train_stride
    )
    test_pts_matched = volume_to_points(
        volume, inline_range=IndexRange(split_i, ni), stride=cfg.train_stride
    )
    eval_k = _resolve_eval_k(train_pts.labels, cfg.eval_k)

    scaler = SharedStandardizer.fit(train_pts.features, train_pts.locations)
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius, normalize=False)
    g_train = build_scaled_graph(
        builder, scaler, train_pts.features, train_pts.locations
    )
    g_test = build_scaled_graph(
        builder, scaler, test_pts_matched.features, test_pts_matched.locations
    )

    bench = run_main_benchmark(
        train_pts,
        g_train.graph,
        g_train.features,
        eval_k=eval_k,
        cfg=cfg,
        fig_dir=fig_dir,
    )
    slice_gen = run_slice_generalization(
        volume, scaler, bench, split_i=split_i, cfg=cfg, fig_dir=fig_dir
    )
    emb_test_matched = embed_numpy(bench.nofe_main.model, g_test.graph)
    matched_metrics = _heldout_matched_metrics(bench.nofe_main.model, g_test, cfg.seed)
    patch_stitching = run_patch_stitching(
        volume, scaler, bench.nofe_main.model, split_i=split_i, cfg=cfg
    )
    resolution_consistency = run_resolution_consistency(
        volume,
        scaler,
        bench.nofe_main.model,
        test_pts_matched=test_pts_matched,
        emb_test_matched=emb_test_matched,
        split_i=split_i,
        cfg=cfg,
    )
    ablation_rows = run_ablations(
        train_pts,
        g_train.graph,
        g_train.features,
        scaler,
        bench=bench,
        cfg=cfg,
        fig_dir=fig_dir,
    )
    clustering = run_clustering_sweep(
        g_train.features,
        train_pts.labels,
        bench,
        slice_gen,
        eval_k=eval_k,
        cfg=cfg,
        fig_dir=fig_dir,
    )

    summary: dict[str, object] = {
        "dataset": volume.name,
        "volume_shape": list(volume.shape),
        "feature_names": list(volume.feature_names),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        },
        "train_graph": graph_summary(g_train.graph),
        "test_matched_graph": graph_summary(g_test.graph),
        "heldout_3d_matched": matched_metrics,
        "patch_stitching": patch_stitching,
        "resolution_consistency": resolution_consistency,
        "heldout_slice_clustering": clustering.heldout_rows,
        "benchmark": bench.benchmark_rows,
        "ablations": ablation_rows,
        "generalization": slice_gen.rows,
        "clustering": clustering.train_rows,
    }
    _write_experiment_outputs(
        out_dir,
        {
            "benchmark.csv": bench.benchmark_rows,
            "ablations.csv": ablation_rows,
            "generalization.csv": slice_gen.rows,
            "clustering.csv": clustering.train_rows,
            "heldout_clustering.csv": clustering.heldout_rows,
        },
        summary,
    )
    return summary
