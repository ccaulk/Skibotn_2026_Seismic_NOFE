"""2D depth-log compression: mapping vertical seismic traces onto a 2D spatial map."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import torch
from nofe.affinity import per_query_kl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch_geometric.data import Data
from umap import UMAP

from nofe_seismic.data.traces import TraceLogs, extract_trace_logs
from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume
from nofe_seismic.graph import (
    ScaledGraph,
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    embed_numpy,
    fit_and_embed_nofe,
    graph_edge_index,
    graph_loss,
    graph_num_nodes,
    graph_summary,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics
from nofe_seismic.types import (
    IndexRange,
    NofeHyperparams,
    RowDict,
    Stride2D,
    TrainedNofe,
)
from nofe_seismic.viz.slices import project_embedding_pca
from nofe_seismic.viz.summary import plot_8d_projection_map

from .plots import (
    DepthMapData,
    plot_depth_diagnostics,
    plot_depth_rgb_maps,
    plot_sample_reconstructions,
)

_MAX_MLP_TRAIN = 25_000
_SALT_FRAC_THRESHOLD = 0.05
_SILHOUETTE_SAMPLE = 5000
_MIN_BINARY_CLASSES = 2
_MAX_VIZ_CLUSTERS = 6
_MIN_SPLIT_INLINES = 6
_NEIGHBOR_K = 15
_EVAL_SAMPLE_SIZE = 3000


class _TimedPca(NamedTuple):
    model: PCA
    elapsed_sec: float


class _TimedUmap(NamedTuple):
    model: UMAP
    elapsed_sec: float


class _DepthEvalOutput(NamedTuple):
    rows: list[RowDict]
    recon_preds_8d: dict[str, np.ndarray]


@dataclass(frozen=True)
class DepthCompressionConfig:
    out_dir: Path = Path("outputs/f3_depth_compression")
    device: str = "cpu"
    seed: int = 42
    stride_2d: tuple[int, int] = Stride2D(3, 3)
    feature_mode: Literal["raw", "envelope"] = "envelope"
    feature_scale: Literal["column", "global"] = "global"
    train_inline_frac: float = 0.75
    radius: float = 0.05
    width: int = 8
    ker_width: int = 32
    depth: int = 2
    width_8d: int = 24
    ker_width_8d: int = 64
    depth_8d: int = 2
    out_dim: int = 3
    nystrom: float | None = 0.15
    chunk_size: int = 8192
    epochs: int = 100
    patience: int = 25
    lr: float = 1e-2


@dataclass(frozen=True)
class _DepthMethodSpec:
    name: str
    z_train: np.ndarray
    z_test: np.ndarray
    elapsed_sec: float
    heldout_kl: float | None


@dataclass(frozen=True)
class _DepthEvalContext:
    x_train: np.ndarray
    x_test: np.ndarray
    test_graph: Data
    train_data: TraceLogs
    test_data: TraceLogs
    num_classes: int
    seed: int


def _subsample_indices(n_samples: int, max_samples: int, seed: int) -> np.ndarray:
    if n_samples <= max_samples:
        return np.arange(n_samples)
    return np.random.default_rng(seed).choice(
        n_samples, size=max_samples, replace=False
    )


def _probe_reconstruction(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    use_mlp: bool = False,
    seed: int = 42,
) -> dict[str, float]:
    ridge = Ridge(alpha=1.0).fit(emb_train, y_train)
    pred_r = ridge.predict(emb_test)
    out = {
        "ridge_r2": round(
            float(r2_score(y_test, pred_r, multioutput="variance_weighted")), 4
        ),
        "ridge_rmse": round(float(np.sqrt(np.mean((y_test - pred_r) ** 2))), 4),
    }
    if not use_mlp:
        return out

    idx = _subsample_indices(emb_train.shape[0], _MAX_MLP_TRAIN, seed)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=100,
        early_stopping=True,
        random_state=seed,
    ).fit(emb_train[idx], y_train[idx])
    pred_m = mlp.predict(emb_test)
    out["mlp_r2"] = round(
        float(r2_score(y_test, pred_m, multioutput="variance_weighted")), 4
    )
    out["mlp_rmse"] = round(float(np.sqrt(np.mean((y_test - pred_m) ** 2))), 4)
    return out


def _fit_timed_pca(x_train: np.ndarray, n_components: int, seed: int) -> _TimedPca:
    t0 = time.perf_counter()
    model = PCA(n_components=n_components, random_state=seed).fit(x_train)
    return _TimedPca(model=model, elapsed_sec=time.perf_counter() - t0)


def _fit_timed_umap(x_train: np.ndarray, n_components: int, seed: int) -> _TimedUmap:
    t0 = time.perf_counter()
    model = UMAP(
        n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=seed
    ).fit(x_train)
    return _TimedUmap(model=model, elapsed_sec=time.perf_counter() - t0)


def _add_label_metrics(
    row: RowDict, spec: _DepthMethodSpec, ctx: _DepthEvalContext
) -> None:
    tr_d, te_d = ctx.train_data, ctx.test_data
    if (
        tr_d.facies_fractions is None
        or te_d.facies_fractions is None
        or tr_d.salt_fraction is None
        or te_d.salt_fraction is None
        or te_d.deepest_facies is None
    ):
        return

    frac_probe = _probe_reconstruction(
        spec.z_train,
        spec.z_test,
        tr_d.facies_fractions,
        te_d.facies_fractions,
        use_mlp=True,
        seed=ctx.seed,
    )
    row["facies_frac_r2"] = frac_probe["mlp_r2"]

    y_salt_tr = (tr_d.salt_fraction > _SALT_FRAC_THRESHOLD).astype(np.int64)
    y_salt_te = (te_d.salt_fraction > _SALT_FRAC_THRESHOLD).astype(np.int64)
    if (
        len(np.unique(y_salt_te)) == _MIN_BINARY_CLASSES
        and len(np.unique(y_salt_tr)) == _MIN_BINARY_CLASSES
    ):
        salt_pred = Ridge(alpha=1.0).fit(spec.z_train, y_salt_tr).predict(spec.z_test)
        row["deep_salt_ridge_auc"] = round(
            float(roc_auc_score(y_salt_te, salt_pred)), 4
        )
        idx_c = _subsample_indices(spec.z_train.shape[0], _MAX_MLP_TRAIN, ctx.seed)
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 32),
            max_iter=80,
            early_stopping=True,
            random_state=ctx.seed,
        ).fit(spec.z_train[idx_c], y_salt_tr[idx_c])
        prob = clf.predict_proba(spec.z_test)[:, 1]
        row["deep_salt_mlp_auc"] = round(float(roc_auc_score(y_salt_te, prob)), 4)

    k_cl = max(3, min(6, ctx.num_classes))
    pred_te = (
        KMeans(n_clusters=k_cl, n_init=10, random_state=ctx.seed)
        .fit(spec.z_train)
        .predict(spec.z_test)
    )
    row["silhouette"] = round(
        float(
            silhouette_score(
                spec.z_test,
                pred_te,
                sample_size=min(_SILHOUETTE_SAMPLE, len(spec.z_test)),
                random_state=ctx.seed,
            )
        ),
        4,
    )
    row["deep_facies_ari"] = round(
        float(adjusted_rand_score(te_d.deepest_facies, pred_te)), 4
    )
    row["deep_facies_nmi"] = round(
        float(normalized_mutual_info_score(te_d.deepest_facies, pred_te)), 4
    )


def _evaluate_depth_suite(
    specs: list[_DepthMethodSpec], ctx: _DepthEvalContext
) -> _DepthEvalOutput:
    rows: list[RowDict] = []
    recon_preds_8d: dict[str, np.ndarray] = {}
    test_ei = graph_edge_index(ctx.test_graph)

    for spec in specs:
        nm = neighborhood_metrics(
            ctx.x_test,
            spec.z_test,
            k=_NEIGHBOR_K,
            sample_size=_EVAL_SAMPLE_SIZE,
            seed=ctx.seed,
        )
        lm = local_graph_metrics(ctx.x_test, spec.z_test, test_ei)
        probe = _probe_reconstruction(
            spec.z_train,
            spec.z_test,
            ctx.x_train,
            ctx.x_test,
            use_mlp=spec.name.startswith(("NOFE", "UMAP")),
            seed=ctx.seed,
        )
        if spec.name in ("NOFE-8D", "PCA-8D", "UMAP-8D"):
            if spec.name.startswith(("NOFE", "UMAP")):
                idx_s = _subsample_indices(
                    spec.z_train.shape[0], _MAX_MLP_TRAIN, ctx.seed
                )
                recon_preds_8d[spec.name] = (
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        max_iter=100,
                        early_stopping=True,
                        random_state=ctx.seed,
                    )
                    .fit(spec.z_train[idx_s], ctx.x_train[idx_s])
                    .predict(spec.z_test)
                )
            else:
                recon_preds_8d[spec.name] = (
                    Ridge(alpha=1.0).fit(spec.z_train, ctx.x_train).predict(spec.z_test)
                )

        row: RowDict = {
            "method": spec.name,
            "train_sec": round(spec.elapsed_sec, 2),
            "heldout_kl": round(spec.heldout_kl, 6)
            if spec.heldout_kl is not None
            else None,
            "trustworthiness": round(nm["trustworthiness"], 4),
            "continuity": round(nm["continuity"], 4),
            "knn_recall": round(nm["knn_recall"], 4),
            "local_stress": round(lm["local_stress"], 4),
            "global_stress": round(nm["global_stress"], 4),
            "depth_recon_ridge_r2": probe["ridge_r2"],
            "depth_recon_mlp_r2": probe.get("mlp_r2", probe["ridge_r2"]),
        }
        _add_label_metrics(row, spec, ctx)
        rows.append(row)

    return _DepthEvalOutput(rows=rows, recon_preds_8d=recon_preds_8d)


@dataclass(frozen=True)
class _FittedDepthModels:
    nofe_main: TrainedNofe
    nofe_8d: TrainedNofe
    pca3: _TimedPca
    pca8: _TimedPca
    umap3: _TimedUmap
    umap8: _TimedUmap


def _nofe_params(
    cfg: DepthCompressionConfig,
    *,
    out_dim: int,
    width: int,
    ker_width: int,
    depth: int,
) -> NofeHyperparams:
    return NofeHyperparams(
        out_dim=out_dim,
        width=width,
        ker_width=ker_width,
        depth=depth,
        nystrom=cfg.nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
    )


def _fit_depth_models(
    g_tr: ScaledGraph, cfg: DepthCompressionConfig
) -> _FittedDepthModels:
    nofe_main = fit_and_embed_nofe(
        g_tr.graph,
        _nofe_params(
            cfg,
            out_dim=cfg.out_dim,
            width=cfg.width,
            ker_width=cfg.ker_width,
            depth=cfg.depth,
        ),
    )
    nofe_8d = fit_and_embed_nofe(
        g_tr.graph,
        _nofe_params(
            cfg,
            out_dim=8,
            width=cfg.width_8d,
            ker_width=cfg.ker_width_8d,
            depth=cfg.depth_8d,
        ),
    )
    return _FittedDepthModels(
        nofe_main=nofe_main,
        nofe_8d=nofe_8d,
        pca3=_fit_timed_pca(g_tr.features, cfg.out_dim, cfg.seed),
        pca8=_fit_timed_pca(g_tr.features, 8, cfg.seed),
        umap3=_fit_timed_umap(g_tr.features, cfg.out_dim, cfg.seed),
        umap8=_fit_timed_umap(g_tr.features, 8, cfg.seed),
    )


def _depth_method_specs(
    models: _FittedDepthModels,
    g_tr: ScaledGraph,
    g_te: ScaledGraph,
    out_dim: int,
) -> list[_DepthMethodSpec]:
    return [
        _DepthMethodSpec(
            f"NOFE-{out_dim}D",
            models.nofe_main.train_embedding,
            embed_numpy(models.nofe_main.model, g_te.graph),
            models.nofe_main.elapsed_sec,
            graph_loss(models.nofe_main.model, g_te.graph),
        ),
        _DepthMethodSpec(
            "NOFE-8D",
            models.nofe_8d.train_embedding,
            embed_numpy(models.nofe_8d.model, g_te.graph),
            models.nofe_8d.elapsed_sec,
            graph_loss(models.nofe_8d.model, g_te.graph),
        ),
        _DepthMethodSpec(
            f"PCA-{out_dim}D",
            models.pca3.model.transform(g_tr.features).astype(np.float32),
            models.pca3.model.transform(g_te.features).astype(np.float32),
            models.pca3.elapsed_sec,
            None,
        ),
        _DepthMethodSpec(
            "PCA-8D",
            models.pca8.model.transform(g_tr.features).astype(np.float32),
            models.pca8.model.transform(g_te.features).astype(np.float32),
            models.pca8.elapsed_sec,
            None,
        ),
        _DepthMethodSpec(
            f"UMAP-{out_dim}D",
            models.umap3.model.transform(g_tr.features).astype(np.float32),
            models.umap3.model.transform(g_te.features).astype(np.float32),
            models.umap3.elapsed_sec,
            None,
        ),
        _DepthMethodSpec(
            "UMAP-8D",
            models.umap8.model.transform(g_tr.features).astype(np.float32),
            models.umap8.model.transform(g_te.features).astype(np.float32),
            models.umap8.elapsed_sec,
            None,
        ),
    ]


def _render_depth_outputs(
    models: _FittedDepthModels,
    graphs: tuple[ScaledGraph, ScaledGraph, ScaledGraph],
    logs: tuple[TraceLogs, TraceLogs],
    eval_out: _DepthEvalOutput,
    *,
    split_i: int,
    cfg: DepthCompressionConfig,
    fig_dir: Path,
) -> list[float]:
    g_tr, g_te, g_full = graphs
    test_data, full_data = logs
    emb_nofe8_full = embed_numpy(models.nofe_8d.model, g_full.graph)
    emb_pca8_tr = models.pca8.model.transform(g_tr.features).astype(np.float32)
    emb_pca8_full = models.pca8.model.transform(g_full.features).astype(np.float32)
    emb_umap8_tr = models.umap8.model.transform(g_tr.features).astype(np.float32)
    emb_umap8_full = models.umap8.model.transform(g_full.features).astype(np.float32)

    nofe8_proj = plot_8d_projection_map(
        emb_nofe8_full,
        full_data.grid_shape,
        full_data.locations,
        fig_dir / "nofe_8d_visualization.png",
        fit_embedding=models.nofe_8d.train_embedding,
        split_inline=float(split_i),
        title="Visualising NOFE-8D Depth Compression: Dims 0-2 vs. PCA-3 & Top PCs",
        seed=cfg.seed,
    )
    kl_nodes, _ = per_query_kl(
        None,
        torch.from_numpy(emb_nofe8_full),
        graph_edge_index(g_full.graph),
        num_nodes=graph_num_nodes(g_full.graph),
        weights=g_full.graph.affinity,
    )
    k_viz = (
        max(3, min(_MAX_VIZ_CLUSTERS, full_data.num_classes))
        if test_data.facies_fractions is not None
        else 5
    )
    clusters_full = {
        name: KMeans(n_clusters=k_viz, n_init=10, random_state=cfg.seed)
        .fit(z_tr)
        .predict(z_full)
        for name, z_tr, z_full in (
            ("NOFE-8D", models.nofe_8d.train_embedding, emb_nofe8_full),
            ("PCA-8D", emb_pca8_tr, emb_pca8_full),
            ("UMAP-8D", emb_umap8_tr, emb_umap8_full),
        )
    }

    map_data = DepthMapData(
        grid_shape=full_data.grid_shape,
        locations=full_data.locations,
        rms_amp=full_data.rms_amp,
        deepest_facies=full_data.deepest_facies,
        salt_fraction=full_data.salt_fraction,
        split_inline=float(split_i),
    )
    plot_depth_rgb_maps(
        map_data,
        {
            f"NOFE-{cfg.out_dim}D (RGB)": embed_numpy(
                models.nofe_main.model, g_full.graph
            ),
            "NOFE-8D (PCA-3 RGB)": nofe8_proj.embedding,
            "PCA-8D (PCA-3 RGB)": project_embedding_pca(
                emb_pca8_full, fit_embedding=emb_pca8_tr, n_components=3, seed=cfg.seed
            ).embedding,
            "UMAP-8D (PCA-3 RGB)": project_embedding_pca(
                emb_umap8_full,
                fit_embedding=emb_umap8_tr,
                n_components=3,
                seed=cfg.seed,
            ).embedding,
        },
        fig_dir,
    )
    plot_depth_diagnostics(map_data, kl_nodes.numpy(), clusters_full, fig_dir)
    plot_sample_reconstructions(
        g_te.features,
        eval_out.recon_preds_8d,
        test_data.locations,
        test_data.deepest_facies,
        fig_dir / "depth_log_reconstructions.png",
    )
    return [round(float(x), 4) for x in nofe8_proj.explained_variance]


def run_depth_compression(
    volume: SeismicVolume | LazySeismicVolume, cfg: DepthCompressionConfig
) -> dict[str, object]:
    """Train and evaluate 2D (inline, crossline) depth-log compression with NOFE."""
    out_dir = cfg.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ni, _, _ = volume.shape
    si = cfg.stride_2d[0]
    split_i = max(
        si,
        (max(_MIN_SPLIT_INLINES, round(ni * cfg.train_inline_frac)) // si) * si,
    )

    train_data = extract_trace_logs(
        volume,
        inline_range=IndexRange(0, split_i),
        stride_2d=cfg.stride_2d,
        feature_mode=cfg.feature_mode,
    )
    test_data = extract_trace_logs(
        volume,
        inline_range=IndexRange(split_i, ni),
        stride_2d=cfg.stride_2d,
        feature_mode=cfg.feature_mode,
    )
    full_data = extract_trace_logs(
        volume, stride_2d=cfg.stride_2d, feature_mode=cfg.feature_mode
    )

    scaler = SharedStandardizer.fit(
        train_data.traces, train_data.locations, feature_scale=cfg.feature_scale
    )
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius, normalize=False)
    g_tr = build_scaled_graph(builder, scaler, train_data.traces, train_data.locations)
    g_te = build_scaled_graph(builder, scaler, test_data.traces, test_data.locations)
    g_full = build_scaled_graph(builder, scaler, full_data.traces, full_data.locations)

    models = _fit_depth_models(g_tr, cfg)
    specs = _depth_method_specs(models, g_tr, g_te, cfg.out_dim)
    eval_out = _evaluate_depth_suite(
        specs,
        _DepthEvalContext(
            x_train=g_tr.features,
            x_test=g_te.features,
            test_graph=g_te.graph,
            train_data=train_data,
            test_data=test_data,
            num_classes=full_data.num_classes,
            seed=cfg.seed,
        ),
    )
    explained_var = _render_depth_outputs(
        models,
        (g_tr, g_te, g_full),
        (test_data, full_data),
        eval_out,
        split_i=split_i,
        cfg=cfg,
        fig_dir=fig_dir,
    )

    pd.DataFrame(eval_out.rows).to_csv(
        out_dir / "depth_compression_metrics.csv", index=False
    )
    summary: dict[str, object] = {
        "dataset": volume.name,
        "volume_shape": list(volume.shape),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        },
        "train_graph": graph_summary(g_tr.graph),
        "full_map_graph": graph_summary(g_full.graph),
        "nofe8_pca3_explained_var": explained_var,
        "history_3d_last": models.nofe_main.history[-1]
        if models.nofe_main.history
        else None,
        "history_8d_last": models.nofe_8d.history[-1]
        if models.nofe_8d.history
        else None,
        "metrics": eval_out.rows,
    }
    (out_dir / "depth_compression_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
