"""2D depth-log compression: mapping vertical seismic traces onto a 2D spatial map."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from nofe import NOFE
from nofe.affinity import per_query_kl
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
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

from .data import LazySeismicVolume, SeismicVolume
from .graph import (
    SeismicPointGraph,
    SharedStandardizer,
    embed_numpy,
    graph_edge_index,
    graph_loss,
    graph_num_nodes,
    graph_summary,
    train_nofe,
)
from .metrics import local_graph_metrics, neighborhood_metrics
from .viz import embedding_to_rgb, plot_8d_projection_map, project_embedding_pca

_MAX_MLP_TRAIN = 25_000
_SAMPLE_TRACE_PANELS = 4
_SALT_FRAC_THRESHOLD = 0.05
_SILHOUETTE_SAMPLE = 5000
_MIN_BINARY_CLASSES = 2
_MIN_FACIES_FOR_HIGH_START = 6
_NEIGHBOR_K = 15
_EVAL_SAMPLE_SIZE = 3000

MetricRow: TypeAlias = dict[str, float | int | str | None]


@dataclass(frozen=True)
class DepthCompressionConfig:
    out_dir: Path = Path("outputs/f3_depth_compression")
    device: str = "cpu"
    seed: int = 42
    stride_2d: tuple[int, int] = (3, 3)
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
class TraceLogs:
    """2D survey trace locations, depth-log features and optional facies summaries."""

    traces: np.ndarray
    locations: np.ndarray
    grid_shape: tuple[int, int]
    rms_amp: np.ndarray
    facies_fractions: np.ndarray | None
    dominant_facies: np.ndarray | None
    deepest_facies: np.ndarray | None
    salt_fraction: np.ndarray | None
    num_classes: int


def _subsample_indices(n_samples: int, max_samples: int, seed: int) -> np.ndarray:
    if n_samples <= max_samples:
        return np.arange(n_samples)
    return np.random.default_rng(seed).choice(
        n_samples, size=max_samples, replace=False
    )


def extract_trace_logs(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    inline_range: tuple[int, int] | None = None,
    crossline_range: tuple[int, int] | None = None,
    stride_2d: tuple[int, int] = (1, 1),
    feature_mode: Literal["raw", "envelope"] = "raw",
) -> TraceLogs:
    """Extract 2D ``(inline, crossline)`` trace locations and depth logs ``[N, Z]``."""
    ni, nx, _ = volume.shape
    i0, i1 = inline_range if inline_range is not None else (0, ni)
    x0, x1 = crossline_range if crossline_range is not None else (0, nx)
    si, sx = stride_2d

    sl = (slice(i0, i1, si), slice(x0, x1, sx), slice(None))
    traces_3d = np.asarray(volume.amplitude[sl], dtype=np.float32)
    gi, gx, gz = traces_3d.shape
    traces_raw = traces_3d.reshape(gi * gx, gz)
    if feature_mode == "envelope":
        env = np.abs(hilbert(traces_raw.astype(np.float64), axis=1)).astype(np.float32)
        traces = gaussian_filter1d(env, sigma=2.0, axis=1).astype(np.float32)
    elif feature_mode == "raw":
        traces = traces_raw
    else:
        raise ValueError(f"Unsupported feature_mode {feature_mode!r}")

    i_idx = np.arange(i0, i1, si, dtype=np.float32)
    x_idx = np.arange(x0, x1, sx, dtype=np.float32)
    ii, xx = np.meshgrid(i_idx, x_idx, indexing="ij")
    locations = np.stack([ii.ravel(), xx.ravel()], axis=1).astype(np.float32)
    rms_amp = np.sqrt(np.mean(traces_raw * traces_raw, axis=1)).astype(np.float32)

    if volume.labels is None:
        return TraceLogs(
            traces=traces,
            locations=locations,
            grid_shape=(gi, gx),
            rms_amp=rms_amp,
            facies_fractions=None,
            dominant_facies=None,
            deepest_facies=None,
            salt_fraction=None,
            num_classes=0,
        )

    lbl_2d = np.asarray(volume.labels[sl], dtype=np.int64).reshape(gi * gx, gz)
    num_classes = int(max(int(lbl_2d.max()) + 1, 1))
    fracs = np.stack(
        [(lbl_2d == c).mean(axis=1, dtype=np.float32) for c in range(num_classes)],
        axis=1,
    )
    high_start = (
        4 if num_classes >= _MIN_FACIES_FOR_HIGH_START else max(1, num_classes - 1)
    )
    return TraceLogs(
        traces=traces,
        locations=locations,
        grid_shape=(gi, gx),
        rms_amp=rms_amp,
        facies_fractions=fracs,
        dominant_facies=np.argmax(fracs, axis=1).astype(np.int64),
        deepest_facies=lbl_2d[:, -1].astype(np.int64),
        salt_fraction=fracs[:, high_start:].sum(axis=1).astype(np.float32),
        num_classes=num_classes,
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
    """Measure how much depth-log variance is recoverable from compressed 2D codes."""
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


def _plot_rgb_maps(
    full_data: TraceLogs,
    embeddings: dict[str, np.ndarray],
    *,
    split_inline: float,
    out_dir: Path,
) -> None:
    gi, gx = full_data.grid_shape
    locs = full_data.locations
    extent = (
        float(locs[:, 1].min()),
        float(locs[:, 1].max()),
        float(locs[:, 0].max()),
        float(locs[:, 0].min()),
    )
    has_geo = full_data.salt_fraction is not None
    n_panels = 1 + (1 if has_geo else 0) + len(embeddings)
    fig, raw_axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.4 * n_panels, 4.4),
        sharey=True,
        constrained_layout=True,
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    p = 0
    rms_img = full_data.rms_amp.reshape(gi, gx)
    axes[p].imshow(rms_img, cmap="viridis", extent=extent, aspect="auto")
    axes[p].axhline(split_inline, color="white", linestyle="--", linewidth=1.2)
    axes[p].set_title("Trace RMS Energy")
    axes[p].set_xlabel("Crossline")
    axes[p].set_ylabel("Inline")
    p += 1

    if full_data.deepest_facies is not None:
        deep_img = full_data.deepest_facies.reshape(gi, gx)
        axes[p].imshow(
            deep_img, cmap="tab10", vmin=0, vmax=9, extent=extent, aspect="auto"
        )
        axes[p].axhline(split_inline, color="white", linestyle="--", linewidth=1.2)
        axes[p].set_title("Deepest Facies Province")
        axes[p].set_xlabel("Crossline")
        p += 1

    for name, emb in embeddings.items():
        rgb = embedding_to_rgb(emb).reshape(gi, gx, 3)
        axes[p].imshow(rgb, extent=extent, aspect="auto")
        axes[p].axhline(split_inline, color="white", linestyle="--", linewidth=1.2)
        axes[p].set_title(name)
        axes[p].set_xlabel("Crossline")
        p += 1

    for idx, ax in enumerate(axes):
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if idx > 0:
            ax.tick_params(labelleft=False)

    fig.suptitle(
        "2D Depth-Log Compression Map (dashed line = train/held-out inline split)",
        fontsize=12,
    )
    fig.savefig(out_dir / "depth_compression_maps.png", dpi=180)
    plt.close(fig)


def _plot_kl_and_cluster_maps(
    full_data: TraceLogs,
    kl_map: np.ndarray,
    clusters_by_method: dict[str, np.ndarray],
    *,
    split_inline: float,
    out_dir: Path,
) -> None:
    gi, gx = full_data.grid_shape
    locs = full_data.locations
    extent = (
        float(locs[:, 1].min()),
        float(locs[:, 1].max()),
        float(locs[:, 0].max()),
        float(locs[:, 0].min()),
    )
    n_c = 2 + len(clusters_by_method)
    fig, raw_axes = plt.subplots(
        1,
        n_c,
        figsize=(4.4 * n_c, 4.4),
        sharey=True,
        constrained_layout=True,
    )
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    kl_img = kl_map.reshape(gi, gx)
    vmax = float(np.percentile(kl_img, 98.0)) or 1e-3
    im_kl = axes[0].imshow(
        kl_img, cmap="inferno", vmin=0.0, vmax=vmax, extent=extent, aspect="auto"
    )
    axes[0].axhline(split_inline, color="cyan", linestyle="--", linewidth=1.2)
    axes[0].set_title("Per-Trace KL Residual (NOFE-8D)")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Inline")
    fig.colorbar(im_kl, ax=axes[0], fraction=0.046, pad=0.04)

    if full_data.salt_fraction is not None:
        salt_img = full_data.salt_fraction.reshape(gi, gx)
        im_s = axes[1].imshow(salt_img, cmap="magma", extent=extent, aspect="auto")
        axes[1].set_title("Deep Unit / Salt Fraction")
        axes[1].set_xlabel("Crossline")
        fig.colorbar(im_s, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].imshow(
            full_data.rms_amp.reshape(gi, gx),
            cmap="viridis",
            extent=extent,
            aspect="auto",
        )
        axes[1].set_title("Trace RMS Energy")
        axes[1].set_xlabel("Crossline")

    for idx, (m_name, cl) in enumerate(clusters_by_method.items(), start=2):
        cl_img = cl.reshape(gi, gx)
        axes[idx].imshow(
            cl_img, cmap="tab10", vmin=0, vmax=9, extent=extent, aspect="auto"
        )
        axes[idx].axhline(split_inline, color="white", linestyle="--", linewidth=1.2)
        axes[idx].set_title(f"{m_name} Clusters")
        axes[idx].set_xlabel("Crossline")

    for idx, ax in enumerate(axes):
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if idx > 0:
            ax.tick_params(labelleft=False)

    fig.suptitle("2D Depth-Log KL Residuals & Spatial Province Clustering", fontsize=12)
    fig.savefig(out_dir / "depth_compression_diagnostics.png", dpi=180)
    plt.close(fig)


def _plot_sample_reconstructions(
    traces_true: np.ndarray,
    traces_pred_by_method: dict[str, np.ndarray],
    locations: np.ndarray,
    deepest_facies: np.ndarray | None,
    out_path: Path,
) -> None:
    """Plot example depth logs vs probe reconstructions from compressed 2D codes."""
    rng = np.random.default_rng(0)
    chosen: list[int] = []
    if deepest_facies is not None:
        for cls in np.unique(deepest_facies)[:_SAMPLE_TRACE_PANELS]:
            idxs = np.flatnonzero(deepest_facies == cls)
            if idxs.size > 0:
                chosen.append(int(idxs[len(idxs) // 2]))
    while len(chosen) < _SAMPLE_TRACE_PANELS:
        chosen.append(int(rng.integers(0, traces_true.shape[0])))
    chosen = chosen[:_SAMPLE_TRACE_PANELS]

    z_axis = np.arange(traces_true.shape[1])
    fig, raw_axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    for ax, idx in zip(axes, chosen, strict=True):
        ax.plot(
            z_axis,
            traces_true[idx],
            color="black",
            linewidth=1.2,
            label="True depth log",
        )
        for m_name, pred in traces_pred_by_method.items():
            ax.plot(z_axis, pred[idx], linewidth=1.0, alpha=0.85, label=m_name)
        i_val, x_val = int(locations[idx, 0]), int(locations[idx, 1])
        suffix = (
            f", deep_facies={int(deepest_facies[idx])}"
            if deepest_facies is not None
            else ""
        )
        ax.set_title(f"Trace (inline={i_val}, crossline={x_val}{suffix})", fontsize=10)
        ax.set_xlabel("Depth / Vertical Sample")
        ax.set_ylabel("Std. Amplitude")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Depth-Log Reconstruction Probe from 2D Compressed Embeddings", fontsize=12
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _train_nofe_pair(
    train_graph: Data, cfg: DepthCompressionConfig
) -> tuple[NOFE, float, list[dict[str, float]], NOFE, float, list[dict[str, float]]]:
    t0 = time.perf_counter()
    nofe_main = NOFE(
        out_dim=cfg.out_dim,
        width=cfg.width,
        ker_width=cfg.ker_width,
        depth=cfg.depth,
        nystrom=cfg.nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
    )
    hist_3d = train_nofe(
        nofe_main,
        train_graph,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
        seed=cfg.seed,
    )
    t_nofe3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    nofe_8d = NOFE(
        out_dim=8,
        width=cfg.width_8d,
        ker_width=cfg.ker_width_8d,
        depth=cfg.depth_8d,
        nystrom=cfg.nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
    )
    hist_8d = train_nofe(
        nofe_8d,
        train_graph,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
        seed=cfg.seed,
    )
    t_nofe8 = time.perf_counter() - t0
    return nofe_main, t_nofe3, hist_3d, nofe_8d, t_nofe8, hist_8d


def _reconstruction_pred_8d(
    m_name: str,
    z_tr: np.ndarray,
    z_te: np.ndarray,
    x_train: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if m_name.startswith(("NOFE", "UMAP")):
        idx_s = _subsample_indices(z_tr.shape[0], _MAX_MLP_TRAIN, seed)
        mlp_viz = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=100,
            early_stopping=True,
            random_state=seed,
        ).fit(z_tr[idx_s], x_train[idx_s])
        return mlp_viz.predict(z_te)
    return Ridge(alpha=1.0).fit(z_tr, x_train).predict(z_te)


def _add_label_metrics(
    row: MetricRow,
    z_tr: np.ndarray,
    z_te: np.ndarray,
    train_data: TraceLogs,
    test_data: TraceLogs,
    *,
    num_classes: int,
    seed: int,
) -> None:
    if (
        train_data.facies_fractions is None
        or test_data.facies_fractions is None
        or train_data.salt_fraction is None
        or test_data.salt_fraction is None
        or test_data.deepest_facies is None
    ):
        return

    frac_probe = _probe_reconstruction(
        z_tr,
        z_te,
        train_data.facies_fractions,
        test_data.facies_fractions,
        use_mlp=True,
        seed=seed,
    )
    row["facies_frac_r2"] = frac_probe["mlp_r2"]

    y_salt_tr = (train_data.salt_fraction > _SALT_FRAC_THRESHOLD).astype(np.int64)
    y_salt_te = (test_data.salt_fraction > _SALT_FRAC_THRESHOLD).astype(np.int64)
    if (
        len(np.unique(y_salt_te)) == _MIN_BINARY_CLASSES
        and len(np.unique(y_salt_tr)) == _MIN_BINARY_CLASSES
    ):
        salt_pred = Ridge(alpha=1.0).fit(z_tr, y_salt_tr).predict(z_te)
        row["deep_salt_ridge_auc"] = round(
            float(roc_auc_score(y_salt_te, salt_pred)), 4
        )
        idx_c = _subsample_indices(z_tr.shape[0], _MAX_MLP_TRAIN, seed)
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 32),
            max_iter=80,
            early_stopping=True,
            random_state=seed,
        ).fit(z_tr[idx_c], y_salt_tr[idx_c])
        prob = clf.predict_proba(z_te)[:, 1]
        row["deep_salt_mlp_auc"] = round(float(roc_auc_score(y_salt_te, prob)), 4)

    k_cl = max(3, min(6, num_classes))
    km = KMeans(n_clusters=k_cl, n_init=10, random_state=seed).fit(z_tr)
    pred_te = km.predict(z_te)
    row["silhouette"] = round(
        float(
            silhouette_score(
                z_te,
                pred_te,
                sample_size=min(_SILHOUETTE_SAMPLE, len(z_te)),
                random_state=seed,
            )
        ),
        4,
    )
    row["deep_facies_ari"] = round(
        float(adjusted_rand_score(test_data.deepest_facies, pred_te)), 4
    )
    row["deep_facies_nmi"] = round(
        float(normalized_mutual_info_score(test_data.deepest_facies, pred_te)), 4
    )


def _evaluate_depth_suite(
    eval_specs: list[tuple[str, np.ndarray, np.ndarray, float, float | None]],
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    test_graph: Data,
    train_data: TraceLogs,
    test_data: TraceLogs,
    num_classes: int,
    seed: int,
) -> tuple[list[MetricRow], dict[str, np.ndarray]]:
    metrics_rows: list[MetricRow] = []
    recon_preds_8d: dict[str, np.ndarray] = {}
    test_ei = graph_edge_index(test_graph)

    for m_name, z_tr, z_te, dt, kl_te in eval_specs:
        nm = neighborhood_metrics(
            x_test, z_te, k=_NEIGHBOR_K, sample_size=_EVAL_SAMPLE_SIZE, seed=seed
        )
        lm = local_graph_metrics(x_test, z_te, test_ei)
        probe = _probe_reconstruction(
            z_tr,
            z_te,
            x_train,
            x_test,
            use_mlp=m_name.startswith(("NOFE", "UMAP")),
            seed=seed,
        )
        if m_name in ("NOFE-8D", "PCA-8D", "UMAP-8D"):
            recon_preds_8d[m_name] = _reconstruction_pred_8d(
                m_name, z_tr, z_te, x_train, seed=seed
            )

        row: MetricRow = {
            "method": m_name,
            "train_sec": round(dt, 2),
            "heldout_kl": round(kl_te, 6) if kl_te is not None else None,
            "trustworthiness": round(nm["trustworthiness"], 4),
            "continuity": round(nm["continuity"], 4),
            "knn_recall": round(nm["knn_recall"], 4),
            "local_stress": round(lm["local_stress"], 4),
            "global_stress": round(nm["global_stress"], 4),
            "depth_recon_ridge_r2": probe["ridge_r2"],
            "depth_recon_mlp_r2": probe.get("mlp_r2", probe["ridge_r2"]),
        }
        _add_label_metrics(
            row,
            z_tr,
            z_te,
            train_data,
            test_data,
            num_classes=num_classes,
            seed=seed,
        )
        metrics_rows.append(row)

    return metrics_rows, recon_preds_8d


def _fit_timed_pca(
    x_train: np.ndarray, n_components: int, seed: int
) -> tuple[PCA, float]:
    t0 = time.perf_counter()
    model = PCA(n_components=n_components, random_state=seed).fit(x_train)
    return model, time.perf_counter() - t0


def _fit_timed_umap(
    x_train: np.ndarray, n_components: int, seed: int
) -> tuple[UMAP, float]:
    t0 = time.perf_counter()
    model = UMAP(
        n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=seed
    ).fit(x_train)
    return model, time.perf_counter() - t0


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
    split_i = max(si, (max(6, round(ni * cfg.train_inline_frac)) // si) * si)

    train_data = extract_trace_logs(
        volume,
        inline_range=(0, split_i),
        stride_2d=cfg.stride_2d,
        feature_mode=cfg.feature_mode,
    )
    test_data = extract_trace_logs(
        volume,
        inline_range=(split_i, ni),
        stride_2d=cfg.stride_2d,
        feature_mode=cfg.feature_mode,
    )
    full_data = extract_trace_logs(
        volume, stride_2d=cfg.stride_2d, feature_mode=cfg.feature_mode
    )

    scaler = SharedStandardizer.fit(
        train_data.traces, train_data.locations, feature_scale=cfg.feature_scale
    )
    x_train = scaler.transform_features(train_data.traces)
    x_test = scaler.transform_features(test_data.traces)
    x_full = scaler.transform_features(full_data.traces)

    builder = SeismicPointGraph(mode="radius", radius=cfg.radius, normalize=False)
    train_graph = builder.build(
        x_train, scaler.transform_locations(train_data.locations)
    )
    test_graph = builder.build(x_test, scaler.transform_locations(test_data.locations))
    full_graph = builder.build(x_full, scaler.transform_locations(full_data.locations))

    nofe_main, t_nofe3, hist_3d, nofe_8d, t_nofe8, hist_8d = _train_nofe_pair(
        train_graph, cfg
    )
    emb_nofe3_tr, emb_nofe3_te = (
        embed_numpy(nofe_main, train_graph),
        embed_numpy(nofe_main, test_graph),
    )
    emb_nofe8_tr, emb_nofe8_te = (
        embed_numpy(nofe_8d, train_graph),
        embed_numpy(nofe_8d, test_graph),
    )
    emb_nofe8_full = embed_numpy(nofe_8d, full_graph)

    pca3, t_pca3 = _fit_timed_pca(x_train, cfg.out_dim, cfg.seed)
    pca8, t_pca8 = _fit_timed_pca(x_train, 8, cfg.seed)
    umap3, t_umap3 = _fit_timed_umap(x_train, cfg.out_dim, cfg.seed)
    umap8, t_umap8 = _fit_timed_umap(x_train, 8, cfg.seed)

    emb_pca8_tr = pca8.transform(x_train).astype(np.float32)
    emb_pca8_full = pca8.transform(x_full).astype(np.float32)
    emb_umap8_tr = umap8.transform(x_train).astype(np.float32)
    emb_umap8_full = umap8.transform(x_full).astype(np.float32)

    eval_specs: list[tuple[str, np.ndarray, np.ndarray, float, float | None]] = [
        (
            f"NOFE-{cfg.out_dim}D",
            emb_nofe3_tr,
            emb_nofe3_te,
            t_nofe3,
            graph_loss(nofe_main, test_graph),
        ),
        (
            "NOFE-8D",
            emb_nofe8_tr,
            emb_nofe8_te,
            t_nofe8,
            graph_loss(nofe_8d, test_graph),
        ),
        (
            f"PCA-{cfg.out_dim}D",
            pca3.transform(x_train).astype(np.float32),
            pca3.transform(x_test).astype(np.float32),
            t_pca3,
            None,
        ),
        (
            "PCA-8D",
            emb_pca8_tr,
            pca8.transform(x_test).astype(np.float32),
            t_pca8,
            None,
        ),
        (
            f"UMAP-{cfg.out_dim}D",
            umap3.transform(x_train).astype(np.float32),
            umap3.transform(x_test).astype(np.float32),
            t_umap3,
            None,
        ),
        (
            "UMAP-8D",
            emb_umap8_tr,
            umap8.transform(x_test).astype(np.float32),
            t_umap8,
            None,
        ),
    ]
    metrics_rows, recon_preds_8d = _evaluate_depth_suite(
        eval_specs,
        x_train=x_train,
        x_test=x_test,
        test_graph=test_graph,
        train_data=train_data,
        test_data=test_data,
        num_classes=full_data.num_classes,
        seed=cfg.seed,
    )

    nofe8_pca3_full, nofe8_var = plot_8d_projection_map(
        emb_nofe8_full,
        full_data.grid_shape,
        full_data.locations,
        fig_dir / "nofe_8d_visualization.png",
        fit_embedding=emb_nofe8_tr,
        split_inline=float(split_i),
        title="Visualising NOFE-8D Depth Compression: Dims 0-2 vs. PCA-3 & Top PCs",
        seed=cfg.seed,
    )
    pca8_pca3_full, _ = project_embedding_pca(
        emb_pca8_full, fit_embedding=emb_pca8_tr, n_components=3, seed=cfg.seed
    )
    umap8_pca3_full, _ = project_embedding_pca(
        emb_umap8_full, fit_embedding=emb_umap8_tr, n_components=3, seed=cfg.seed
    )

    kl_nodes, _ = per_query_kl(
        None,
        torch.from_numpy(emb_nofe8_full),
        graph_edge_index(full_graph),
        num_nodes=graph_num_nodes(full_graph),
        weights=full_graph.affinity,
    )
    k_viz = (
        max(3, min(6, full_data.num_classes))
        if test_data.facies_fractions is not None
        else 5
    )
    clusters_full: dict[str, np.ndarray] = {
        m_name: KMeans(n_clusters=k_viz, n_init=10, random_state=cfg.seed)
        .fit(z_tr)
        .predict(z_full)
        for m_name, z_tr, z_full in (
            ("NOFE-8D", emb_nofe8_tr, emb_nofe8_full),
            ("PCA-8D", emb_pca8_tr, emb_pca8_full),
            ("UMAP-8D", emb_umap8_tr, emb_umap8_full),
        )
    }

    _plot_rgb_maps(
        full_data,
        {
            f"NOFE-{cfg.out_dim}D (RGB)": embed_numpy(nofe_main, full_graph),
            "NOFE-8D (PCA-3 RGB)": nofe8_pca3_full,
            "PCA-8D (PCA-3 RGB)": pca8_pca3_full,
            "UMAP-8D (PCA-3 RGB)": umap8_pca3_full,
        },
        split_inline=float(split_i),
        out_dir=fig_dir,
    )
    _plot_kl_and_cluster_maps(
        full_data,
        kl_nodes.numpy(),
        clusters_full,
        split_inline=float(split_i),
        out_dir=fig_dir,
    )
    _plot_sample_reconstructions(
        x_test,
        recon_preds_8d,
        test_data.locations,
        test_data.deepest_facies,
        fig_dir / "depth_log_reconstructions.png",
    )

    pd.DataFrame(metrics_rows).to_csv(
        out_dir / "depth_compression_metrics.csv", index=False
    )
    summary: dict[str, object] = {
        "dataset": volume.name,
        "volume_shape": list(volume.shape),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        },
        "train_graph": graph_summary(train_graph),
        "full_map_graph": graph_summary(full_graph),
        "nofe8_pca3_explained_var": [round(float(x), 4) for x in nofe8_var],
        "history_3d_last": hist_3d[-1] if hist_3d else None,
        "history_8d_last": hist_8d[-1] if hist_8d else None,
        "metrics": metrics_rows,
    }
    (out_dir / "depth_compression_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
