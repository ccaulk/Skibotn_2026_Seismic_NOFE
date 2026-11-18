"""Latent-dim ablations, facies separation & publication figures."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from nofe import NOFE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from umap import UMAP

from .data import LazySeismicVolume, PointCloud, SeismicVolume, volume_to_points
from .depth_compression import TraceLogs, extract_trace_logs
from .graph import SeismicPointGraph, SharedStandardizer
from .metrics import local_graph_metrics, neighborhood_metrics
from .viz import embedding_to_rgb, project_embedding_pca

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
TEXTURE_CHANNEL_INDICES: tuple[int, ...] = (1, 4, 6, 7, 8, 9, 10)
FACIES_COLORS: tuple[str, ...] = (
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#984ea3",
    "#8c564b",
)

_MIN_NYSTROM_2D = 5000
_MIN_NYSTROM_3D = 4000
_UMAP_MAX_DIM = 16
_UMAP_MIN_NODES = 3
_F3_INLINE_THRESHOLD = 400
_F3_CUT_I = 300
_F3_CUT_X = 450
_F3_CUT_Z = 160
_F3_HERO_DIM = 16
_F3_PCA_REF_DIM = 8
_GALLERY_MAP_PANELS = 3
_CM_HIGHLIGHT_THRESH = 0.5
_MIN_NEIGHBORHOOD_ROWS = 4
_PCA_RGB_DIMS = 3
_FEATURE_SCALE_TUNED_3D = 1.5


def _embed_np(model: NOFE, graph: Data) -> np.ndarray:
    """Run ``model.embed(graph)`` and narrow to a NumPy array."""
    emb = model.embed(graph)
    if isinstance(emb, list):
        emb = emb[0]
    return emb.numpy()


def _loss_float(model: NOFE, graph: Data) -> float:
    """Run ``model.loss(graph)`` and narrow to a scalar float."""
    val = model.loss(graph)
    return float(val[0] if isinstance(val, list) else val)


def _edges(graph: Data) -> torch.Tensor:
    """Return non-null ``edge_index`` from a built PyG ``Data`` graph."""
    if graph.edge_index is None:
        raise ValueError("Graph is missing edge_index.")
    return graph.edge_index


def _interleaved_inline_mask(locations: np.ndarray, holdout_mod: int = 4) -> np.ndarray:
    """Boolean mask putting every ``holdout_mod``-th unique inline into evaluation."""
    unique_i = np.unique(locations[:, 0])
    held_out = set(
        unique_i[np.arange(len(unique_i)) % holdout_mod == (holdout_mod - 1)].tolist()
    )
    return np.array([val not in held_out for val in locations[:, 0]], dtype=bool)


def _balanced_sample_indices(
    labels: np.ndarray, *, max_per_class: int = 800, seed: int = 42
) -> np.ndarray:
    """Draw a class-balanced subsample so rare seismic units stay visible."""
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        take = min(max_per_class, idx.shape[0])
        parts.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def _contrast_stretch_rgb(proj: np.ndarray) -> np.ndarray:
    """Percentile-stretched RGB helper wrapping ``embedding_to_rgb`` cleanly."""
    return embedding_to_rgb(proj, low_pct=2.0, high_pct=98.0)


def _build_3d_texture_features(pts: PointCloud, depth_max: int) -> np.ndarray:
    """Combine phase-invariant 3D texture attributes with normalized depth."""
    rel_z = (pts.locations[:, 2:3] / max(float(depth_max), 1.0)).astype(np.float32)
    return np.concatenate(
        [pts.features[:, TEXTURE_CHANNEL_INDICES], rel_z], axis=1
    ).astype(np.float32)


def _facies_cmap_and_norm(classes: np.ndarray) -> tuple[ListedColormap, BoundaryNorm]:
    """Build discrete colormap and boundary norm for geological facies classes."""
    max_cls = int(classes.max())
    cmap = ListedColormap(
        [FACIES_COLORS[c % len(FACIES_COLORS)] for c in range(max_cls + 1)]
    )
    norm = BoundaryNorm(np.arange(-0.5, max_cls + 1.5, 1.0), cmap.N)
    return cmap, norm


def _evaluate_facies_probes(
    z_tr: np.ndarray,
    z_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate linear, k-NN, clustering, silhouette, and per-class F1 scores."""
    classes = np.unique(np.concatenate([y_tr, y_te])).astype(int)
    sc = StandardScaler().fit(z_tr)
    z_tr_s = sc.transform(z_tr)
    z_te_s = sc.transform(z_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(z_tr_s, y_tr)
    pred_lr = clf.predict(z_te_s)

    n_nbrs = max(1, min(7, z_tr.shape[0]))
    knn = KNeighborsClassifier(n_neighbors=n_nbrs).fit(z_tr_s, y_tr)
    pred_knn = knn.predict(z_te_s)

    n_clust = max(2, len(classes))
    km = KMeans(n_clusters=n_clust, n_init=10, random_state=seed).fit(z_te_s)
    sil_idx = _balanced_sample_indices(y_te, max_per_class=600, seed=seed)
    unique_sil = len(np.unique(y_te[sil_idx]))
    sil = (
        float(silhouette_score(z_te_s[sil_idx], y_te[sil_idx]))
        if unique_sil > 1 and len(sil_idx) > unique_sil
        else 0.0
    )

    per_cls = f1_score(y_te, pred_lr, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_te, pred_lr, labels=classes, normalize="true")
    return {
        "lr_bal_acc": round(float(balanced_accuracy_score(y_te, pred_lr)), 4),
        "lr_macro_f1": round(
            float(f1_score(y_te, pred_lr, average="macro", zero_division=0)), 4
        ),
        "knn_acc": round(float(np.mean(pred_knn == y_te)), 4),
        "kmeans_ari": round(float(adjusted_rand_score(y_te, km.labels_)), 4),
        "kmeans_nmi": round(float(normalized_mutual_info_score(y_te, km.labels_)), 4),
        "silhouette": round(sil, 4),
        "classes": [int(c) for c in classes],
        "per_class_f1": {
            int(c): round(float(v), 4) for c, v in zip(classes, per_cls, strict=True)
        },
        "confusion_matrix": np.round(cm, 4).tolist(),
    }


def _evaluate_2d_probes(
    *,
    z_tr: np.ndarray,
    z_te: np.ndarray,
    x_tr_raw: np.ndarray,
    x_te_raw: np.ndarray,
    full2d: TraceLogs,
    m_tr: np.ndarray,
    m_te: np.ndarray,
    g_te: Data,
    seed: int,
) -> dict[str, Any]:
    """Compute geometry, depth reconstruction, fraction R^2, and facies scores."""
    loc = local_graph_metrics(x_te_raw, z_te, _edges(g_te))
    k_val = max(2, min(15, z_te.shape[0] - 2))
    nbr = neighborhood_metrics(
        x_te_raw,
        z_te,
        k=k_val if z_te.shape[0] > _MIN_NEIGHBORHOOD_ROWS else 1,
        sample_size=2500,
        seed=seed,
    )

    reg_trace = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=150,
        early_stopping=True,
        random_state=seed,
    ).fit(z_tr, x_tr_raw)
    depth_recon_r2 = float(r2_score(x_te_raw, reg_trace.predict(z_te)))

    out: dict[str, Any] = {
        "local_stress": round(float(loc["local_stress"]), 4),
        "global_stress": round(float(nbr["global_stress"]), 4),
        "trustworthiness": round(float(nbr["trustworthiness"]), 4),
        "knn_recall": round(float(nbr["knn_recall"]), 4),
        "depth_recon_r2": round(depth_recon_r2, 4),
    }
    if full2d.facies_fractions is not None and full2d.deepest_facies is not None:
        reg_frac = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=150,
            early_stopping=True,
            random_state=seed,
        ).fit(z_tr, full2d.facies_fractions[m_tr])
        out["facies_fraction_r2"] = round(
            float(r2_score(full2d.facies_fractions[m_te], reg_frac.predict(z_te))), 4
        )
        fac = _evaluate_facies_probes(
            z_tr,
            z_te,
            full2d.deepest_facies[m_tr],
            full2d.deepest_facies[m_te],
            seed=seed,
        )
        out["deepest_lr_bal_acc"] = fac["lr_bal_acc"]
        out["deepest_knn_acc"] = fac["knn_acc"]
        out["deepest_macro_f1"] = fac["lr_macro_f1"]
    return out


def _run_2d_latent_ablation(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    dims: tuple[int, ...],
    stride_2d: tuple[int, int],
    radius_2d: float,
    epochs_2d: int,
    device: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, NOFE], SharedStandardizer, TraceLogs]:
    """Sweep latent dim ``d`` on 2D depth-log compression across NOFE, PCA, UMAP."""
    full2d = extract_trace_logs(volume, stride_2d=stride_2d, feature_mode="envelope")
    m_tr = _interleaved_inline_mask(full2d.locations, holdout_mod=4)
    m_te = ~m_tr

    sc = SharedStandardizer.fit(
        full2d.traces[m_tr], full2d.locations[m_tr], feature_scale="global"
    )
    x_tr = sc.transform_features(full2d.traces[m_tr])
    p_tr = sc.transform_locations(full2d.locations[m_tr])
    x_te = sc.transform_features(full2d.traces[m_te])
    p_te = sc.transform_locations(full2d.locations[m_te])

    g_tr = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False).build(
        x_tr, p_tr
    )
    g_te = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False).build(
        x_te, p_te
    )

    rows: list[dict[str, Any]] = []
    nofe_models: dict[int, NOFE] = {}
    n_depth = int(full2d.traces.shape[1])
    n_samples_tr, n_feat = x_tr.shape

    for d in dims:
        t0 = time.perf_counter()
        model = NOFE(
            out_dim=d,
            width=max(24, min(64, 2 * d)),
            ker_width=64,
            depth=1,
            nystrom=0.12 if n_samples_tr > _MIN_NYSTROM_2D else None,
            chunk_size=8192,
            device=device,
            seed=seed,
        )
        model.training(
            g_tr, epochs=epochs_2d, patience=15, lr=1e-2, verbose=False, seed=seed
        )
        train_sec = time.perf_counter() - t0
        nofe_models[d] = model

        z_tr, z_te = _embed_np(model, g_tr), _embed_np(model, g_te)
        metrics_nofe = _evaluate_2d_probes(
            z_tr=z_tr,
            z_te=z_te,
            x_tr_raw=x_tr,
            x_te_raw=x_te,
            full2d=full2d,
            m_tr=m_tr,
            m_te=m_te,
            g_te=g_te,
            seed=seed,
        )
        rows.append(
            {
                "dim": d,
                "compression_ratio": round(n_depth / float(d), 2),
                "method": "NOFE",
                "train_sec": round(train_sec, 2),
                "heldout_kl": round(_loss_float(model, g_te), 5),
                **metrics_nofe,
            }
        )

        if d <= min(n_samples_tr, n_feat):
            t0 = time.perf_counter()
            pca = PCA(n_components=d, random_state=seed).fit(x_tr)
            zp_tr = pca.transform(x_tr).astype(np.float32)
            zp_te = pca.transform(x_te).astype(np.float32)
            metrics_pca = _evaluate_2d_probes(
                z_tr=zp_tr,
                z_te=zp_te,
                x_tr_raw=x_tr,
                x_te_raw=x_te,
                full2d=full2d,
                m_tr=m_tr,
                m_te=m_te,
                g_te=g_te,
                seed=seed,
            )
            rows.append(
                {
                    "dim": d,
                    "compression_ratio": round(n_depth / float(d), 2),
                    "method": "PCA",
                    "train_sec": round(time.perf_counter() - t0, 2),
                    "heldout_kl": None,
                    **metrics_pca,
                }
            )

        if d <= min(_UMAP_MAX_DIM, n_samples_tr - 2):
            t0 = time.perf_counter()
            n_nbrs = min(15, max(_UMAP_MIN_NODES, n_samples_tr - 2))
            umap = UMAP(
                n_components=d, n_neighbors=n_nbrs, min_dist=0.1, random_state=seed
            ).fit(x_tr)
            zu_tr = umap.transform(x_tr).astype(np.float32)
            zu_te = umap.transform(x_te).astype(np.float32)
            metrics_u = _evaluate_2d_probes(
                z_tr=zu_tr,
                z_te=zu_te,
                x_tr_raw=x_tr,
                x_te_raw=x_te,
                full2d=full2d,
                m_tr=m_tr,
                m_te=m_te,
                g_te=g_te,
                seed=seed,
            )
            rows.append(
                {
                    "dim": d,
                    "compression_ratio": round(n_depth / float(d), 2),
                    "method": "UMAP",
                    "train_sec": round(time.perf_counter() - t0, 2),
                    "heldout_kl": None,
                    **metrics_u,
                }
            )

    return rows, nofe_models, sc, full2d


def _fit_and_eval_3d_naive(
    pts: PointCloud,
    m_tr: np.ndarray,
    m_te: np.ndarray,
    *,
    radius_3d: float,
    epochs_3d: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Fit baseline 3D NOFE-8D (all 11 channels, isotropic positions, depth=2)."""
    if pts.labels is None:
        raise ValueError("3D facies evaluation requires labeled seismic volume.")
    y_tr, y_te = pts.labels[m_tr], pts.labels[m_te]
    sc = SharedStandardizer.fit(
        pts.features[m_tr], pts.locations[m_tr], feature_scale="column"
    )
    xn_tr = sc.transform_features(pts.features[m_tr])
    pn_tr = sc.transform_locations(pts.locations[m_tr])
    xn_te = sc.transform_features(pts.features[m_te])
    pn_te = sc.transform_locations(pts.locations[m_te])
    gn_tr = SeismicPointGraph(mode="radius", radius=radius_3d, normalize=False).build(
        xn_tr, pn_tr
    )
    gn_te = SeismicPointGraph(mode="radius", radius=radius_3d, normalize=False).build(
        xn_te, pn_te
    )
    model = NOFE(
        out_dim=8,
        width=24,
        ker_width=48,
        depth=2,
        nystrom=0.10 if xn_tr.shape[0] > _MIN_NYSTROM_3D else None,
        chunk_size=8192,
        device=device,
        seed=seed,
    )
    model.training(
        gn_tr, epochs=epochs_3d, patience=15, lr=1e-2, verbose=False, seed=seed
    )
    zn_tr, zn_te = _embed_np(model, gn_tr), _embed_np(model, gn_te)
    fac = _evaluate_facies_probes(zn_tr, zn_te, y_tr, y_te, seed=seed)
    loc = local_graph_metrics(xn_te, zn_te, _edges(gn_te))
    row = {
        "dim": 8,
        "method": "NOFE-Naive (d=2, iso)",
        "heldout_kl": round(_loss_float(model, gn_te), 5),
        "local_stress": round(float(loc["local_stress"]), 4),
        **{k: v for k, v in fac.items() if k not in ("confusion_matrix", "classes")},
    }
    return {"sc": sc, "model": model, "zn_te": zn_te, "fac": fac, "row": row}


def _run_3d_facies_and_ablation(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    dims: tuple[int, ...],
    epochs_3d: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Compare Naive vs Facies-Tuned 3D NOFE across latent dimensions and baselines."""
    ni, _, nz = volume.shape
    st = (16, 24, 6) if ni >= _F3_INLINE_THRESHOLD else (2, 2, 2)
    pts = volume_to_points(volume, stride=st)
    if pts.labels is None:
        raise ValueError("3D facies evaluation requires labeled seismic volume.")

    m_tr = _interleaved_inline_mask(pts.locations, holdout_mod=4)
    m_te = ~m_tr
    y_tr, y_te = pts.labels[m_tr], pts.labels[m_te]
    r3d = 0.15 if ni >= _F3_INLINE_THRESHOLD else 0.35

    naive = _fit_and_eval_3d_naive(
        pts, m_tr, m_te, radius_3d=r3d, epochs_3d=epochs_3d, device=device, seed=seed
    )

    f_tex = _build_3d_texture_features(pts, depth_max=nz)
    sc_t = SharedStandardizer.fit(
        f_tex[m_tr],
        pts.locations[m_tr],
        feature_scale="column",
        pos_weights=(1.0, 1.0, 2.0),
    )
    xt_tr = (sc_t.transform_features(f_tex[m_tr]) * _FEATURE_SCALE_TUNED_3D).astype(
        np.float32
    )
    pt_tr = sc_t.transform_locations(pts.locations[m_tr])
    xt_te = (sc_t.transform_features(f_tex[m_te]) * _FEATURE_SCALE_TUNED_3D).astype(
        np.float32
    )
    pt_te = sc_t.transform_locations(pts.locations[m_te])
    gt_tr = SeismicPointGraph(mode="radius", radius=r3d, normalize=False).build(
        xt_tr, pt_tr
    )
    gt_te = SeismicPointGraph(mode="radius", radius=r3d, normalize=False).build(
        xt_te, pt_te
    )

    rows_3d: list[dict[str, Any]] = [naive["row"]]
    tuned_models: dict[int, NOFE] = {}
    tuned_embeds_te: dict[int, np.ndarray] = {}
    tuned_fac_details: dict[int, dict[str, Any]] = {}

    for d in dims:
        m_tuned = NOFE(
            out_dim=d,
            width=max(32, min(64, 2 * d)),
            ker_width=64,
            depth=1,
            nystrom=0.10 if xt_tr.shape[0] > _MIN_NYSTROM_3D else None,
            chunk_size=8192,
            device=device,
            seed=seed,
        )
        m_tuned.training(
            gt_tr, epochs=epochs_3d, patience=15, lr=1e-2, verbose=False, seed=seed
        )
        zt_tr, zt_te = _embed_np(m_tuned, gt_tr), _embed_np(m_tuned, gt_te)
        fac = _evaluate_facies_probes(zt_tr, zt_te, y_tr, y_te, seed=seed)
        loc_m = local_graph_metrics(xt_te, zt_te, _edges(gt_te))
        tuned_models[d] = m_tuned
        tuned_embeds_te[d] = zt_te
        tuned_fac_details[d] = fac
        rows_3d.append(
            {
                "dim": d,
                "method": "NOFE-Tuned (d=1, aniso)",
                "heldout_kl": round(_loss_float(m_tuned, gt_te), 5),
                "local_stress": round(float(loc_m["local_stress"]), 4),
                **{
                    k: v
                    for k, v in fac.items()
                    if k not in ("confusion_matrix", "classes")
                },
            }
        )

        if d <= min(xt_tr.shape[0], xt_tr.shape[1]):
            pca = PCA(n_components=d, random_state=seed).fit(xt_tr)
            zp_tr, zp_te = pca.transform(xt_tr), pca.transform(xt_te)
            pfac = _evaluate_facies_probes(zp_tr, zp_te, y_tr, y_te, seed=seed)
            rows_3d.append(
                {
                    "dim": d,
                    "method": "PCA (tex7+z)",
                    "heldout_kl": None,
                    "local_stress": round(
                        float(
                            local_graph_metrics(xt_te, zp_te, _edges(gt_te))[
                                "local_stress"
                            ]
                        ),
                        4,
                    ),
                    **{
                        k: v
                        for k, v in pfac.items()
                        if k not in ("confusion_matrix", "classes")
                    },
                }
            )

        if d <= min(_UMAP_MAX_DIM, xt_tr.shape[0] - 2):
            n_nbrs = min(15, max(_UMAP_MIN_NODES, xt_tr.shape[0] - 2))
            um = UMAP(
                n_components=d, n_neighbors=n_nbrs, min_dist=0.1, random_state=seed
            ).fit(xt_tr)
            zu_tr, zu_te = um.transform(xt_tr), um.transform(xt_te)
            ufac = _evaluate_facies_probes(zu_tr, zu_te, y_tr, y_te, seed=seed)
            rows_3d.append(
                {
                    "dim": d,
                    "method": "UMAP (tex7+z)",
                    "heldout_kl": None,
                    "local_stress": round(
                        float(
                            local_graph_metrics(xt_te, zu_te, _edges(gt_te))[
                                "local_stress"
                            ]
                        ),
                        4,
                    ),
                    **{
                        k: v
                        for k, v in ufac.items()
                        if k not in ("confusion_matrix", "classes")
                    },
                }
            )

    hero_d = _F3_HERO_DIM if _F3_HERO_DIM in tuned_models else dims[-1]
    return {
        "rows": rows_3d,
        "radius_3d": r3d,
        "sc_naive": naive["sc"],
        "naive_model": naive["model"],
        "zn_te": naive["zn_te"],
        "naive_fac": naive["fac"],
        "sc_tuned": sc_t,
        "tuned_models": tuned_models,
        "hero_dim": hero_d,
        "zt_te_hero": tuned_embeds_te[hero_d],
        "tuned_fac_hero": tuned_fac_details[hero_d],
        "y_te": y_te,
    }


def _style_dim_axis(
    ax: Axes, dims: list[int], xlabel: str, ylabel: str, title: str
) -> None:
    """Apply clean log2 latent-dimension ticks and titles consistently."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _plot_fig1_latent_ablation(
    rows_2d: list[dict[str, Any]],
    rows_3d: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Plot 2x3 Pareto and latent-dimension ablation curves for 2D and 3D NOFE."""
    df2 = pd.DataFrame(rows_2d)
    df3 = pd.DataFrame(rows_3d)
    colors = {"NOFE": "#1b7837", "PCA": "#762a83", "UMAP": "#d95f02"}
    colors3 = {
        "NOFE-Tuned (d=1, aniso)": "#1b7837",
        "PCA (tex7+z)": "#762a83",
        "UMAP (tex7+z)": "#d95f02",
    }
    dims2 = sorted(df2["dim"].unique())
    dims3 = sorted(df3["dim"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)

    for method, col in colors.items():
        sub = df2[df2["method"] == method].sort_values("dim")
        axes[0, 0].plot(
            sub["dim"], sub["local_stress"], marker="o", lw=2.2, color=col, label=method
        )
        axes[0, 1].plot(
            sub["dim"],
            sub["depth_recon_r2"],
            marker="s",
            lw=2.2,
            color=col,
            label=method,
        )
        y_col = (
            "facies_fraction_r2"
            if "facies_fraction_r2" in df2.columns
            else "trustworthiness"
        )
        axes[0, 2].plot(
            sub["dim"], sub[y_col], marker="^", lw=2.2, color=col, label=method
        )
    _style_dim_axis(
        axes[0, 0],
        dims2,
        "Latent dimension d",
        "Local spatial stress (lower better)",
        "(a) 2D depth-log local geometry stress",
    )
    _style_dim_axis(
        axes[0, 1],
        dims2,
        "Latent dimension d",
        "MLP depth-log reconstruction R^2 (higher better)",
        "(b) Full trace depth-log reconstruction",
    )
    _style_dim_axis(
        axes[0, 2],
        dims2,
        "Latent dimension d",
        "Stratigraphic fraction R^2"
        if "facies_fraction_r2" in df2.columns
        else "Trustworthiness",
        "(c) 2D formation thickness preservation",
    )

    for method, col in colors3.items():
        sub = df3[df3["method"] == method].sort_values("dim")
        if len(sub) > 0:
            axes[1, 0].plot(
                sub["dim"],
                sub["lr_bal_acc"],
                marker="o",
                lw=2.2,
                color=col,
                label=method,
            )
    naive = df3[df3["method"] == "NOFE-Naive (d=2, iso)"]
    if len(naive) > 0:
        axes[1, 0].scatter(
            naive["dim"],
            naive["lr_bal_acc"],
            s=110,
            marker="X",
            color="#e41a1c",
            zorder=5,
            label="NOFE-naive (d=2, iso)",
        )
    _style_dim_axis(
        axes[1, 0],
        dims3,
        "Latent dimension d",
        "Balanced accuracy (higher better)",
        "(d) 3D facies linear separation vs d",
    )

    sub_n2 = df2[df2["method"] == "NOFE"].sort_values("dim")
    sub_n3 = df3[df3["method"] == "NOFE-Tuned (d=1, aniso)"].sort_values("dim")
    axes[1, 1].plot(
        sub_n2["dim"],
        sub_n2["heldout_kl"],
        marker="o",
        lw=2.2,
        color="#1b7837",
        label="2D NOFE KL",
    )
    axes[1, 1].plot(
        sub_n3["dim"],
        sub_n3["heldout_kl"],
        marker="s",
        lw=2.2,
        ls="--",
        color="#2166ac",
        label="3D tuned NOFE KL",
    )
    axes[1, 1].set_yscale("log")
    _style_dim_axis(
        axes[1, 1],
        dims2,
        "Latent dimension d",
        "Held-out affinity KL loss (log)",
        "(e) Affinity KL decay across latent dims",
    )

    for method, col in colors.items():
        sub = df2[df2["method"] == method].sort_values("compression_ratio")
        joint = 0.5 * (1.0 - sub["local_stress"]) + 0.5 * np.clip(
            sub["depth_recon_r2"], 0.0, 1.0
        )
        axes[1, 2].plot(
            sub["compression_ratio"], joint, marker="o", lw=2.2, color=col, label=method
        )
        if method == "NOFE":
            for _, row in sub.iterrows():
                y_pt = 0.5 * (1.0 - row["local_stress"]) + 0.5 * max(
                    0.0, row["depth_recon_r2"]
                )
                axes[1, 2].annotate(
                    f"d={int(row['dim'])}",
                    (row["compression_ratio"], y_pt),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                    color="#1b7837",
                    fontweight="bold",
                )
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel("Depth compression ratio (253 / d, log)")
    axes[1, 2].set_ylabel("Joint geometry + reconstruction score")
    axes[1, 2].set_title("(f) Rate-distortion Pareto frontier", fontweight="bold")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(fontsize=8)

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
    """Render annotated row-normalized confusion matrix heatmap."""
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


def _plot_fig2_facies_breakthrough(
    volume: SeismicVolume | LazySeismicVolume,
    res3d: dict[str, Any],
    *,
    seed: int,
    out_path: Path,
) -> None:
    """Plot 2x3 before-vs-after 3D facies separation manifolds and metrics."""
    y_te, zn_te, zt_te = res3d["y_te"], res3d["zn_te"], res3d["zt_te_hero"]
    naive_fac, tuned_fac = res3d["naive_fac"], res3d["tuned_fac_hero"]
    hero_d = res3d["hero_dim"]
    rows_3d = pd.DataFrame(res3d["rows"])
    classes = np.asarray(tuned_fac["classes"], dtype=int)
    cmap_fac, norm_fac = _facies_cmap_and_norm(classes)
    short_names = [
        F3_FACIES_SHORT[c] if c < len(F3_FACIES_SHORT) else f"F{c}" for c in classes
    ]

    bal_idx = _balanced_sample_indices(y_te, max_per_class=750, seed=seed)
    y_bal = y_te[bal_idx]
    n_nbrs = min(15, max(_UMAP_MIN_NODES, len(bal_idx) - 2))
    um_naive = UMAP(
        n_components=2, n_neighbors=n_nbrs, min_dist=0.15, random_state=seed
    ).fit_transform(zn_te[bal_idx])
    um_tuned = UMAP(
        n_components=2, n_neighbors=n_nbrs, min_dist=0.15, random_state=seed
    ).fit_transform(zt_te[bal_idx])

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.8), constrained_layout=True)
    i_mid = volume.shape[0] // 2
    if volume.labels is None:
        raise ValueError("3D facies figure requires labeled seismic volume.")
    lbl_slice = np.asarray(volume.labels[i_mid, :, :], dtype=np.int64).T
    im = axes[0, 0].imshow(
        lbl_slice, aspect="auto", cmap=cmap_fac, norm=norm_fac, interpolation="nearest"
    )
    axes[0, 0].set_xlabel("Crossline")
    axes[0, 0].set_ylabel("Depth")
    axes[0, 0].set_title(f"(a) Ground-truth facies (inline {i_mid})", fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0, 0], ticks=classes, fraction=0.045, pad=0.03)
    cbar.ax.set_yticklabels(short_names, fontsize=7.5)

    for ax, um, title in [
        (
            axes[0, 1],
            um_naive,
            (
                "(b) Naive NOFE-8D (depth=2, iso)\n"
                f"BalAcc={naive_fac['lr_bal_acc']:.3f}, "
                f"MacroF1={naive_fac['lr_macro_f1']:.3f}"
            ),
        ),
        (
            axes[0, 2],
            um_tuned,
            (
                f"(c) Tuned NOFE-{hero_d}D (depth=1, aniso-Z)\n"
                f"BalAcc={tuned_fac['lr_bal_acc']:.3f}, "
                f"MacroF1={tuned_fac['lr_macro_f1']:.3f}"
            ),
        ),
    ]:
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
        naive_fac["confusion_matrix"],
        short_names,
        "(d) Naive NOFE-8D confusion matrix",
    )
    _draw_confusion(
        axes[1, 1],
        tuned_fac["confusion_matrix"],
        short_names,
        f"(e) Tuned NOFE-{hero_d}D confusion matrix",
    )

    pca8 = rows_3d[
        (rows_3d["method"] == "PCA (tex7+z)")
        & (rows_3d["dim"] == min(_F3_PCA_REF_DIM, int(rows_3d["dim"].max())))
    ]
    pca_f1 = pca8.iloc[0]["per_class_f1"] if len(pca8) > 0 else {}
    x_pos = np.arange(len(classes))
    w = 0.27
    axes[1, 2].bar(
        x_pos - w,
        [naive_fac["per_class_f1"].get(int(c), 0.0) for c in classes],
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
        [tuned_fac["per_class_f1"].get(int(c), 0.0) for c in classes],
        w,
        label=f"Tuned NOFE-{hero_d}D",
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


def _plot_gallery_cross_section_row(
    axes_row: np.ndarray,
    volume: SeismicVolume | LazySeismicVolume,
    res3d: dict[str, Any],
    *,
    seed: int,
) -> None:
    """Populate bottom row of Fig 3 with Inline cross-section slices."""
    ni, nx, nz = volume.shape
    i_mid = ni // 2
    st_sl = (1, 2, 2) if ni >= _F3_INLINE_THRESHOLD else (1, 1, 1)
    sl_pts = volume_to_points(volume, inline_range=(i_mid, i_mid + 1), stride=st_sl)
    nx_sl = len(range(0, nx, st_sl[1]))
    nz_sl = len(range(0, nz, st_sl[2]))

    amp_sl = sl_pts.amplitude.reshape(nx_sl, nz_sl).T
    vlim = float(np.percentile(np.abs(amp_sl), 98)) or 1.0
    axes_row[0].imshow(amp_sl, cmap="seismic", vmin=-vlim, vmax=vlim, aspect="auto")
    axes_row[0].set_title(f"(e) Raw seismic inline {i_mid}", fontweight="bold")
    axes_row[0].set_xlabel("Crossline")
    axes_row[0].set_ylabel("Depth sample")

    if sl_pts.labels is not None:
        classes_sl = np.unique(sl_pts.labels).astype(int)
        cmap_sl, norm_sl = _facies_cmap_and_norm(classes_sl)
        axes_row[1].imshow(
            sl_pts.labels.reshape(nx_sl, nz_sl).T,
            cmap=cmap_sl,
            norm=norm_sl,
            aspect="auto",
            interpolation="nearest",
        )
        axes_row[1].set_title("(f) True facies cross-section", fontweight="bold")
    axes_row[1].set_xlabel("Crossline")

    r3d = res3d["radius_3d"]
    gn_sl = SeismicPointGraph(mode="radius", radius=r3d, normalize=False).build(
        res3d["sc_naive"].transform_features(sl_pts.features),
        res3d["sc_naive"].transform_locations(sl_pts.locations),
    )
    zn_sl = _embed_np(res3d["naive_model"], gn_sl)
    pn_sl, _ = project_embedding_pca(
        zn_sl, fit_embedding=res3d["zn_te"], n_components=_PCA_RGB_DIMS, seed=seed
    )
    axes_row[2].imshow(
        np.transpose(_contrast_stretch_rgb(pn_sl).reshape(nx_sl, nz_sl, 3), (1, 0, 2)),
        aspect="auto",
    )
    axes_row[2].set_title("(g) Naive 3D NOFE-8D (blurred)", fontweight="bold")
    axes_row[2].set_xlabel("Crossline")

    f_tex_sl = _build_3d_texture_features(sl_pts, depth_max=nz)
    gt_sl = SeismicPointGraph(mode="radius", radius=r3d, normalize=False).build(
        (
            res3d["sc_tuned"].transform_features(f_tex_sl) * _FEATURE_SCALE_TUNED_3D
        ).astype(np.float32),
        res3d["sc_tuned"].transform_locations(sl_pts.locations),
    )
    hero_d = res3d["hero_dim"]
    zt_sl = _embed_np(res3d["tuned_models"][hero_d], gt_sl)
    pt_sl, _ = project_embedding_pca(
        zt_sl, fit_embedding=res3d["zt_te_hero"], n_components=_PCA_RGB_DIMS, seed=seed
    )
    axes_row[3].imshow(
        np.transpose(_contrast_stretch_rgb(pt_sl).reshape(nx_sl, nz_sl, 3), (1, 0, 2)),
        aspect="auto",
    )
    axes_row[3].set_title(
        f"(h) Tuned 3D NOFE-{hero_d}D (crisp beds)", fontweight="bold"
    )
    axes_row[3].set_xlabel("Crossline")


def _plot_fig3_gallery(
    volume: SeismicVolume | LazySeismicVolume,
    nofe_2d_models: dict[int, NOFE],
    sc_2d: SharedStandardizer,
    full2d: TraceLogs,
    res3d: dict[str, Any],
    *,
    radius_2d: float,
    seed: int,
    out_path: Path,
) -> None:
    """Plot 2x4 multi-scale 2D depth maps (d=3,8,16) + Inline cross-sections."""
    gi, gx = full2d.grid_shape
    x_full = sc_2d.transform_features(full2d.traces)
    p_full = sc_2d.transform_locations(full2d.locations)
    g_full = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False).build(
        x_full, p_full
    )

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)

    if full2d.deepest_facies is not None:
        classes = np.unique(full2d.deepest_facies).astype(int)
        cmap_fac, norm_fac = _facies_cmap_and_norm(classes)
        axes[0, 0].imshow(
            full2d.deepest_facies.reshape(gi, gx),
            cmap=cmap_fac,
            norm=norm_fac,
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, 0].set_title("(a) Ground-truth deepest facies", fontweight="bold")
    else:
        axes[0, 0].imshow(full2d.rms_amp.reshape(gi, gx), cmap="magma", aspect="auto")
        axes[0, 0].set_title("(a) Mean trace RMS energy", fontweight="bold")
    axes[0, 0].set_xlabel("Crossline index")
    axes[0, 0].set_ylabel("Inline index")

    show_dims = [d for d in (3, 8, 16) if d in nofe_2d_models]
    while len(show_dims) < _GALLERY_MAP_PANELS:
        show_dims.append(max(nofe_2d_models.keys()))
    for idx, (col_idx, d) in enumerate(zip((1, 2, 3), show_dims, strict=True)):
        z_map = _embed_np(nofe_2d_models[d], g_full)
        proj, _ = project_embedding_pca(z_map, n_components=_PCA_RGB_DIMS, seed=seed)
        axes[0, col_idx].imshow(
            _contrast_stretch_rgb(proj).reshape(gi, gx, 3), aspect="auto"
        )
        ratio = full2d.traces.shape[1] / float(d)
        tag = chr(ord("b") + idx)
        axes[0, col_idx].set_title(
            f"({tag}) 2D NOFE-{d}D map ({ratio:.1f}x compression)", fontweight="bold"
        )
        axes[0, col_idx].set_xlabel("Crossline index")

    _plot_gallery_cross_section_row(axes[1], volume, res3d, seed=seed)
    fig.suptitle(
        "Multi-scale 2D depth-log maps & 3D stratigraphic cross-section alignment",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _embed_tuned_plane(
    volume: SeismicVolume | LazySeismicVolume,
    res3d: dict[str, Any],
    *,
    shape_2d: tuple[int, int],
    seed: int,
    inline_range: tuple[int, int] | None = None,
    crossline_range: tuple[int, int] | None = None,
    depth_range: tuple[int, int] | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Embed one orthogonal slice using the facies-tuned 3D NOFE model."""
    _, _, nz = volume.shape
    hero_d = res3d["hero_dim"]
    pts = volume_to_points(
        volume,
        inline_range=inline_range,
        crossline_range=crossline_range,
        depth_range=depth_range,
        stride=stride,
    )
    f_tex = _build_3d_texture_features(pts, depth_max=nz)
    gp = SeismicPointGraph(
        mode="radius", radius=res3d["radius_3d"], normalize=False
    ).build(
        (res3d["sc_tuned"].transform_features(f_tex) * _FEATURE_SCALE_TUNED_3D).astype(
            np.float32
        ),
        res3d["sc_tuned"].transform_locations(pts.locations),
    )
    zp = _embed_np(res3d["tuned_models"][hero_d], gp)
    proj, _ = project_embedding_pca(
        zp, fit_embedding=res3d["zt_te_hero"], n_components=_PCA_RGB_DIMS, seed=seed
    )
    rgb = _contrast_stretch_rgb(proj).reshape(shape_2d[0], shape_2d[1], 3)
    rgba = np.concatenate([rgb, np.ones((*shape_2d, 1), dtype=np.float32)], axis=2)
    return pts.locations, rgba


def _plot_fig4_hero_3d_cube(
    volume: SeismicVolume | LazySeismicVolume,
    res3d: dict[str, Any],
    *,
    seed: int,
    out_path: Path,
) -> None:
    """Render high-contrast 3D chair-cut cube + class-balanced 3D latent manifold."""
    ni, nx, nz = volume.shape
    hero_d = res3d["hero_dim"]
    is_f3 = ni >= _F3_INLINE_THRESHOLD
    i_idx = _F3_CUT_I if is_f3 else ni // 2
    x_idx = _F3_CUT_X if is_f3 else nx // 2
    z_idx = _F3_CUT_Z if is_f3 else nz // 2
    s_in, s_xl, s_dp = (
        ((1, 6, 3), (4, 1, 3), (4, 6, 1))
        if is_f3
        else ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    )

    ni_xl = len(range(0, ni, s_xl[0]))
    nx_in = len(range(0, nx, s_in[1]))
    nz_in = len(range(0, nz, s_in[2]))
    ni_dp = len(range(0, ni, s_dp[0]))
    nx_dp = len(range(0, nx, s_dp[1]))

    loc_in, rgba_in = _embed_tuned_plane(
        volume,
        res3d,
        shape_2d=(nx_in, nz_in),
        seed=seed,
        inline_range=(i_idx, i_idx + 1),
        stride=s_in,
    )
    loc_xl, rgba_xl = _embed_tuned_plane(
        volume,
        res3d,
        shape_2d=(ni_xl, nz_in),
        seed=seed,
        crossline_range=(x_idx, x_idx + 1),
        stride=s_xl,
    )
    loc_dp, rgba_dp = _embed_tuned_plane(
        volume,
        res3d,
        shape_2d=(ni_dp, nx_dp),
        seed=seed,
        depth_range=(z_idx, z_idx + 1),
        stride=s_dp,
    )

    fig = plt.figure(figsize=(15.8, 6.8), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1.2, 1.0], figure=fig)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    for locs, rgba, sh in [
        (loc_in, rgba_in, (nx_in, nz_in)),
        (loc_xl, rgba_xl, (ni_xl, nz_in)),
        (loc_dp, rgba_dp, (ni_dp, nx_dp)),
    ]:
        ax3d.plot_surface(
            locs[:, 0].reshape(sh),
            locs[:, 1].reshape(sh),
            locs[:, 2].reshape(sh),
            facecolors=rgba,
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
        f"Facies-tuned 3D NOFE-{hero_d}D orthogonal cutaway\n"
        f"(depth=1, aniso-Z, inline={i_idx}, crossline={x_idx}, depth={z_idx})",
        fontsize=10.5,
        fontweight="bold",
    )

    ax_man = fig.add_subplot(gs[0, 1], projection="3d")
    y_te = res3d["y_te"]
    bal_idx = _balanced_sample_indices(y_te, max_per_class=700, seed=seed)
    z_proj, var_r = project_embedding_pca(
        res3d["zt_te_hero"], n_components=_PCA_RGB_DIMS, seed=seed
    )
    z_bal, y_bal = z_proj[bal_idx], y_te[bal_idx]
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

    tuned_fac = res3d["tuned_fac_hero"]
    ax_man.set_xlabel(f"PC1 ({100 * var_r[0]:.1f}%)")
    ax_man.set_ylabel(f"PC2 ({100 * var_r[1]:.1f}%)")
    ax_man.set_zlabel(f"PC3 ({100 * var_r[2]:.1f}%)")
    ax_man.set_title(
        f"Class-balanced 3D latent manifold (stars = centroids)\n"
        f"BalAcc={tuned_fac['lr_bal_acc']:.3f} | "
        f"MacroF1={tuned_fac['lr_macro_f1']:.3f} | "
        f"7-NN={tuned_fac['knn_acc']:.3f}",
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
    """Shade background depth intervals by true seismic formation class."""
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
    z_all: np.ndarray,
    x_full: np.ndarray,
    train_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit an MLP depth-log decoder on training traces and predict the full map."""
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
    """Populate bottom row of Fig 5 with ground truth, k-means clusters & RMSE map."""
    gi, gx = full2d.grid_shape
    km_ari, km_nmi = 0.0, 0.0
    if full2d.deepest_facies is not None:
        classes = np.unique(full2d.deepest_facies).astype(int)
        cmap_fac, norm_fac = _facies_cmap_and_norm(classes)
        axes_row[0].imshow(
            full2d.deepest_facies.reshape(gi, gx),
            cmap=cmap_fac,
            norm=norm_fac,
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


def _plot_fig5_trace_profiles_and_clusters(
    volume: SeismicVolume | LazySeismicVolume,
    nofe_2d_models: dict[int, NOFE],
    sc_2d: SharedStandardizer,
    full2d: TraceLogs,
    *,
    radius_2d: float,
    seed: int,
    out_path: Path,
) -> dict[str, float]:
    """Plot reconstructed seismic depth-log profiles & unsupervised 2D clusters."""
    gi, gx = full2d.grid_shape
    x_full = sc_2d.transform_features(full2d.traces)
    p_full = sc_2d.transform_locations(full2d.locations)
    g_full = SeismicPointGraph(mode="radius", radius=radius_2d, normalize=False).build(
        x_full, p_full
    )
    m_tr = _interleaved_inline_mask(full2d.locations, holdout_mod=4)

    d_low = (
        _F3_PCA_REF_DIM if _F3_PCA_REF_DIM in nofe_2d_models else min(nofe_2d_models)
    )
    d_high = _F3_HERO_DIM if _F3_HERO_DIM in nofe_2d_models else max(nofe_2d_models)
    z_low = _embed_np(nofe_2d_models[d_low], g_full)
    z_high = _embed_np(nofe_2d_models[d_high], g_full)

    pca_d = min(d_low, x_full.shape[1], int(m_tr.sum()))
    zp_full = (
        PCA(n_components=pca_d, random_state=seed)
        .fit(x_full[m_tr])
        .transform(x_full)
        .astype(np.float32)
    )
    rec_low = _predict_full_traces(z_low, x_full, m_tr, seed=seed)
    rec_high = _predict_full_traces(z_high, x_full, m_tr, seed=seed)
    rec_pca = _predict_full_traces(zp_full, x_full, m_tr, seed=seed)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), constrained_layout=True)
    specs = [
        ("(a) Shelf parallel beds", int(0.18 * gi), int(0.22 * gx)),
        ("(b) Chalk dipping flank", int(0.50 * gi), int(0.42 * gx)),
        ("(c) Zechstein salt diapir", int(0.45 * gi), int(0.62 * gx)),
    ]
    z_axis = np.arange(x_full.shape[1])
    for ax, (title, ri, rx) in zip(axes[0], specs, strict=True):
        idx = ri * gx + rx
        loc_i, loc_x = int(full2d.locations[idx, 0]), int(full2d.locations[idx, 1])
        lbl_col = (
            np.asarray(volume.labels[loc_i, loc_x, :], dtype=np.int64)
            if volume.labels is not None
            else None
        )
        _shade_trace_facies(ax, lbl_col)
        ax.plot(z_axis, x_full[idx], color="#111111", lw=1.7, label="True envelope")
        ax.plot(
            z_axis,
            rec_high[idx],
            color="#1b7837",
            lw=1.5,
            label=f"NOFE-{d_high}D ({x_full.shape[1] / d_high:.1f}x)",
        )
        ax.plot(
            z_axis,
            rec_low[idx],
            color="#2166ac",
            lw=1.2,
            ls="--",
            label=f"NOFE-{d_low}D ({x_full.shape[1] / d_low:.1f}x)",
        )
        ax.plot(
            z_axis,
            rec_pca[idx],
            color="#762a83",
            lw=1.1,
            ls=":",
            label=f"PCA-{pca_d}D ({x_full.shape[1] / pca_d:.1f}x)",
        )
        ax.set_title(f"{title} (inline {loc_i}, xline {loc_x})", fontweight="bold")
        ax.set_xlabel("Depth sample")
        ax.set_ylabel("Normalised trace envelope")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7.5, loc="upper right")

    trace_rmse = np.sqrt(np.mean((x_full - rec_high) ** 2, axis=1)).reshape(gi, gx)
    clust_stats = _plot_fig5_cluster_row(
        fig, axes[1], full2d, z_high, trace_rmse, d_high=d_high, seed=seed
    )
    fig.suptitle(
        "Trace-level depth-log reconstructions & unsupervised facies clustering",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return clust_stats


def run_ablation_and_facies_suite(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    out_dir: Path = Path("outputs/f3_ablations_and_facies"),
    device: str = "cpu",
    seed: int = 42,
    stride_2d: tuple[int, int] = (3, 3),
    radius_2d: float = 0.05,
    dims_2d: tuple[int, ...] = (2, 3, 4, 8, 16, 32),
    dims_3d: tuple[int, ...] = (3, 8, 16, 32),
    epochs_2d: int = 60,
    epochs_3d: int = 60,
) -> dict[str, Any]:
    """Run 2D/3D latent-dim ablations, optimize 3D facies separation & plot figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows_2d, nofe_2d_models, sc_2d, full2d = _run_2d_latent_ablation(
        volume,
        dims=dims_2d,
        stride_2d=stride_2d,
        radius_2d=radius_2d,
        epochs_2d=epochs_2d,
        device=device,
        seed=seed,
    )
    res3d = _run_3d_facies_and_ablation(
        volume, dims=dims_3d, epochs_3d=epochs_3d, device=device, seed=seed
    )

    _plot_fig1_latent_ablation(
        rows_2d, res3d["rows"], fig_dir / "fig1_latent_dim_ablation.png"
    )
    _plot_fig2_facies_breakthrough(
        volume,
        res3d,
        seed=seed,
        out_path=fig_dir / "fig2_facies_separation_breakthrough.png",
    )
    _plot_fig3_gallery(
        volume,
        nofe_2d_models,
        sc_2d,
        full2d,
        res3d,
        radius_2d=radius_2d,
        seed=seed,
        out_path=fig_dir / "fig3_multiscale_maps_and_cross_section.png",
    )
    _plot_fig4_hero_3d_cube(
        volume,
        res3d,
        seed=seed,
        out_path=fig_dir / "fig4_hero_3d_cutaway_and_facies.png",
    )
    trace_clust = _plot_fig5_trace_profiles_and_clusters(
        volume,
        nofe_2d_models,
        sc_2d,
        full2d,
        radius_2d=radius_2d,
        seed=seed,
        out_path=fig_dir / "fig5_trace_profiles_and_clusters.png",
    )

    pd.DataFrame(rows_2d).to_csv(out_dir / "latent_ablation_2d.csv", index=False)
    pd.DataFrame(res3d["rows"]).to_csv(
        out_dir / "latent_ablation_3d_facies.csv", index=False
    )
    summary = {
        "ablation_2d": rows_2d,
        "ablation_3d_facies": res3d["rows"],
        "facies_breakthrough": {
            "naive_nofe_8d": res3d["naive_fac"],
            f"tuned_nofe_{res3d['hero_dim']}d": res3d["tuned_fac_hero"],
        },
        "trace_clusters_2d": trace_clust,
    }
    (out_dir / "ablation_and_facies_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
