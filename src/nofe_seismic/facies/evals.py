"""2D/3D latent-dimension ablations and 3D geological facies separation evals."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

from nofe_seismic.data.traces import TraceLogs, extract_trace_logs
from nofe_seismic.data.volumes import (
    LazySeismicVolume,
    PointCloud,
    SeismicVolume,
    volume_to_points,
)
from nofe_seismic.graph import (
    ScaledGraph,
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    embed_numpy,
    fit_and_embed_nofe,
    graph_edge_index,
    graph_loss,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics
from nofe_seismic.types import NofeHyperparams, RowDict, Stride2D, Stride3D

TEXTURE_CHANNEL_INDICES: tuple[int, ...] = (1, 4, 6, 7, 8, 9, 10)
_MIN_NYSTROM_2D = 5000
_MIN_NYSTROM_3D = 4000
_UMAP_MAX_DIM = 16
_UMAP_MIN_NODES = 3
_F3_INLINE_THRESHOLD = 400
_F3_HERO_DIM = 16
_MIN_NEIGHBORHOOD_ROWS = 4
_FEATURE_SCALE_TUNED_3D = 1.5
_DEFAULT_STRIDE_2D = Stride2D(3, 3)


@dataclass(frozen=True)
class AblationFaciesConfig:
    """Parameters for the 2D/3D latent-dimension ablation and facies suite."""

    out_dir: Path = Path("outputs/f3_ablations_and_facies")
    device: str = "cpu"
    seed: int = 42
    stride_2d: Stride2D = _DEFAULT_STRIDE_2D
    radius_2d: float = 0.05
    dims_2d: tuple[int, ...] = (2, 3, 4, 8, 16, 32)
    dims_3d: tuple[int, ...] = (3, 8, 16, 32)
    epochs_2d: int = 60
    epochs_3d: int = 60


@dataclass(frozen=True)
class FaciesProbeResult:
    """Linear, k-NN, clustering and per-class scores on 3D facies labels."""

    lr_bal_acc: float
    lr_macro_f1: float
    knn_acc: float
    kmeans_ari: float
    kmeans_nmi: float
    silhouette: float
    classes: tuple[int, ...]
    per_class_f1: dict[int, float]
    confusion_matrix: list[list[float]]

    def scalar_metrics(self) -> RowDict:
        return {
            "lr_bal_acc": self.lr_bal_acc,
            "lr_macro_f1": self.lr_macro_f1,
            "knn_acc": self.knn_acc,
            "kmeans_ari": self.kmeans_ari,
            "kmeans_nmi": self.kmeans_nmi,
            "silhouette": self.silhouette,
        }

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.scalar_metrics(),
            "classes": list(self.classes),
            "per_class_f1": dict(self.per_class_f1),
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True)
class Ablation2DResult:
    """Results and fitted NOFE models from the 2D latent-dimension sweep."""

    rows: list[RowDict]
    nofe_models: dict[int, NOFE]
    scaler: SharedStandardizer
    full2d: TraceLogs


@dataclass(frozen=True)
class Naive3DResult:
    """Baseline 3D isotropic NOFE-8D result on all 11 seismic channels."""

    scaler: SharedStandardizer
    model: NOFE
    zn_te: np.ndarray
    fac: FaciesProbeResult
    row: RowDict


@dataclass(frozen=True)
class Ablation3DResult:
    """Comparison of Naive vs Facies-Tuned 3D NOFE across dimensions and baselines."""

    rows: list[RowDict]
    radius_3d: float
    sc_naive: SharedStandardizer
    naive_model: NOFE
    zn_te: np.ndarray
    naive_fac: FaciesProbeResult
    sc_tuned: SharedStandardizer
    tuned_models: dict[int, NOFE]
    hero_dim: int
    zt_te_hero: np.ndarray
    tuned_fac_hero: FaciesProbeResult
    pca_ref_f1: dict[int, float]
    y_te: np.ndarray


@dataclass(frozen=True)
class _Eval2DContext:
    x_tr_raw: np.ndarray
    x_te_raw: np.ndarray
    full2d: TraceLogs
    m_tr: np.ndarray
    m_te: np.ndarray
    g_te: Data
    seed: int


def interleaved_inline_mask(locations: np.ndarray, holdout_mod: int = 4) -> np.ndarray:
    """Boolean mask putting every ``holdout_mod``-th unique inline into eval."""
    unique_i = np.unique(locations[:, 0])
    held_out = set(
        unique_i[np.arange(len(unique_i)) % holdout_mod == (holdout_mod - 1)].tolist()
    )
    return np.array([val not in held_out for val in locations[:, 0]], dtype=bool)


def balanced_sample_indices(
    labels: np.ndarray, *, max_per_class: int = 800, seed: int = 42
) -> np.ndarray:
    """Draw a class-balanced subsample so rare seismic units stay visible."""
    rng = np.random.default_rng(seed)
    parts = [
        rng.choice(
            idx := np.flatnonzero(labels == cls),
            size=min(max_per_class, idx.shape[0]),
            replace=False,
        )
        for cls in np.unique(labels)
    ]
    return np.sort(np.concatenate(parts))


def build_3d_texture_features(pts: PointCloud, depth_max: int) -> np.ndarray:
    """Combine phase-invariant 3D texture attributes with normalised depth."""
    rel_z = (pts.locations[:, 2:3] / max(float(depth_max), 1.0)).astype(np.float32)
    return np.concatenate(
        [pts.features[:, TEXTURE_CHANNEL_INDICES], rel_z], axis=1
    ).astype(np.float32)


def evaluate_facies_probes(
    z_tr: np.ndarray,
    z_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int,
) -> FaciesProbeResult:
    """Evaluate linear, k-NN, clustering, silhouette and per-class F1 scores."""
    classes = tuple(int(c) for c in np.unique(np.concatenate([y_tr, y_te])))
    sc = StandardScaler().fit(z_tr)
    z_tr_s, z_te_s = sc.transform(z_tr), sc.transform(z_te)

    pred_lr = (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        .fit(z_tr_s, y_tr)
        .predict(z_te_s)
    )
    pred_knn = (
        KNeighborsClassifier(n_neighbors=max(1, min(7, z_tr.shape[0])))
        .fit(z_tr_s, y_tr)
        .predict(z_te_s)
    )
    km = KMeans(n_clusters=max(2, len(classes)), n_init=10, random_state=seed).fit(
        z_te_s
    )
    sil_idx = balanced_sample_indices(y_te, max_per_class=600, seed=seed)
    unique_sil = len(np.unique(y_te[sil_idx]))
    sil = (
        float(silhouette_score(z_te_s[sil_idx], y_te[sil_idx]))
        if unique_sil > 1 and len(sil_idx) > unique_sil
        else 0.0
    )

    cls_arr = np.asarray(classes, dtype=int)
    per_cls = f1_score(y_te, pred_lr, labels=cls_arr, average=None, zero_division=0)
    cm = confusion_matrix(y_te, pred_lr, labels=cls_arr, normalize="true")
    return FaciesProbeResult(
        lr_bal_acc=round(float(balanced_accuracy_score(y_te, pred_lr)), 4),
        lr_macro_f1=round(
            float(f1_score(y_te, pred_lr, average="macro", zero_division=0)), 4
        ),
        knn_acc=round(float(np.mean(pred_knn == y_te)), 4),
        kmeans_ari=round(float(adjusted_rand_score(y_te, km.labels_)), 4),
        kmeans_nmi=round(float(normalized_mutual_info_score(y_te, km.labels_)), 4),
        silhouette=round(sil, 4),
        classes=classes,
        per_class_f1={
            c: round(float(v), 4) for c, v in zip(classes, per_cls, strict=True)
        },
        confusion_matrix=np.round(cm, 4).tolist(),
    )


def _evaluate_2d_probes(
    z_tr: np.ndarray, z_te: np.ndarray, ctx: _Eval2DContext
) -> RowDict:
    loc = local_graph_metrics(ctx.x_te_raw, z_te, graph_edge_index(ctx.g_te))
    k_val = max(2, min(15, z_te.shape[0] - 2))
    nbr = neighborhood_metrics(
        ctx.x_te_raw,
        z_te,
        k=k_val if z_te.shape[0] > _MIN_NEIGHBORHOOD_ROWS else 1,
        sample_size=2500,
        seed=ctx.seed,
    )
    reg_trace = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=150,
        early_stopping=True,
        random_state=ctx.seed,
    ).fit(z_tr, ctx.x_tr_raw)
    out: RowDict = {
        "local_stress": round(float(loc["local_stress"]), 4),
        "global_stress": round(float(nbr["global_stress"]), 4),
        "trustworthiness": round(float(nbr["trustworthiness"]), 4),
        "knn_recall": round(float(nbr["knn_recall"]), 4),
        "depth_recon_r2": round(
            float(r2_score(ctx.x_te_raw, reg_trace.predict(z_te))), 4
        ),
    }
    if (
        ctx.full2d.facies_fractions is not None
        and ctx.full2d.deepest_facies is not None
    ):
        reg_frac = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=150,
            early_stopping=True,
            random_state=ctx.seed,
        ).fit(z_tr, ctx.full2d.facies_fractions[ctx.m_tr])
        out["facies_fraction_r2"] = round(
            float(
                r2_score(ctx.full2d.facies_fractions[ctx.m_te], reg_frac.predict(z_te))
            ),
            4,
        )
        fac = evaluate_facies_probes(
            z_tr,
            z_te,
            ctx.full2d.deepest_facies[ctx.m_tr],
            ctx.full2d.deepest_facies[ctx.m_te],
            seed=ctx.seed,
        )
        out["deepest_lr_bal_acc"] = fac.lr_bal_acc
        out["deepest_knn_acc"] = fac.knn_acc
        out["deepest_macro_f1"] = fac.lr_macro_f1
    return out


def _2d_ablation_row(
    d: int,
    method: str,
    *,
    n_depth: int,
    train_sec: float,
    heldout_kl: float | None,
    metrics: RowDict,
) -> RowDict:
    return {
        "dim": d,
        "compression_ratio": round(n_depth / float(d), 2),
        "method": method,
        "train_sec": round(train_sec, 2),
        "heldout_kl": round(heldout_kl, 5) if heldout_kl is not None else None,
        **metrics,
    }


def run_2d_latent_ablation(
    volume: SeismicVolume | LazySeismicVolume, cfg: AblationFaciesConfig
) -> Ablation2DResult:
    """Sweep latent dim ``d`` on 2D depth-log compression across NOFE, PCA and UMAP."""
    full2d = extract_trace_logs(
        volume, stride_2d=cfg.stride_2d, feature_mode="envelope"
    )
    m_tr = interleaved_inline_mask(full2d.locations, holdout_mod=4)
    m_te = ~m_tr

    sc = SharedStandardizer.fit(
        full2d.traces[m_tr], full2d.locations[m_tr], feature_scale="global"
    )
    builder = SeismicPointGraph(mode="radius", radius=cfg.radius_2d, normalize=False)
    g_tr = build_scaled_graph(builder, sc, full2d.traces[m_tr], full2d.locations[m_tr])
    g_te = build_scaled_graph(builder, sc, full2d.traces[m_te], full2d.locations[m_te])

    ctx = _Eval2DContext(
        x_tr_raw=g_tr.features,
        x_te_raw=g_te.features,
        full2d=full2d,
        m_tr=m_tr,
        m_te=m_te,
        g_te=g_te.graph,
        seed=cfg.seed,
    )
    rows: list[RowDict] = []
    nofe_models: dict[int, NOFE] = {}
    n_depth = int(full2d.traces.shape[1])
    n_samples_tr, n_feat = g_tr.features.shape

    for d in cfg.dims_2d:
        trained = fit_and_embed_nofe(
            g_tr.graph,
            NofeHyperparams(
                out_dim=d,
                width=max(24, min(64, 2 * d)),
                ker_width=64,
                depth=1,
                nystrom=0.12 if n_samples_tr > _MIN_NYSTROM_2D else None,
                chunk_size=8192,
                device=cfg.device,
                seed=cfg.seed,
                epochs=cfg.epochs_2d,
                patience=15,
                lr=1e-2,
            ),
        )
        nofe_models[d] = trained.model
        z_te = embed_numpy(trained.model, g_te.graph)
        rows.append(
            _2d_ablation_row(
                d,
                "NOFE",
                n_depth=n_depth,
                train_sec=trained.elapsed_sec,
                heldout_kl=graph_loss(trained.model, g_te.graph),
                metrics=_evaluate_2d_probes(trained.train_embedding, z_te, ctx),
            )
        )

        if d <= min(n_samples_tr, n_feat):
            t0 = time.perf_counter()
            pca = PCA(n_components=d, random_state=cfg.seed).fit(g_tr.features)
            rows.append(
                _2d_ablation_row(
                    d,
                    "PCA",
                    n_depth=n_depth,
                    train_sec=time.perf_counter() - t0,
                    heldout_kl=None,
                    metrics=_evaluate_2d_probes(
                        pca.transform(g_tr.features).astype(np.float32),
                        pca.transform(g_te.features).astype(np.float32),
                        ctx,
                    ),
                )
            )

        if d <= min(_UMAP_MAX_DIM, n_samples_tr - 2):
            t0 = time.perf_counter()
            n_nbrs = min(15, max(_UMAP_MIN_NODES, n_samples_tr - 2))
            umap = UMAP(
                n_components=d, n_neighbors=n_nbrs, min_dist=0.1, random_state=cfg.seed
            ).fit(g_tr.features)
            rows.append(
                _2d_ablation_row(
                    d,
                    "UMAP",
                    n_depth=n_depth,
                    train_sec=time.perf_counter() - t0,
                    heldout_kl=None,
                    metrics=_evaluate_2d_probes(
                        umap.transform(g_tr.features).astype(np.float32),
                        umap.transform(g_te.features).astype(np.float32),
                        ctx,
                    ),
                )
            )

    return Ablation2DResult(
        rows=rows, nofe_models=nofe_models, scaler=sc, full2d=full2d
    )


def _fit_and_eval_3d_naive(
    pts: PointCloud,
    m_tr: np.ndarray,
    m_te: np.ndarray,
    *,
    radius_3d: float,
    cfg: AblationFaciesConfig,
) -> Naive3DResult:
    if pts.labels is None:
        raise ValueError("3D facies evaluation requires labeled seismic volume.")
    y_tr, y_te = pts.labels[m_tr], pts.labels[m_te]
    sc = SharedStandardizer.fit(
        pts.features[m_tr], pts.locations[m_tr], feature_scale="column"
    )
    builder = SeismicPointGraph(mode="radius", radius=radius_3d, normalize=False)
    gn_tr = build_scaled_graph(builder, sc, pts.features[m_tr], pts.locations[m_tr])
    gn_te = build_scaled_graph(builder, sc, pts.features[m_te], pts.locations[m_te])

    trained = fit_and_embed_nofe(
        gn_tr.graph,
        NofeHyperparams(
            out_dim=8,
            width=24,
            ker_width=48,
            depth=2,
            nystrom=0.10 if gn_tr.features.shape[0] > _MIN_NYSTROM_3D else None,
            chunk_size=8192,
            device=cfg.device,
            seed=cfg.seed,
            epochs=cfg.epochs_3d,
            patience=15,
            lr=1e-2,
        ),
    )
    zn_te = embed_numpy(trained.model, gn_te.graph)
    fac = evaluate_facies_probes(
        trained.train_embedding, zn_te, y_tr, y_te, seed=cfg.seed
    )
    loc = local_graph_metrics(gn_te.features, zn_te, graph_edge_index(gn_te.graph))
    row: RowDict = {
        "dim": 8,
        "method": "NOFE-Naive (d=2, iso)",
        "heldout_kl": round(graph_loss(trained.model, gn_te.graph), 5),
        "local_stress": round(float(loc["local_stress"]), 4),
        **fac.scalar_metrics(),
    }
    return Naive3DResult(scaler=sc, model=trained.model, zn_te=zn_te, fac=fac, row=row)


@dataclass(frozen=True)
class _Method3DContext:
    y_tr: np.ndarray
    y_te: np.ndarray
    gt_te: ScaledGraph
    seed: int


@dataclass(frozen=True)
class _Method3DEval:
    row: RowDict
    fac: FaciesProbeResult


def _eval_3d_method_row(
    d: int,
    method: str,
    z_tr: np.ndarray,
    z_te: np.ndarray,
    ctx: _Method3DContext,
    *,
    heldout_kl: float | None = None,
) -> _Method3DEval:
    fac = evaluate_facies_probes(z_tr, z_te, ctx.y_tr, ctx.y_te, seed=ctx.seed)
    loc_m = local_graph_metrics(
        ctx.gt_te.features, z_te, graph_edge_index(ctx.gt_te.graph)
    )
    row: RowDict = {
        "dim": d,
        "method": method,
        "heldout_kl": round(heldout_kl, 5) if heldout_kl is not None else None,
        "local_stress": round(float(loc_m["local_stress"]), 4),
        **fac.scalar_metrics(),
    }
    return _Method3DEval(row=row, fac=fac)


def run_3d_facies_and_ablation(
    volume: SeismicVolume | LazySeismicVolume, cfg: AblationFaciesConfig
) -> Ablation3DResult:
    """Compare Naive vs Facies-Tuned 3D NOFE across latent dims and baselines."""
    ni, _, nz = volume.shape
    st = Stride3D(16, 24, 6) if ni >= _F3_INLINE_THRESHOLD else Stride3D(2, 2, 2)
    pts = volume_to_points(volume, stride=st)
    if pts.labels is None:
        raise ValueError("3D facies evaluation requires labeled seismic volume.")

    m_tr = interleaved_inline_mask(pts.locations, holdout_mod=4)
    m_te = ~m_tr
    y_tr, y_te = pts.labels[m_tr], pts.labels[m_te]
    r3d = 0.15 if ni >= _F3_INLINE_THRESHOLD else 0.35

    naive = _fit_and_eval_3d_naive(pts, m_tr, m_te, radius_3d=r3d, cfg=cfg)

    f_tex = build_3d_texture_features(pts, depth_max=nz)
    sc_t = SharedStandardizer.fit(
        f_tex[m_tr],
        pts.locations[m_tr],
        feature_scale="column",
        pos_weights=(1.0, 1.0, 2.0),
    )
    xt_tr = (sc_t.transform_features(f_tex[m_tr]) * _FEATURE_SCALE_TUNED_3D).astype(
        np.float32
    )
    xt_te = (sc_t.transform_features(f_tex[m_te]) * _FEATURE_SCALE_TUNED_3D).astype(
        np.float32
    )
    builder = SeismicPointGraph(mode="radius", radius=r3d, normalize=False)
    gt_tr = ScaledGraph(
        builder.build(xt_tr, sc_t.transform_locations(pts.locations[m_tr])),
        xt_tr,
        sc_t.transform_locations(pts.locations[m_tr]),
    )
    gt_te = ScaledGraph(
        builder.build(xt_te, sc_t.transform_locations(pts.locations[m_te])),
        xt_te,
        sc_t.transform_locations(pts.locations[m_te]),
    )

    rows_3d: list[RowDict] = [naive.row]
    tuned_models: dict[int, NOFE] = {}
    tuned_embeds_te: dict[int, np.ndarray] = {}
    tuned_fac_details: dict[int, FaciesProbeResult] = {}
    pca_ref_dim = min(8, max(cfg.dims_3d))
    pca_ref_f1: dict[int, float] = {}
    eval_ctx = _Method3DContext(y_tr=y_tr, y_te=y_te, gt_te=gt_te, seed=cfg.seed)

    for d in cfg.dims_3d:
        trained = fit_and_embed_nofe(
            gt_tr.graph,
            NofeHyperparams(
                out_dim=d,
                width=max(32, min(64, 2 * d)),
                ker_width=64,
                depth=1,
                nystrom=0.10 if xt_tr.shape[0] > _MIN_NYSTROM_3D else None,
                chunk_size=8192,
                device=cfg.device,
                seed=cfg.seed,
                epochs=cfg.epochs_3d,
                patience=15,
                lr=1e-2,
            ),
        )
        zt_te = embed_numpy(trained.model, gt_te.graph)
        res_t = _eval_3d_method_row(
            d,
            "NOFE-Tuned (d=1, aniso)",
            trained.train_embedding,
            zt_te,
            eval_ctx,
            heldout_kl=graph_loss(trained.model, gt_te.graph),
        )
        tuned_models[d] = trained.model
        tuned_embeds_te[d] = zt_te
        tuned_fac_details[d] = res_t.fac
        rows_3d.append(res_t.row)

        if d <= min(xt_tr.shape[0], xt_tr.shape[1]):
            pca = PCA(n_components=d, random_state=cfg.seed).fit(xt_tr)
            res_p = _eval_3d_method_row(
                d, "PCA (tex7+z)", pca.transform(xt_tr), pca.transform(xt_te), eval_ctx
            )
            if d == pca_ref_dim:
                pca_ref_f1 = res_p.fac.per_class_f1
            rows_3d.append(res_p.row)

        if d <= min(_UMAP_MAX_DIM, xt_tr.shape[0] - 2):
            n_nbrs = min(15, max(_UMAP_MIN_NODES, xt_tr.shape[0] - 2))
            um = UMAP(
                n_components=d, n_neighbors=n_nbrs, min_dist=0.1, random_state=cfg.seed
            ).fit(xt_tr)
            res_u = _eval_3d_method_row(
                d, "UMAP (tex7+z)", um.transform(xt_tr), um.transform(xt_te), eval_ctx
            )
            rows_3d.append(res_u.row)

    hero_d = _F3_HERO_DIM if _F3_HERO_DIM in tuned_models else cfg.dims_3d[-1]
    return Ablation3DResult(
        rows=rows_3d,
        radius_3d=r3d,
        sc_naive=naive.scaler,
        naive_model=naive.model,
        zn_te=naive.zn_te,
        naive_fac=naive.fac,
        sc_tuned=sc_t,
        tuned_models=tuned_models,
        hero_dim=hero_d,
        zt_te_hero=tuned_embeds_te[hero_d],
        tuned_fac_hero=tuned_fac_details[hero_d],
        pca_ref_f1=pca_ref_f1,
        y_te=y_te,
    )
