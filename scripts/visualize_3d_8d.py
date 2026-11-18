"""3D NOFE-8D vs PCA-8D / UMAP-8D comparison and PCA-to-RGB slice visualisation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nofe import NOFE
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from umap import UMAP

from nofe_seismic.data.volumes import (
    LazySeismicVolume,
    load_f3_directory,
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
    graph_summary,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics
from nofe_seismic.types import IndexRange, NofeHyperparams, RowDict, Stride3D
from nofe_seismic.viz.slices import (
    SliceFrame,
    plot_slice_comparison,
    project_embedding_pca,
)


@dataclass(frozen=True)
class SliceSpec:
    """Sampling coordinates and labels for a 2D slice visualisation."""

    key: str
    inline_range: IndexRange | None
    crossline_range: IndexRange | None
    depth_range: IndexRange | None
    stride: Stride3D
    coord_cols: tuple[int, int]
    xlabel: str
    ylabel: str
    invert_y: bool


@dataclass(frozen=True)
class _SliceVizModels:
    scaler: SharedStandardizer
    nofe8: NOFE
    pca8: PCA
    umap8: UMAP
    z_nofe_tr: np.ndarray
    z_pca_tr: np.ndarray


@dataclass(frozen=True)
class _EvalCandidate:
    name: str
    z_te: np.ndarray
    train_sec: float
    heldout_kl: float | None


def _slice_specs() -> tuple[SliceSpec, ...]:
    return (
        SliceSpec(
            key="inline_540",
            inline_range=IndexRange(540, 541),
            crossline_range=None,
            depth_range=None,
            stride=Stride3D(1, 4, 2),
            coord_cols=(1, 2),
            xlabel="Crossline",
            ylabel="Depth",
            invert_y=True,
        ),
        SliceSpec(
            key="crossline_450",
            inline_range=None,
            crossline_range=IndexRange(450, 451),
            depth_range=None,
            stride=Stride3D(4, 1, 2),
            coord_cols=(0, 2),
            xlabel="Inline",
            ylabel="Depth",
            invert_y=True,
        ),
        SliceSpec(
            key="depth_160",
            inline_range=None,
            crossline_range=None,
            depth_range=IndexRange(160, 161),
            stride=Stride3D(4, 4, 1),
            coord_cols=(0, 1),
            xlabel="Inline",
            ylabel="Crossline",
            invert_y=False,
        ),
    )


def _render_slice_comparisons(
    vol: LazySeismicVolume,
    models: _SliceVizModels,
    fig_dir: Path,
) -> dict[str, list[float]]:
    var_info: dict[str, list[float]] = {}
    builder = SeismicPointGraph(mode="radius", radius=0.15, normalize=False)
    for spec in _slice_specs():
        pts = volume_to_points(
            vol,
            inline_range=spec.inline_range,
            crossline_range=spec.crossline_range,
            depth_range=spec.depth_range,
            stride=spec.stride,
        )
        scaled = build_scaled_graph(builder, models.scaler, pts.features, pts.locations)
        emb8_n = embed_numpy(models.nofe8, scaled.graph)
        emb8_p = models.pca8.transform(scaled.features).astype(np.float32)
        emb8_u = models.umap8.transform(scaled.features).astype(np.float32)

        n_proj = project_embedding_pca(
            emb8_n, fit_embedding=models.z_nofe_tr, n_components=3
        )
        p_proj = project_embedding_pca(
            emb8_p, fit_embedding=models.z_pca_tr, n_components=3
        )
        u_proj = project_embedding_pca(emb8_u, n_components=3)
        var_info[spec.key] = [round(float(v), 4) for v in n_proj.explained_variance]

        frame = SliceFrame(
            coords=pts.locations[:, list(spec.coord_cols)],
            xlabel=spec.xlabel,
            ylabel=spec.ylabel,
            invert_y=spec.invert_y,
        )
        plot_slice_comparison(
            frame,
            pts.amplitude,
            {
                "NOFE-8D Dims 0-2": emb8_n[:, :3],
                f"NOFE-8D PCA-3 ({100 * sum(n_proj.explained_variance):.0f}% var)": (
                    n_proj.embedding
                ),
                "PCA-8D PCA-3": p_proj.embedding,
                "UMAP-8D PCA-3": u_proj.embedding,
            },
            title=f"3D F3 8-D Embedding Comparison - {spec.key}",
            labels=pts.labels,
            out_path=fig_dir / f"slice_{spec.key}_8d.png",
        )
    return var_info


def _evaluate_candidates(
    g_te: ScaledGraph,
    te_ei_graph: Data,
    candidates: tuple[_EvalCandidate, ...],
) -> list[RowDict]:
    te_ei = graph_edge_index(te_ei_graph)
    rows: list[RowDict] = []
    for item in candidates:
        nm = neighborhood_metrics(
            g_te.features, item.z_te, k=15, sample_size=3000, seed=42
        )
        lm = local_graph_metrics(g_te.features, item.z_te, te_ei)
        rows.append(
            {
                "method": item.name,
                "train_sec": round(item.train_sec, 2),
                "heldout_kl": (
                    round(item.heldout_kl, 6) if item.heldout_kl is not None else None
                ),
                "trustworthiness": round(nm["trustworthiness"], 4),
                "continuity": round(nm["continuity"], 4),
                "knn_recall": round(nm["knn_recall"], 4),
                "local_stress": round(lm["local_stress"], 4),
                "global_stress": round(nm["global_stress"], 4),
            }
        )
    return rows


def main() -> None:
    out_dir = Path("outputs/f3_full_3d_8d")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_f3_directory(Path("f3-full"))["full"]
    if not isinstance(loaded, LazySeismicVolume):
        raise TypeError("Expected f3-full volume to be a LazySeismicVolume")
    ni = loaded.shape.ni
    split_i = round(ni * 0.75)

    stride = Stride3D(11, 20, 6)
    train_pts = volume_to_points(
        loaded, inline_range=IndexRange(0, split_i), stride=stride
    )
    test_pts = volume_to_points(
        loaded, inline_range=IndexRange(split_i, ni), stride=stride
    )
    scaler = SharedStandardizer.fit(train_pts.features, train_pts.locations)
    builder = SeismicPointGraph(mode="radius", radius=0.15, normalize=False)
    g_tr = build_scaled_graph(builder, scaler, train_pts.features, train_pts.locations)
    g_te = build_scaled_graph(builder, scaler, test_pts.features, test_pts.locations)

    trained = fit_and_embed_nofe(
        g_tr.graph,
        NofeHyperparams(
            out_dim=8,
            width=16,
            ker_width=48,
            depth=2,
            nystrom=0.10,
            chunk_size=8192,
            epochs=80,
            patience=20,
            lr=1e-2,
            seed=42,
        ),
    )
    z_nofe_te = embed_numpy(trained.model, g_te.graph)

    t0 = time.perf_counter()
    pca8 = PCA(n_components=8, random_state=42).fit(g_tr.features)
    z_pca_tr = pca8.transform(g_tr.features).astype(np.float32)
    z_pca_te = pca8.transform(g_te.features).astype(np.float32)
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    umap8 = UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=42).fit(
        g_tr.features
    )
    z_umap_te = umap8.transform(g_te.features).astype(np.float32)
    t_umap = time.perf_counter() - t0

    rows = _evaluate_candidates(
        g_te,
        g_te.graph,
        (
            _EvalCandidate(
                "NOFE-8D (w16,d2)",
                z_nofe_te,
                trained.elapsed_sec,
                graph_loss(trained.model, g_te.graph),
            ),
            _EvalCandidate("PCA-8D", z_pca_te, t_pca, None),
            _EvalCandidate("UMAP-8D", z_umap_te, t_umap, None),
        ),
    )
    var_info = _render_slice_comparisons(
        loaded,
        _SliceVizModels(
            scaler=scaler,
            nofe8=trained.model,
            pca8=pca8,
            umap8=umap8,
            z_nofe_tr=trained.train_embedding,
            z_pca_tr=z_pca_tr,
        ),
        fig_dir,
    )
    summary = {
        "train_graph": graph_summary(g_tr.graph),
        "test_graph": graph_summary(g_te.graph),
        "nofe8_pca3_var_ratio": var_info,
        "metrics": rows,
    }
    (out_dir / "metrics_8d.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
