"""3D NOFE-8D vs PCA-8D / UMAP-8D comparison and PCA-to-RGB slice visualisation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nofe import NOFE
from sklearn.decomposition import PCA
from umap import UMAP

from nofe_seismic.data import LazySeismicVolume, load_f3_directory, volume_to_points
from nofe_seismic.graph import (
    SeismicPointGraph,
    SharedStandardizer,
    embed_numpy,
    graph_edge_index,
    graph_loss,
    graph_summary,
    train_nofe,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics
from nofe_seismic.viz import plot_slice_comparison, project_embedding_pca


@dataclass(frozen=True)
class SliceSpec:
    key: str
    inline_range: tuple[int, int] | None
    crossline_range: tuple[int, int] | None
    depth_range: tuple[int, int] | None
    stride: tuple[int, int, int]
    coord_cols: tuple[int, int]
    xlabel: str
    ylabel: str
    invert_y: bool


def _render_slice_comparisons(
    vol: LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    pca8: PCA,
    umap8: UMAP,
    *,
    z_nofe_tr: np.ndarray,
    z_pca_tr: np.ndarray,
    fig_dir: Path,
) -> dict[str, list[float]]:
    specs = (
        SliceSpec(
            key="inline_540",
            inline_range=(540, 541),
            crossline_range=None,
            depth_range=None,
            stride=(1, 4, 2),
            coord_cols=(1, 2),
            xlabel="Crossline",
            ylabel="Depth",
            invert_y=True,
        ),
        SliceSpec(
            key="crossline_450",
            inline_range=None,
            crossline_range=(450, 451),
            depth_range=None,
            stride=(4, 1, 2),
            coord_cols=(0, 2),
            xlabel="Inline",
            ylabel="Depth",
            invert_y=True,
        ),
        SliceSpec(
            key="depth_160",
            inline_range=None,
            crossline_range=None,
            depth_range=(160, 161),
            stride=(4, 4, 1),
            coord_cols=(0, 1),
            xlabel="Inline",
            ylabel="Crossline",
            invert_y=False,
        ),
    )

    var_info: dict[str, list[float]] = {}
    builder = SeismicPointGraph(mode="radius", radius=0.15, normalize=False)
    for spec in specs:
        pts = volume_to_points(
            vol,
            inline_range=spec.inline_range,
            crossline_range=spec.crossline_range,
            depth_range=spec.depth_range,
            stride=spec.stride,
        )
        xs = scaler.transform_features(pts.features)
        ps = scaler.transform_locations(pts.locations)
        g_sl = builder.build(xs, ps)
        emb8_n = embed_numpy(nofe8, g_sl)
        emb8_p = pca8.transform(xs).astype(np.float32)
        emb8_u = umap8.transform(xs).astype(np.float32)

        n_pca3, n_var = project_embedding_pca(
            emb8_n, fit_embedding=z_nofe_tr, n_components=3
        )
        p_pca3, _ = project_embedding_pca(
            emb8_p, fit_embedding=z_pca_tr, n_components=3
        )
        u_pca3, _ = project_embedding_pca(emb8_u, n_components=3)
        var_info[spec.key] = [round(float(v), 4) for v in n_var]

        plot_slice_comparison(
            pts.locations[:, spec.coord_cols],
            pts.amplitude,
            {
                "NOFE-8D Dims 0-2": emb8_n[:, :3],
                f"NOFE-8D PCA-3 ({100 * sum(n_var):.0f}% var)": n_pca3,
                "PCA-8D PCA-3": p_pca3,
                "UMAP-8D PCA-3": u_pca3,
            },
            title=f"3D F3 8-D Embedding Comparison - {spec.key}",
            xlabel=spec.xlabel,
            ylabel=spec.ylabel,
            labels=pts.labels,
            invert_y=spec.invert_y,
            out_path=fig_dir / f"slice_{spec.key}_8d.png",
        )
    return var_info


def main() -> None:
    out_dir = Path("outputs/f3_full_3d_8d")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_f3_directory(Path("f3-full"))["full"]
    if not isinstance(loaded, LazySeismicVolume):
        raise TypeError("Expected f3-full volume to be a LazySeismicVolume")
    ni, _, _ = loaded.shape
    split_i = round(ni * 0.75)

    train_pts = volume_to_points(loaded, inline_range=(0, split_i), stride=(11, 20, 6))
    test_pts = volume_to_points(loaded, inline_range=(split_i, ni), stride=(11, 20, 6))

    scaler = SharedStandardizer.fit(train_pts.features, train_pts.locations)
    x_tr = scaler.transform_features(train_pts.features)
    x_te = scaler.transform_features(test_pts.features)

    builder = SeismicPointGraph(mode="radius", radius=0.15, normalize=False)
    g_tr = builder.build(x_tr, scaler.transform_locations(train_pts.locations))
    g_te = builder.build(x_te, scaler.transform_locations(test_pts.locations))

    t0 = time.perf_counter()
    nofe8 = NOFE(
        out_dim=8,
        width=16,
        ker_width=48,
        depth=2,
        nystrom=0.10,
        chunk_size=8192,
        device="cpu",
        seed=42,
    )
    train_nofe(nofe8, g_tr, epochs=80, patience=20, lr=1e-2, seed=42)
    z_nofe_tr = embed_numpy(nofe8, g_tr)
    z_nofe_te = embed_numpy(nofe8, g_te)
    t_nofe = time.perf_counter() - t0

    t0 = time.perf_counter()
    pca8 = PCA(n_components=8, random_state=42).fit(x_tr)
    z_pca_tr = pca8.transform(x_tr).astype(np.float32)
    z_pca_te = pca8.transform(x_te).astype(np.float32)
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    umap8 = UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=42).fit(
        x_tr
    )
    z_umap_te = umap8.transform(x_te).astype(np.float32)
    t_umap = time.perf_counter() - t0

    te_ei = graph_edge_index(g_te)
    eval_inputs: tuple[tuple[str, np.ndarray, float, float | None], ...] = (
        ("NOFE-8D (w16,d2)", z_nofe_te, t_nofe, graph_loss(nofe8, g_te)),
        ("PCA-8D", z_pca_te, t_pca, None),
        ("UMAP-8D", z_umap_te, t_umap, None),
    )
    rows = []
    for name, z_te, dt, kl in eval_inputs:
        nm = neighborhood_metrics(x_te, z_te, k=15, sample_size=3000, seed=42)
        lm = local_graph_metrics(x_te, z_te, te_ei)
        rows.append(
            {
                "method": name,
                "train_sec": round(dt, 2),
                "heldout_kl": round(kl, 6) if kl is not None else None,
                "trustworthiness": round(nm["trustworthiness"], 4),
                "continuity": round(nm["continuity"], 4),
                "knn_recall": round(nm["knn_recall"], 4),
                "local_stress": round(lm["local_stress"], 4),
                "global_stress": round(nm["global_stress"], 4),
            }
        )

    var_info = _render_slice_comparisons(
        loaded,
        scaler,
        nofe8,
        pca8,
        umap8,
        z_nofe_tr=z_nofe_tr,
        z_pca_tr=z_pca_tr,
        fig_dir=fig_dir,
    )
    summary = {
        "train_graph": graph_summary(g_tr),
        "test_graph": graph_summary(g_te),
        "nofe8_pca3_var_ratio": var_info,
        "metrics": rows,
    }
    (out_dir / "metrics_8d.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
