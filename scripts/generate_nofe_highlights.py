"""Compute operator-generalisation benchmarks and summary figures for NOFE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from nofe import NOFE
from sklearn.decomposition import PCA
from umap import UMAP

from nofe_seismic.data.traces import TraceLogs, extract_trace_logs
from nofe_seismic.data.volumes import (
    LazySeismicVolume,
    SeismicVolume,
    load_f3_directory,
)
from nofe_seismic.graph import (
    ScaledGraph,
    SeismicPointGraph,
    SharedStandardizer,
    build_scaled_graph,
    coord_map_2d,
    embed_numpy,
    fit_and_embed_nofe,
    graph_edge_index,
    graph_summary,
)
from nofe_seismic.metrics import local_graph_metrics, neighborhood_metrics, norm_rmse
from nofe_seismic.types import (
    IndexRange,
    NofeHyperparams,
    RowDict,
    Stride2D,
    extent_from_locations,
)
from nofe_seismic.viz.slices import embedding_to_rgb, project_embedding_pca

_SPLIT_INLINE = 450
_STRIDE_2D = Stride2D(3, 3)


@dataclass(frozen=True)
class _MethodTrainTestEmbed:
    name: str
    z_train: np.ndarray
    z_test: np.ndarray


@dataclass(frozen=True)
class _ResolutionTransfer2D:
    metrics: RowDict
    zn_native: np.ndarray
    ite: list[int]
    inat: list[int]


@dataclass(frozen=True)
class _Figure2Data:
    dc_rows: dict[str, dict[str, float]]
    f3_3d8: dict[str, dict[str, float]]
    stitch_2d: RowDict
    res_transfer: _ResolutionTransfer2D
    z_nofe_te: np.ndarray


def _compute_generalization_gap(
    g_tr: ScaledGraph,
    g_te: ScaledGraph,
    methods: tuple[_MethodTrainTestEmbed, ...],
) -> list[RowDict]:
    ei_tr = graph_edge_index(g_tr.graph)
    ei_te = graph_edge_index(g_te.graph)
    rows: list[RowDict] = []
    for item in methods:
        nm_tr = neighborhood_metrics(
            g_tr.features, item.z_train, k=15, sample_size=3000, seed=42
        )
        nm_te = neighborhood_metrics(
            g_te.features, item.z_test, k=15, sample_size=3000, seed=42
        )
        lm_tr = local_graph_metrics(g_tr.features, item.z_train, ei_tr)
        lm_te = local_graph_metrics(g_te.features, item.z_test, ei_te)
        rows.append(
            {
                "method": item.name,
                "train_trust": round(nm_tr["trustworthiness"], 4),
                "heldout_trust": round(nm_te["trustworthiness"], 4),
                "trust_drop": round(
                    nm_tr["trustworthiness"] - nm_te["trustworthiness"], 4
                ),
                "train_local_stress": round(lm_tr["local_stress"], 4),
                "heldout_local_stress": round(lm_te["local_stress"], 4),
                "local_stress_increase": round(
                    lm_te["local_stress"] - lm_tr["local_stress"], 4
                ),
            }
        )
    return rows


def _compute_patch_stitching_2d(
    vol: SeismicVolume | LazySeismicVolume,
    sc: SharedStandardizer,
    nofe8: NOFE,
) -> RowDict:
    ni, nx, _ = vol.shape
    ov_lo = (int(0.35 * nx) // 3) * 3
    ov_hi = int(0.65 * nx)
    margin = max(6, round(0.05 * nx / np.sqrt(12)))
    left = extract_trace_logs(
        vol,
        inline_range=IndexRange(_SPLIT_INLINE, ni),
        crossline_range=IndexRange(0, ov_hi),
        stride_2d=_STRIDE_2D,
        feature_mode="envelope",
    )
    right = extract_trace_logs(
        vol,
        inline_range=IndexRange(_SPLIT_INLINE, ni),
        crossline_range=IndexRange(ov_lo, nx),
        stride_2d=_STRIDE_2D,
        feature_mode="envelope",
    )
    builder = SeismicPointGraph(mode="radius", radius=0.05, normalize=False)
    gl = build_scaled_graph(builder, sc, left.traces, left.locations)
    gr = build_scaled_graph(builder, sc, right.traces, right.locations)

    zn_l, zn_r = embed_numpy(nofe8, gl.graph), embed_numpy(nofe8, gr.graph)
    zp_l = (
        PCA(n_components=8, random_state=42)
        .fit_transform(gl.features)
        .astype(np.float32)
    )
    zp_r = (
        PCA(n_components=8, random_state=42)
        .fit_transform(gr.features)
        .astype(np.float32)
    )
    zu_l = (
        UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=42)
        .fit_transform(gl.features)
        .astype(np.float32)
    )
    zu_r = (
        UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=42)
        .fit_transform(gr.features)
        .astype(np.float32)
    )

    ml, mr = coord_map_2d(left.locations), coord_map_2d(right.locations)
    shared = [k for k in ml if k in mr and (ov_lo + margin) <= k[1] <= (ov_hi - margin)]
    il, ir = [ml[k] for k in shared], [mr[k] for k in shared]
    return {
        "shared_interior_traces": len(shared),
        "NOFE-8D": round(norm_rmse(zn_l[il], zn_r[ir]), 4),
        "PCA-8D": round(norm_rmse(zp_l[il], zp_r[ir]), 4),
        "UMAP-8D": round(norm_rmse(zu_l[il], zu_r[ir]), 4),
    }


def _annotate_bars(
    ax: Axes, bars: BarContainer, vals: list[float], *, bold_first: bool = False
) -> None:
    for idx, (bar, val) in enumerate(zip(bars, vals, strict=True)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5 if not bold_first else 9,
            fontweight="bold" if (bold_first and idx == 0) else "normal",
        )


def _plot_figure_1(
    dc_rows: dict[str, dict[str, float]],
    fig_dir: Path,
) -> None:
    methods_8d = ("NOFE-8D", "PCA-8D", "UMAP-8D")
    bar_colors = ("#1b7837", "#762a83", "#d95f02")
    fig, raw_axes = plt.subplots(1, 4, figsize=(14.5, 3.8), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    specs = (
        ("local_stress", "Local Spatial Stress (lower better)"),
        ("global_stress", "Global Stress (lower better)"),
        ("depth_recon_mlp_r2", "Depth Log Recon R2 (higher better)"),
        ("continuity", "Continuity (higher better)"),
    )
    for ax, (key, title) in zip(axes, specs, strict=True):
        vals = [float(dc_rows[m][key]) for m in methods_8d]
        bars = ax.bar(methods_8d, vals, color=bar_colors, width=0.58)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(min(0.0, *vals), max(vals) * 1.14)
        _annotate_bars(ax, bars, vals, bold_first=True)
    fig.suptitle(
        "2D Depth-Log Compression (253 -> 8 dims on F3): NOFE-8D vs. Baselines",
        fontsize=12,
    )
    fig.savefig(fig_dir / "fig1_depth_compression_8d_bars.png", dpi=180)
    plt.close(fig)


def _plot_figure_2(data: _Figure2Data, fig_dir: Path) -> None:
    bar_colors = ("#1b7837", "#762a83", "#d95f02")
    fig, raw_axes = plt.subplots(1, 3, figsize=(14.5, 4.0), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())

    m_st = ("NOFE-8D\n(2D)", "PCA-8D\n(2D)", "UMAP-8D\n(2D)")
    v_st = [
        float(data.stitch_2d["NOFE-8D"] or 0.0),
        float(data.stitch_2d["PCA-8D"] or 0.0),
        float(data.stitch_2d["UMAP-8D"] or 0.0),
    ]
    b2 = axes[0].bar(m_st, v_st, color=bar_colors, width=0.55)
    axes[0].set_title(
        "2D Tile Patch-Stitching Error\n(Norm. RMSE ↓)", fontsize=10, fontweight="bold"
    )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.0, max(v_st) * 1.14)
    _annotate_bars(axes[0], b2, v_st)

    labels_s = (
        "NOFE-8D\n(2D)",
        "PCA-8D\n(2D)",
        "UMAP-8D\n(2D)",
        "NOFE-8D\n(3D)",
        "UMAP-8D\n(3D)",
    )
    vals_s = [
        data.dc_rows["NOFE-8D"]["local_stress"],
        data.dc_rows["PCA-8D"]["local_stress"],
        data.dc_rows["UMAP-8D"]["local_stress"],
        data.f3_3d8["NOFE-8D (w16,d2)"]["local_stress"],
        data.f3_3d8["UMAP-8D"]["local_stress"],
    ]
    b3 = axes[1].bar(
        labels_s,
        vals_s,
        color=("#1b7837", "#762a83", "#d95f02", "#1b7837", "#d95f02"),
        width=0.6,
    )
    axes[1].set_title(
        "Held-Out Local Spatial Stress\n(2D & 3D F3, ↓)", fontsize=10, fontweight="bold"
    )
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0.0, max(vals_s) * 1.14)
    _annotate_bars(axes[1], b3, vals_s)

    res = data.res_transfer
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(res.ite), size=min(2000, len(res.ite)), replace=False)
    x_plot = data.z_nofe_te[res.ite][sample_idx, 0]
    y_plot = res.zn_native[res.inat][sample_idx, 0]
    axes[2].scatter(x_plot, y_plot, s=6, alpha=0.35, color="#1b7837")
    lims = [
        float(min(x_plot.min(), y_plot.min())),
        float(max(x_plot.max(), y_plot.max())),
    ]
    axes[2].plot(lims, lims, "k--", linewidth=1.2)
    axes[2].set_title(
        "Zero-Shot 9x Density Transfer\n"
        f"(deg {res.metrics['sparse_mean_degree']} -> "
        f"{res.metrics['native_mean_degree']}, "
        f"r={res.metrics['pearson_r']:.4f})",
        fontsize=10,
        fontweight="bold",
    )
    axes[2].set_xlabel("Sparse (3x3) NOFE-8D dim 0")
    axes[2].set_ylabel("Native (1x1) NOFE-8D dim 0")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "Continuous Operator Properties on F3: Stitching, Stress & Density Transfer",
        fontsize=12,
    )
    fig.savefig(fig_dir / "fig2_operator_generalization.png", dpi=180)
    plt.close(fig)


def _plot_figure_3(
    full: TraceLogs,
    *,
    z_nofe_full: np.ndarray,
    z_nofe_tr: np.ndarray,
    z_umap_full: np.ndarray,
    z_umap_tr: np.ndarray,
    fig_dir: Path,
) -> None:
    gi, gx = full.grid_shape
    extent = extent_from_locations(full.locations)
    nofe_rgb = embedding_to_rgb(
        project_embedding_pca(z_nofe_full, fit_embedding=z_nofe_tr).embedding
    ).reshape(gi, gx, 3)
    umap_rgb = embedding_to_rgb(
        project_embedding_pca(z_umap_full, fit_embedding=z_umap_tr).embedding
    ).reshape(gi, gx, 3)
    deep_img = (
        full.deepest_facies.reshape(gi, gx)
        if full.deepest_facies is not None
        else full.rms_amp.reshape(gi, gx)
    )

    fig, raw_axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    axes: list[Axes] = list(np.atleast_1d(raw_axes).ravel())
    axes[0].imshow(deep_img, cmap="tab10", vmin=0, vmax=9, extent=extent, aspect="auto")
    axes[0].set_title("Ground Truth Deep Facies")
    axes[1].imshow(nofe_rgb, extent=extent, aspect="auto")
    axes[1].set_title("NOFE-8D (Smooth across boundary)")
    axes[2].imshow(umap_rgb, extent=extent, aspect="auto")
    axes[2].set_title("UMAP-8D (Out-of-sample speckle)")
    for ax in axes:
        ax.axhline(_SPLIT_INLINE, color="white", linestyle="--", linewidth=1.4)
        ax.set_ylim(600, 300)
        ax.set_xlabel("Crossline")
    axes[0].set_ylabel("Inline (dashed = unseen split at 450)")
    fig.suptitle(
        "Visualising Out-of-Sample Generalisation on F3 (Inlines 300-600)", fontsize=12
    )
    fig.savefig(fig_dir / "fig3_out_of_sample_visual_proof.png", dpi=180)
    plt.close(fig)


def _compute_resolution_transfer_2d(
    vol: SeismicVolume | LazySeismicVolume,
    sc: SharedStandardizer,
    nofe8: NOFE,
    te: TraceLogs,
    *,
    g_te: ScaledGraph,
    z_nofe_te: np.ndarray,
    t_nofe: float,
) -> _ResolutionTransfer2D:
    ni = vol.shape.ni
    builder = SeismicPointGraph(mode="radius", radius=0.05, normalize=False)
    native = extract_trace_logs(
        vol,
        inline_range=IndexRange(_SPLIT_INLINE, min(ni, _SPLIT_INLINE + 45)),
        stride_2d=Stride2D(1, 1),
        feature_mode="envelope",
    )
    gn = build_scaled_graph(builder, sc, native.traces, native.locations)
    zn_native = embed_numpy(nofe8, gn.graph)
    m_te, m_nat = coord_map_2d(te.locations), coord_map_2d(native.locations)
    shared_res = [k for k in m_te if k in m_nat]
    ite = [m_te[k] for k in shared_res]
    inat = [m_nat[k] for k in shared_res]
    res_corr_2d = float(
        np.corrcoef(z_nofe_te[ite].ravel(), zn_native[inat].ravel())[0, 1]
    )
    return _ResolutionTransfer2D(
        metrics={
            "shared_traces": len(shared_res),
            "sparse_mean_degree": round(
                float(graph_summary(g_te.graph)["mean_degree"]), 1
            ),
            "native_mean_degree": round(
                float(graph_summary(gn.graph)["mean_degree"]), 1
            ),
            "norm_rmse": round(norm_rmse(zn_native[inat], z_nofe_te[ite]), 4),
            "pearson_r": round(res_corr_2d, 4),
            "nofe_train_sec": round(t_nofe, 2),
        },
        zn_native=zn_native,
        ite=ite,
        inat=inat,
    )


def main() -> None:
    out_dir = Path("outputs/nofe_highlights")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dc_summary = json.loads(
        Path(
            "outputs/f3_full_depth_compression_8d_tuned/depth_compression_summary.json"
        ).read_text()
    )
    f3_3d_summary = json.loads(Path("outputs/f3_full/metrics_summary.json").read_text())
    f3_3d_8d_summary = json.loads(
        Path("outputs/f3_full_3d_8d/metrics_8d.json").read_text()
    )

    vol = load_f3_directory(Path("f3-full"))["full"]
    ni = vol.shape.ni
    tr = extract_trace_logs(
        vol,
        inline_range=IndexRange(0, _SPLIT_INLINE),
        stride_2d=_STRIDE_2D,
        feature_mode="envelope",
    )
    te = extract_trace_logs(
        vol,
        inline_range=IndexRange(_SPLIT_INLINE, ni),
        stride_2d=_STRIDE_2D,
        feature_mode="envelope",
    )
    full = extract_trace_logs(vol, stride_2d=_STRIDE_2D, feature_mode="envelope")

    sc = SharedStandardizer.fit(tr.traces, tr.locations, feature_scale="global")
    builder = SeismicPointGraph(mode="radius", radius=0.05, normalize=False)
    g_tr = build_scaled_graph(builder, sc, tr.traces, tr.locations)
    g_te = build_scaled_graph(builder, sc, te.traces, te.locations)
    g_full = build_scaled_graph(builder, sc, full.traces, full.locations)

    trained = fit_and_embed_nofe(
        g_tr.graph,
        NofeHyperparams(
            out_dim=8,
            width=24,
            ker_width=64,
            depth=2,
            nystrom=0.10,
            chunk_size=8192,
            epochs=100,
            patience=25,
            lr=1e-2,
            seed=42,
        ),
    )
    z_nofe_te = embed_numpy(trained.model, g_te.graph)
    z_nofe_full = embed_numpy(trained.model, g_full.graph)

    pca8 = PCA(n_components=8, random_state=42).fit(g_tr.features)
    umap8 = UMAP(n_components=8, n_neighbors=15, min_dist=0.1, random_state=42).fit(
        g_tr.features
    )
    z_umap_tr = umap8.transform(g_tr.features).astype(np.float32)

    gen_gap_rows = _compute_generalization_gap(
        g_tr,
        g_te,
        (
            _MethodTrainTestEmbed("NOFE-8D", trained.train_embedding, z_nofe_te),
            _MethodTrainTestEmbed(
                "PCA-8D",
                pca8.transform(g_tr.features).astype(np.float32),
                pca8.transform(g_te.features).astype(np.float32),
            ),
            _MethodTrainTestEmbed(
                "UMAP-8D", z_umap_tr, umap8.transform(g_te.features).astype(np.float32)
            ),
        ),
    )
    stitch_2d = _compute_patch_stitching_2d(vol, sc, trained.model)
    res_transfer = _compute_resolution_transfer_2d(
        vol,
        sc,
        trained.model,
        te,
        g_te=g_te,
        z_nofe_te=z_nofe_te,
        t_nofe=trained.elapsed_sec,
    )

    dc_rows = {r["method"]: r for r in dc_summary["metrics"]}
    f3_3d8 = {r["method"]: r for r in f3_3d_8d_summary["metrics"]}
    _plot_figure_1(dc_rows, fig_dir)
    _plot_figure_2(
        _Figure2Data(
            dc_rows=dc_rows,
            f3_3d8=f3_3d8,
            stitch_2d=stitch_2d,
            res_transfer=res_transfer,
            z_nofe_te=z_nofe_te,
        ),
        fig_dir,
    )
    _plot_figure_3(
        full,
        z_nofe_full=z_nofe_full,
        z_nofe_tr=trained.train_embedding,
        z_umap_full=umap8.transform(g_full.features).astype(np.float32),
        z_umap_tr=z_umap_tr,
        fig_dir=fig_dir,
    )

    summary = {
        "depth_compression_8d_tuned": dc_summary["metrics"],
        "f3_3d_8d": f3_3d_8d_summary["metrics"],
        "f3_3d_patch_stitching": f3_3d_summary["patch_stitching"],
        "f3_3d_resolution_consistency": f3_3d_summary["resolution_consistency"],
        "generalization_gap_2d": gen_gap_rows,
        "patch_stitching_2d": stitch_2d,
        "resolution_transfer_2d_9x": res_transfer.metrics,
    }
    (out_dir / "highlights_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(gen_gap_rows).to_csv(
        out_dir / "generalization_gap_2d.csv", index=False
    )
    print(
        json.dumps(
            {
                "patch_stitching_2d": stitch_2d,
                "resolution_transfer_2d_9x": res_transfer.metrics,
                "generalization_gap_2d": gen_gap_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
