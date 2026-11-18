"""End-to-end horizon boundary reconstruction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

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
    fit_and_embed_nofe,
    graph_summary,
)
from nofe_seismic.types import IndexRange, NofeHyperparams, RowDict, TrainedNofe

from .config import HorizonConfig
from .evals import (
    NativeSlice,
    ProbeEvalResult,
    SliceEmbedModels,
    UnsupervisedEvalResult,
    embed_slice_native_z,
    eval_probe_horizons,
    eval_unsupervised_and_rgt,
    score_discovered_horizons,
)
from .plots import (
    plot_fig1_unsupervised,
    plot_fig2_tracking_comparison,
    plot_fig3_robustness,
    plot_fig4_3d_surfaces,
)
from .robustness import (
    TileSeamResult,
    evaluate_3d_slab,
    evaluate_noise_stress,
    evaluate_tile_seam,
)

_LARGE_INLINE_THRESHOLD = 600
_LARGE_CROSSLINE_THRESHOLD = 900
_HELD_OUT_INLINE = 540
_HELD_OUT_CROSSLINE = 450


@dataclass(frozen=True)
class _TrainedHorizonState:
    train_points: PointCloud
    train_graph_stats: dict[str, float | int]
    train_features: np.ndarray
    train_pca: np.ndarray
    trained_nofe: TrainedNofe
    models: SliceEmbedModels


@dataclass(frozen=True)
class _HorizonFigureInputs:
    sl_in: NativeSlice
    sl_xl: NativeSlice
    unsup_in: UnsupervisedEvalResult
    unsup_xl: UnsupervisedEvalResult
    probe_in: ProbeEvalResult | None
    probe_xl: ProbeEvalResult | None
    seam_info: TileSeamResult
    stress_rows: list[RowDict]
    z_tr_nofe: np.ndarray
    inline_idx: int
    unit_ids: list[int]


def _hyperparams_from_config(cfg: HorizonConfig) -> NofeHyperparams:
    return NofeHyperparams(
        out_dim=cfg.out_dim,
        width=cfg.width,
        ker_width=cfg.ker_width,
        depth=cfg.depth,
        nystrom=cfg.nystrom,
        chunk_size=cfg.chunk_size,
        device=cfg.device,
        seed=cfg.seed,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
    )


def _train_horizon_models(
    volume: SeismicVolume | LazySeismicVolume,
    split_i: int,
    cfg: HorizonConfig,
) -> _TrainedHorizonState:
    tr_pts = volume_to_points(
        volume, inline_range=IndexRange(0, split_i), stride=cfg.train_stride
    )
    scaler = SharedStandardizer.fit(tr_pts.features, tr_pts.locations)
    train_scaled = build_scaled_graph(
        SeismicPointGraph(mode="radius", radius=cfg.radius_train, normalize=False),
        scaler,
        tr_pts.features,
        tr_pts.locations,
    )
    trained = fit_and_embed_nofe(train_scaled.graph, _hyperparams_from_config(cfg))
    pca8 = PCA(
        n_components=min(cfg.out_dim, train_scaled.features.shape[1]),
        random_state=cfg.seed,
    ).fit(train_scaled.features)
    z_tr_pca = pca8.transform(train_scaled.features).astype(np.float32)
    return _TrainedHorizonState(
        train_points=tr_pts,
        train_graph_stats=graph_summary(train_scaled.graph),
        train_features=train_scaled.features,
        train_pca=z_tr_pca,
        trained_nofe=trained,
        models=SliceEmbedModels(scaler=scaler, nofe=trained.model, pca=pca8),
    )


def _render_horizon_figures(
    inputs: _HorizonFigureInputs,
    cfg: HorizonConfig,
    fig_dir: Path,
) -> None:
    plot_fig1_unsupervised(
        inputs.sl_in,
        inputs.unsup_in.artifacts["NOFE-8D"],
        inputs.z_tr_nofe,
        slice_name=f"Held-Out Inline {inputs.inline_idx}",
        out_path=fig_dir / "fig1_unsupervised_boundary_reconstruction.png",
        seed=cfg.seed,
    )
    if inputs.probe_in is not None and inputs.probe_xl is not None:
        plot_fig2_tracking_comparison(
            inputs.sl_in,
            inputs.sl_xl,
            unsup_in=inputs.unsup_in.artifacts,
            unsup_xl=inputs.unsup_xl.artifacts,
            probe_in=inputs.probe_in.artifacts,
            probe_xl=inputs.probe_xl.artifacts,
            unit_ids=inputs.unit_ids,
            out_path=fig_dir / "fig2_horizon_tracking_and_probe.png",
        )
    plot_fig3_robustness(
        inputs.sl_in,
        inputs.seam_info,
        inputs.stress_rows,
        cfg,
        fig_dir / "fig3_noise_and_seam_robustness.png",
    )


def _save_horizon_outputs(
    out_dir: Path,
    tables: dict[str, list[RowDict]],
    summary: dict[str, object],
) -> None:
    for fname, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_dir / fname, index=False)
    (out_dir / "horizon_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def run_horizon_reconstruction(
    volume: SeismicVolume | LazySeismicVolume,
    cfg: HorizonConfig,
) -> dict[str, object]:
    """Train 3D NOFE-8D on training inlines and evaluate horizon reconstruction."""
    out_dir = cfg.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ni, nx, _ = volume.shape
    split_i = max(8, round(ni * cfg.train_inline_frac))
    state = _train_horizon_models(volume, split_i, cfg)

    inline_idx = (
        _HELD_OUT_INLINE
        if ni >= _LARGE_INLINE_THRESHOLD
        else min(ni - 2, split_i + max(1, (ni - split_i) // 2))
    )
    cross_idx = _HELD_OUT_CROSSLINE if nx >= _LARGE_CROSSLINE_THRESHOLD else nx // 2
    sl_in = embed_slice_native_z(
        volume,
        state.models,
        axis="inline",
        index=inline_idx,
        lateral_stride=cfg.slice_lateral_stride,
        radius=cfg.radius_slice,
    )
    sl_xl = embed_slice_native_z(
        volume,
        state.models,
        axis="crossline",
        index=cross_idx,
        lateral_stride=cfg.slice_lateral_stride,
        radius=cfg.radius_slice,
    )

    tr_labels = state.train_points.labels
    present = (
        sorted(int(c) for c in np.unique(tr_labels))
        if tr_labels is not None
        else [0, 1, 2]
    )
    unit_ids = [c for c in present if c > present[0]]

    unsup_in = eval_unsupervised_and_rgt(
        sl_in, f"inline_{inline_idx}", cfg, unit_ids=unit_ids
    )
    unsup_xl = eval_unsupervised_and_rgt(
        sl_xl, f"crossline_{cross_idx}", cfg, unit_ids=unit_ids
    )

    probe_in: ProbeEvalResult | None = None
    probe_xl: ProbeEvalResult | None = None
    probes_cache: dict[str, Pipeline] | None = None
    probe_rows: list[RowDict] = []
    if tr_labels is not None:
        train_embeds = {
            "NOFE-8D": state.trained_nofe.train_embedding,
            "PCA-8D": state.train_pca,
            "Raw-11D": state.train_features,
        }
        probe_in = eval_probe_horizons(
            train_embeds,
            tr_labels,
            sl_in,
            unsup_in.artifacts,
            slice_key=f"inline_{inline_idx}",
            cfg=cfg,
            unit_ids=unit_ids,
        )
        probe_xl = eval_probe_horizons(
            train_embeds,
            tr_labels,
            sl_xl,
            unsup_xl.artifacts,
            slice_key=f"crossline_{cross_idx}",
            cfg=cfg,
            unit_ids=unit_ids,
            probes_cache=probe_in.probes,
        )
        probe_rows = probe_in.rows + probe_xl.rows
        probes_cache = probe_in.probes

    stress_rows = evaluate_noise_stress(
        volume,
        state.models,
        inline_idx=inline_idx,
        sl_clean=sl_in,
        cfg=cfg,
        unit_ids=unit_ids,
    )
    seam_info = evaluate_tile_seam(
        volume,
        state.models.scaler,
        state.trained_nofe.model,
        inline_idx=inline_idx,
        sl_clean=sl_in,
        cfg=cfg,
    )
    slab_res = evaluate_3d_slab(
        volume,
        state.models.scaler,
        state.trained_nofe.model,
        split_i=split_i,
        unit_ids=unit_ids,
        probes_cache=probes_cache,
        cfg=cfg,
    )
    _render_horizon_figures(
        _HorizonFigureInputs(
            sl_in=sl_in,
            sl_xl=sl_xl,
            unsup_in=unsup_in,
            unsup_xl=unsup_xl,
            probe_in=probe_in,
            probe_xl=probe_xl,
            seam_info=seam_info,
            stress_rows=stress_rows,
            z_tr_nofe=state.trained_nofe.train_embedding,
            inline_idx=inline_idx,
            unit_ids=unit_ids,
        ),
        cfg,
        fig_dir,
    )
    if slab_res.plot_data is not None:
        plot_fig4_3d_surfaces(
            slab_res.plot_data, fig_dir / "fig4_3d_horizon_surfaces.png"
        )

    unsup_rows = unsup_in.unsup_rows + unsup_xl.unsup_rows
    rgt_rows = unsup_in.rgt_rows + unsup_xl.rgt_rows
    summary: dict[str, object] = {
        "dataset": volume.name,
        "volume_shape": list(volume.shape),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        },
        "nofe_train_sec": round(state.trained_nofe.elapsed_sec, 2),
        "history_last": state.trained_nofe.history[-1]
        if state.trained_nofe.history
        else None,
        "train_graph": state.train_graph_stats,
        "unsupervised_boundaries": unsup_rows,
        "rgt_horizon_tracking": rgt_rows,
        "zero_seed_discovered_horizons": score_discovered_horizons(
            sl_in, unsup_in.artifacts["NOFE-8D"].discovered, unit_ids
        ),
        "probe_horizon_reconstruction": probe_rows,
        "robustness_stress": stress_rows,
        "tile_seam_consistency": {
            "shared_interior_points": seam_info.shared_interior_points,
            "nofe_seam_norm_rmse": seam_info.nofe_seam_norm_rmse,
            "pca_tile_seam_norm_rmse": seam_info.pca_tile_seam_norm_rmse,
        },
        "heldout_3d_slab": slab_res.metrics,
    }
    _save_horizon_outputs(
        out_dir,
        {
            "unsupervised_boundaries.csv": unsup_rows,
            "rgt_horizon_tracking.csv": rgt_rows,
            "probe_horizon_reconstruction.csv": probe_rows,
            "robustness_stress.csv": stress_rows,
        },
        summary,
    )
    return summary
