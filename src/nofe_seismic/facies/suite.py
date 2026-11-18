"""Latent-dim ablations and facies separation suite orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume
from nofe_seismic.types import Stride2D

from .evals import (
    AblationFaciesConfig,
    run_2d_latent_ablation,
    run_3d_facies_and_ablation,
)
from .metrics_plots import (
    plot_fig1_latent_ablation,
    plot_fig2_facies_breakthrough,
)
from .spatial_plots import (
    plot_fig3_gallery,
    plot_fig4_hero_3d_cube,
    plot_fig5_trace_profiles_and_clusters,
)


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
) -> dict[str, object]:
    """Run 2D/3D latent-dim ablations, 3D facies separation and figures."""
    cfg = AblationFaciesConfig(
        out_dir=out_dir,
        device=device,
        seed=seed,
        stride_2d=Stride2D(int(stride_2d[0]), int(stride_2d[1])),
        radius_2d=radius_2d,
        dims_2d=dims_2d,
        dims_3d=dims_3d,
        epochs_2d=epochs_2d,
        epochs_3d=epochs_3d,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ab2d = run_2d_latent_ablation(volume, cfg)
    res3d = run_3d_facies_and_ablation(volume, cfg)

    plot_fig1_latent_ablation(
        ab2d.rows, res3d.rows, fig_dir / "fig1_latent_dim_ablation.png"
    )
    plot_fig2_facies_breakthrough(
        volume,
        res3d,
        seed=seed,
        out_path=fig_dir / "fig2_facies_separation_breakthrough.png",
    )
    plot_fig3_gallery(
        volume,
        ab2d,
        res3d,
        cfg,
        fig_dir / "fig3_multiscale_maps_and_cross_section.png",
    )
    plot_fig4_hero_3d_cube(
        volume,
        res3d,
        seed=seed,
        out_path=fig_dir / "fig4_hero_3d_cutaway_and_facies.png",
    )
    trace_clust = plot_fig5_trace_profiles_and_clusters(
        volume,
        ab2d,
        cfg,
        fig_dir / "fig5_trace_profiles_and_clusters.png",
    )

    pd.DataFrame(ab2d.rows).to_csv(out_dir / "latent_ablation_2d.csv", index=False)
    pd.DataFrame(res3d.rows).to_csv(
        out_dir / "latent_ablation_3d_facies.csv", index=False
    )
    summary: dict[str, object] = {
        "ablation_2d": ab2d.rows,
        "ablation_3d_facies": res3d.rows,
        "facies_breakthrough": {
            "naive_nofe_8d": res3d.naive_fac.to_summary_dict(),
            f"tuned_nofe_{res3d.hero_dim}d": res3d.tuned_fac_hero.to_summary_dict(),
        },
        "trace_clusters_2d": trace_clust,
    }
    (out_dir / "ablation_and_facies_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
