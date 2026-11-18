"""Smoke tests for latent-dimension ablations and 3D facies separation."""

from __future__ import annotations

import math
from pathlib import Path

from nofe_seismic.ablations_and_facies import run_ablation_and_facies_suite
from nofe_seismic.data import make_synthetic_seismic_cube


def test_ablation_and_facies_suite_smoke(tmp_path: Path) -> None:
    volume = make_synthetic_seismic_cube(shape=(24, 24, 32), seed=7)
    out_dir = tmp_path / "ablations"
    summary = run_ablation_and_facies_suite(
        volume,
        out_dir=out_dir,
        device="cpu",
        seed=7,
        stride_2d=(1, 1),
        radius_2d=0.45,
        dims_2d=(2, 4, 8),
        dims_3d=(3, 8),
        epochs_2d=3,
        epochs_3d=3,
    )

    assert (out_dir / "ablation_and_facies_summary.json").exists()
    assert (out_dir / "latent_ablation_2d.csv").exists()
    assert (out_dir / "latent_ablation_3d_facies.csv").exists()

    fig_dir = out_dir / "figures"
    for name in (
        "fig1_latent_dim_ablation.png",
        "fig2_facies_separation_breakthrough.png",
        "fig3_multiscale_maps_and_cross_section.png",
        "fig4_hero_3d_cutaway_and_facies.png",
        "fig5_trace_profiles_and_clusters.png",
    ):
        p = fig_dir / name
        assert p.exists(), f"Missing {p}"
        assert p.stat().st_size > 1024

    assert len(summary["ablation_2d"]) >= 6
    assert len(summary["ablation_3d_facies"]) >= 5
    for row in summary["ablation_2d"]:
        assert math.isfinite(row["local_stress"])
    for row in summary["ablation_3d_facies"]:
        assert math.isfinite(row["lr_bal_acc"])
