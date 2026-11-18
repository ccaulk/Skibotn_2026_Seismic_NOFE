"""Tests for the hard evaluation suite and advanced visualisations."""

from __future__ import annotations

import math
from pathlib import Path

from nofe_seismic.data import make_synthetic_seismic_cube
from nofe_seismic.hard_evals import run_hard_eval_suite


def test_run_hard_eval_suite_smoke(tmp_path: Path) -> None:
    volume = make_synthetic_seismic_cube(shape=(24, 24, 32), seed=3)
    out_dir = tmp_path / "hard_evals"

    summary = run_hard_eval_suite(
        volume,
        out_dir=out_dir,
        device="cpu",
        seed=3,
        base_stride_2d=(1, 1),
        radius_2d=0.45,
        epochs_2d=3,
        epochs_3d=3,
        run_3d_cube=True,
    )

    assert (out_dir / "hard_eval_summary.json").exists()
    assert (out_dir / "noise_stress.csv").exists()
    assert (out_dir / "density_curve.csv").exists()

    fig_dir = out_dir / "figures"
    expected_figs = (
        "figA_nofe_radar_overview.png",
        "figB_4tile_mosaic_seams.png",
        "figC_noise_and_density_stress.png",
        "figD_3d_cube_and_manifold.png",
        "figE_salt_dome_native_zoom.png",
    )
    for name in expected_figs:
        path = fig_dir / name
        assert path.exists(), f"Missing figure {path}"
        assert path.stat().st_size > 1024

    noise = summary["noise_stress_test"]
    density = summary["density_transfer_curve"]
    mosaic = summary["mosaic_4tile_seam_norm_rmse"]
    auc = summary["salt_zoom_boundary_auc"]
    assert isinstance(noise, list)
    assert isinstance(density, list)
    assert isinstance(mosaic, dict)
    assert isinstance(auc, dict)
    assert len(noise) > 0
    assert len(density) > 0
    assert "NOFE-8D" in mosaic
    assert math.isfinite(mosaic["NOFE-8D"])
    for val in auc.values():
        assert math.isfinite(val)
