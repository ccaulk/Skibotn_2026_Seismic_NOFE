"""Tests for NOFE-based boundary reconstruction and horizon tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nofe_seismic.data.synthetic import make_synthetic_seismic_cube
from nofe_seismic.horizons.boundaries import (
    evaluate_horizon_curve,
    extract_boundary_ridges,
    extract_transition_depths,
    extract_true_interfaces,
    extract_unit_top_depths,
    multiscale_boundary_contrast,
)
from nofe_seismic.horizons.config import HorizonConfig
from nofe_seismic.horizons.dp import (
    coupled_monotonic_boundaries_2d,
    monotonic_dp_boundaries,
)
from nofe_seismic.horizons.pipeline import run_horizon_reconstruction
from nofe_seismic.horizons.rgt import (
    discover_horizons_from_rgt,
    estimate_structure_tensor_2d,
    extract_horizon_iso_rgt,
    extract_surface_iso_rgt_3d,
    rgt_aligned_smooth,
    solve_rgt_2d,
    solve_rgt_3d,
)


def test_boundary_extraction_and_contrast_peak() -> None:
    nu, nz, dim = 12, 40, 4
    labels = np.zeros((nu, nz), dtype=np.int64)
    labels[:, 20:] = 1
    ifaces = extract_true_interfaces(labels)
    assert ifaces.shape == (nu, nz - 1)
    assert bool(ifaces[:, 19].all())
    assert int(ifaces.sum()) == nu

    trans = extract_transition_depths(labels, [(0, 1), (1, 2)])
    assert np.allclose(trans[(0, 1)], 19.5)
    assert np.isnan(trans[(1, 2)]).all()

    tops = extract_unit_top_depths(labels, [1, 2])
    assert np.allclose(tops[1], 19.5)
    assert np.isnan(tops[2]).all()

    emb = np.zeros((nu, nz, dim), dtype=np.float32)
    emb[:, 20:, :] = 2.0
    contrast = multiscale_boundary_contrast(emb, windows=(3, 6), lateral_sigma=0.0)
    assert contrast.shape == (nu, nz - 1)
    assert int(np.argmax(contrast[nu // 2])) == 19

    ridges = extract_boundary_ridges(contrast, nms_size=7, top_k=1)
    assert bool(ridges[:, 19].all())


def test_rgt_recovers_dipping_horizon_2d_and_3d() -> None:
    nu, nz = 24, 48
    uu, zz = np.meshgrid(
        np.arange(nu, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    true_slope = 0.6
    phase = zz - true_slope * uu
    emb = np.stack(
        [np.sin(0.4 * phase), np.cos(0.4 * phase), 0.1 * phase], axis=2
    ).astype(np.float32)

    field = estimate_structure_tensor_2d(
        emb, grad_sigma=(0.8, 0.8), tensor_sigma=(2.0, 2.0)
    )
    rgt = solve_rgt_2d(field, w_dip=25.0, w_mono=1.0)
    pred = extract_horizon_iso_rgt(rgt, seed_u=0, seed_z=16)
    expected = 16.0 + true_slope * np.arange(nu, dtype=np.float32)
    ev = evaluate_horizon_curve(expected, pred)
    assert ev["mae"] < 0.8

    contrast = np.zeros((nu, nz - 1), dtype=np.float32)
    for u in range(nu):
        z_b = round(14 + true_slope * u)
        if 0 <= z_b < nz - 1:
            contrast[u, z_b] = 5.0
    disc = discover_horizons_from_rgt(
        rgt, contrast, field.coherency, num_horizons=1, min_sep=6
    )
    assert len(disc) == 1
    assert np.abs(disc[0].depths - (14.0 + true_slope * np.arange(nu))).mean() < 1.5

    ni, nx, nz3 = 10, 10, 30
    ii, xx, zz3 = np.meshgrid(
        np.arange(ni, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(nz3, dtype=np.float32),
        indexing="ij",
    )
    ph3 = zz3 - 0.4 * ii - 0.3 * xx
    emb3 = np.stack([np.sin(0.5 * ph3), np.cos(0.5 * ph3)], axis=3).astype(np.float32)
    rgt3 = solve_rgt_3d(
        emb3, grad_sigma=(0.6, 0.6, 0.6), tensor_sigma=(1.5, 1.5, 1.5), w_dip=25.0
    )
    surf = extract_surface_iso_rgt_3d(rgt3, 0, 0, 12)
    expected3 = 12.0 + 0.4 * ii[:, :, 0] + 0.3 * xx[:, :, 0]
    assert float(np.abs(surf - expected3).mean()) < 1.0


def test_monotonic_dp_handles_pinchout_and_rgt_smooth() -> None:
    nu, nz, ncls = 6, 30, 4
    classes = np.array([0, 1, 2, 3], dtype=np.int64)
    probs = np.full((nu, nz, ncls), 0.02, dtype=np.float32)
    probs[:3, :8, 0] = 0.94
    probs[:3, 8:16, 1] = 0.94
    probs[:3, 16:24, 2] = 0.94
    probs[:3, 24:, 3] = 0.94

    probs[3:, :10, 0] = 0.94
    probs[3:, 10:20, 1] = 0.94
    probs[3:, 20:, 3] = 0.94

    logp = np.log(probs)
    rgt = np.broadcast_to(np.arange(nz, dtype=np.float32)[None, :], (nu, nz)).copy()
    sm = rgt_aligned_smooth(logp, rgt, sigma_u=0.5, sigma_z=0.5)
    assert sm.shape == logp.shape

    _, seg = monotonic_dp_boundaries(logp, classes)
    assert np.all(seg[:3, :8] == 0)
    assert np.all(seg[:3, 8:16] == 1)
    assert np.all(seg[:3, 16:24] == 2)
    assert np.all(seg[:3, 24:] == 3)
    assert np.all(seg[3:, :10] == 0)
    assert np.all(seg[3:, 10:20] == 1)
    assert np.all(seg[3:, 20:] == 3)

    # Inject a brief single-trace shallow spike on trace 1; 2D Viterbi rejects it.
    probs_spiky = probs.copy()
    probs_spiky[1, 1:5, 2] = 0.99
    _, seg_2d = coupled_monotonic_boundaries_2d(
        np.log(probs_spiky), classes, rgt, lam_lat=1.5, max_step=12
    )
    assert np.all(seg_2d[1, :8] == 0)
    assert np.all(seg_2d[3:, 20:] == 3)


def test_run_horizon_reconstruction_smoke(tmp_path: Path) -> None:
    vol = make_synthetic_seismic_cube((18, 18, 28), seed=11)
    cfg = HorizonConfig(
        out_dir=tmp_path / "horizons",
        device="cpu",
        seed=11,
        train_stride=(2, 2, 2),
        slice_lateral_stride=2,
        radius_train=0.45,
        radius_slice=0.45,
        width=8,
        ker_width=16,
        depth=2,
        out_dim=8,
        nystrom=0.5,
        chunk_size=128,
        epochs=3,
        patience=3,
        contrast_windows=(2, 4),
    )
    summary = run_horizon_reconstruction(vol, cfg)
    assert (cfg.out_dir / "horizon_summary.json").exists()
    assert (cfg.out_dir / "unsupervised_boundaries.csv").exists()
    assert (cfg.out_dir / "rgt_horizon_tracking.csv").exists()
    assert (cfg.out_dir / "probe_horizon_reconstruction.csv").exists()
    assert (cfg.out_dir / "robustness_stress.csv").exists()
    for fig_name in (
        "fig1_unsupervised_boundary_reconstruction.png",
        "fig2_horizon_tracking_and_probe.png",
        "fig3_noise_and_seam_robustness.png",
        "fig4_3d_horizon_surfaces.png",
    ):
        p = cfg.out_dir / "figures" / fig_name
        assert p.exists()
        assert p.stat().st_size > 1024
    train_graph = summary["train_graph"]
    seam = summary["tile_seam_consistency"]
    assert isinstance(train_graph, dict)
    assert isinstance(seam, dict)
    assert train_graph["num_nodes"] > 0
    assert seam["shared_interior_points"] > 0
