"""Smoke test for 2D depth-log compression pipeline."""

from pathlib import Path

import numpy as np

from nofe_seismic.data.synthetic import make_synthetic_seismic_cube
from nofe_seismic.data.traces import extract_trace_logs
from nofe_seismic.depth.compression import (
    DepthCompressionConfig,
    run_depth_compression,
)
from nofe_seismic.viz.summary import plot_8d_projection_map


def test_extract_trace_logs_envelope_is_nonnegative() -> None:
    vol = make_synthetic_seismic_cube((12, 12, 24), seed=1)

    raw = extract_trace_logs(vol, stride_2d=(2, 2), feature_mode="raw")
    env = extract_trace_logs(vol, stride_2d=(2, 2), feature_mode="envelope")

    assert raw.traces.shape == (36, 24)
    assert env.traces.shape == (36, 24)
    assert (env.traces >= 0.0).all()
    assert not (raw.traces >= 0.0).all()


def test_depth_compression_smoke(tmp_path: Path) -> None:
    vol = make_synthetic_seismic_cube((20, 20, 32), seed=5)
    cfg = DepthCompressionConfig(
        out_dir=tmp_path / "depth_comp",
        device="cpu",
        seed=5,
        stride_2d=(1, 1),
        radius=0.35,
        width=8,
        ker_width=16,
        depth=2,
        width_8d=12,
        ker_width_8d=16,
        depth_8d=2,
        nystrom=0.5,
        chunk_size=128,
        epochs=3,
        patience=3,
    )

    summary = run_depth_compression(vol, cfg)

    full_graph = summary["full_map_graph"]
    assert isinstance(full_graph, dict)
    assert full_graph["num_nodes"] == 400
    assert (tmp_path / "depth_comp" / "depth_compression_summary.json").exists()
    assert (tmp_path / "depth_comp" / "depth_compression_metrics.csv").exists()
    assert (tmp_path / "depth_comp" / "figures" / "depth_compression_maps.png").exists()
    assert (
        tmp_path / "depth_comp" / "figures" / "depth_compression_diagnostics.png"
    ).exists()
    assert (
        tmp_path / "depth_comp" / "figures" / "depth_log_reconstructions.png"
    ).exists()
    assert (tmp_path / "depth_comp" / "figures" / "nofe_8d_visualization.png").exists()


def test_plot_8d_projection_map(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    emb_8d = rng.normal(size=(100, 8)).astype(np.float32)
    ii, xx = np.meshgrid(
        np.arange(10, dtype=np.float32),
        np.arange(10, dtype=np.float32),
        indexing="ij",
    )
    locs = np.stack([ii.ravel(), xx.ravel()], axis=1)
    out_png = tmp_path / "proj8d.png"

    pca3, var = plot_8d_projection_map(
        emb_8d, (10, 10), locs, out_png, split_inline=7.0
    )

    assert pca3.shape == (100, 3)
    assert var.shape == (3,)
    assert out_png.exists()
