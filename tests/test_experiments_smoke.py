"""End-to-end smoke test of the full seismic experiment pipeline on a small cube."""

from pathlib import Path

from nofe_seismic.data import make_synthetic_seismic_cube
from nofe_seismic.experiments import ExperimentConfig, run_all_experiments


def test_experiment_pipeline_smoke(tmp_path: Path) -> None:
    vol = make_synthetic_seismic_cube((16, 16, 24), seed=7)
    cfg = ExperimentConfig(
        out_dir=tmp_path / "exp_smoke",
        device="cpu",
        seed=7,
        train_stride=(2, 2, 2),
        radius=0.55,
        width=8,
        ker_width=16,
        depth=2,
        nystrom=0.5,
        chunk_size=256,
        epochs=3,
        patience=3,
    )

    summary = run_all_experiments(vol, cfg)

    train_graph = summary["train_graph"]
    assert isinstance(train_graph, dict)
    assert train_graph["num_nodes"] > 0
    assert (tmp_path / "exp_smoke" / "metrics_summary.json").exists()
    assert (tmp_path / "exp_smoke" / "benchmark.csv").exists()
    assert (tmp_path / "exp_smoke" / "ablations.csv").exists()
    assert (tmp_path / "exp_smoke" / "figures" / "slice_inline_heldout.png").exists()
    assert (tmp_path / "exp_smoke" / "figures" / "ablation_summary.png").exists()
