"""Run latent-dim ablations, facies separation optimization & figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from nofe_seismic.data.synthetic import make_synthetic_seismic_cube
from nofe_seismic.data.volumes import load_f3_directory
from nofe_seismic.facies.suite import run_ablation_and_facies_suite
from nofe_seismic.types import Stride2D, VolumeShape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("f3", "synthetic"), default="f3")
    parser.add_argument("--data-dir", type=Path, default=Path("f3-full"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/f3_ablations_and_facies"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stride-2d", type=int, nargs=2, default=(3, 3), metavar=("DI", "DX")
    )
    parser.add_argument("--radius-2d", type=float, default=0.05)
    parser.add_argument("--dims-2d", type=int, nargs="+", default=[2, 3, 4, 8, 16, 32])
    parser.add_argument("--dims-3d", type=int, nargs="+", default=[3, 8, 16, 32])
    parser.add_argument("--epochs-2d", type=int, default=60)
    parser.add_argument("--epochs-3d", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "f3":
        volumes = load_f3_directory(args.data_dir)
        volume = volumes["full"] if "full" in volumes else next(iter(volumes.values()))
    else:
        volume = make_synthetic_seismic_cube(
            shape=VolumeShape(36, 36, 48), seed=args.seed
        )

    summary = run_ablation_and_facies_suite(
        volume,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        stride_2d=Stride2D(args.stride_2d[0], args.stride_2d[1]),
        radius_2d=args.radius_2d,
        dims_2d=tuple(args.dims_2d),
        dims_3d=tuple(args.dims_3d),
        epochs_2d=args.epochs_2d,
        epochs_3d=args.epochs_3d,
    )
    summary_path = args.out_dir / "ablation_and_facies_summary.json"
    print(f"Saved ablation & facies summary to {summary_path}")
    breakthrough = summary["facies_breakthrough"]
    if isinstance(breakthrough, dict):
        for key, val in breakthrough.items():
            if isinstance(val, dict):
                print(
                    f"  {key}: bal_acc={val['lr_bal_acc']}, "
                    f"macro_f1={val['lr_macro_f1']}, "
                    f"knn={val['knn_acc']}, sil={val['silhouette']}"
                )


if __name__ == "__main__":
    main()
