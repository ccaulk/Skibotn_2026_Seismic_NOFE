"""Run the NOFE hard evaluation and visualisation suite on F3 or synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

from nofe_seismic.data import load_f3_directory, make_synthetic_seismic_cube
from nofe_seismic.hard_evals import run_hard_eval_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("f3", "synthetic"),
        default="f3",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("f3-full"),
        help="Path to F3 directory when --dataset=f3.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/f3_hard_evals"),
        help="Directory for hard evaluation JSON, CSVs, and figures.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run NOFE on (default: cpu).",
    )
    parser.add_argument(
        "--stride-2d",
        type=int,
        nargs=2,
        default=(3, 3),
        metavar=("DI", "DX"),
        help="Base (inline, crossline) stride for 2D trace maps.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.05,
        help="Base normalised spatial radius for 2D radius graphs.",
    )
    parser.add_argument(
        "--epochs-2d",
        type=int,
        default=100,
        help="Training epochs for 2D NOFE depth-compression models.",
    )
    parser.add_argument(
        "--epochs-3d",
        type=int,
        default=70,
        help="Training epochs for the 3D cutaway NOFE model.",
    )
    parser.add_argument(
        "--no-3d-cube",
        action="store_true",
        help="Skip the 3D cutaway cube and 3D latent manifold figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "f3":
        volumes = load_f3_directory(args.data_dir)
        volume = volumes["full"] if "full" in volumes else next(iter(volumes.values()))
    else:
        volume = make_synthetic_seismic_cube(shape=(48, 48, 64), seed=args.seed)

    summary = run_hard_eval_suite(
        volume,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        base_stride_2d=(args.stride_2d[0], args.stride_2d[1]),
        radius_2d=args.radius,
        epochs_2d=args.epochs_2d,
        epochs_3d=args.epochs_3d,
        run_3d_cube=not args.no_3d_cube,
    )
    print(f"Saved hard evaluation summary to {args.out_dir / 'hard_eval_summary.json'}")
    if "mosaic_4tile_seam_norm_rmse" in summary:
        print("4-tile seam NRMSE:", summary["mosaic_4tile_seam_norm_rmse"])
    if "salt_zoom_boundary_auc" in summary:
        print("Salt-zoom boundary AUC:", summary["salt_zoom_boundary_auc"])


if __name__ == "__main__":
    main()
