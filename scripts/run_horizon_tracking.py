"""CLI for boundary reconstruction and horizon tracking from NOFE reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nofe_seismic.data import load_f3_directory, make_synthetic_seismic_cube
from nofe_seismic.horizon_pipeline import run_horizon_reconstruction
from nofe_seismic.horizons import HorizonConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("f3", "synthetic"), default="f3")
    parser.add_argument("--data-dir", type=Path, default=Path("f3-full"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--shape", type=int, nargs=3, default=(48, 48, 64))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "synthetic":
        volume = make_synthetic_seismic_cube(tuple(args.shape), seed=args.seed)
        cfg = HorizonConfig(
            out_dir=args.out_dir or Path("outputs/synthetic_horizon_tracking"),
            device=args.device,
            seed=args.seed,
            train_stride=(2, 2, 2),
            slice_lateral_stride=1,
            radius_train=0.30,
            radius_slice=0.30,
            width=12,
            ker_width=32,
            nystrom=0.20,
            chunk_size=1024,
            epochs=args.epochs,
            patience=15,
            contrast_windows=(3, 6, 10),
        )
    else:
        volumes = load_f3_directory(args.data_dir)
        split_name = (
            ("full" if "full" in volumes else next(iter(sorted(volumes))))
            if args.split is None
            else args.split
        )
        volume = volumes[split_name]
        cfg = HorizonConfig(
            out_dir=args.out_dir or Path("outputs/f3_horizon_tracking"),
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
        )

    summary = run_horizon_reconstruction(volume, cfg)
    print(f"Saved horizon tracking outputs to {cfg.out_dir}")
    print(
        json.dumps(
            {
                "dataset": summary["dataset"],
                "nofe_train_sec": summary["nofe_train_sec"],
                "tile_seam_consistency": summary["tile_seam_consistency"],
                "heldout_3d_slab": summary["heldout_3d_slab"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
