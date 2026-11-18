"""CLI entrypoint for 2D (inline, crossline) depth-log compression with NOFE."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nofe_seismic.data.synthetic import make_synthetic_seismic_cube
from nofe_seismic.data.volumes import (
    LazySeismicVolume,
    SeismicVolume,
    load_f3_directory,
)
from nofe_seismic.depth.compression import (
    DepthCompressionConfig,
    run_depth_compression,
)
from nofe_seismic.types import Stride2D, VolumeShape

_LARGE_VOLUME_INLINES = 400
_PAIR_LEN = 2
_DEFAULT_SYNTHETIC_SHAPE = VolumeShape(64, 64, 96)


@dataclass(frozen=True)
class _LoadedDepthRun:
    volume: SeismicVolume | LazySeismicVolume
    config: DepthCompressionConfig


def _parse_stride_2d(values: Sequence[int]) -> Stride2D:
    if len(values) != _PAIR_LEN:
        raise ValueError(f"Expected 2 integers, got {len(values)}")
    return Stride2D(int(values[0]), int(values[1]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("f3", "synthetic"), default="f3")
    parser.add_argument("--data-dir", type=Path, default=Path("f3-full"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--stride-2d", type=int, nargs=2, default=None)
    parser.add_argument(
        "--feature-mode", choices=("raw", "envelope"), default="envelope"
    )
    parser.add_argument("--feature-scale", choices=("column", "global"), default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--width-8d", type=int, default=24)
    parser.add_argument("--ker-width-8d", type=int, default=64)
    parser.add_argument("--depth-8d", type=int, default=2)
    parser.add_argument("--nystrom", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare_depth_run(args: argparse.Namespace) -> _LoadedDepthRun:
    if args.dataset == "synthetic":
        volume: SeismicVolume | LazySeismicVolume = make_synthetic_seismic_cube(
            _DEFAULT_SYNTHETIC_SHAPE, seed=args.seed
        )
        default_stride = Stride2D(1, 1)
        default_radius = 0.12
        default_out = Path("outputs/synthetic_depth_compression")
    else:
        volumes = load_f3_directory(args.data_dir)
        split_name = (
            ("full" if "full" in volumes else next(iter(sorted(volumes))))
            if args.split is None
            else str(args.split)
        )
        volume = volumes[split_name]
        default_stride = (
            Stride2D(3, 3)
            if volume.shape.ni >= _LARGE_VOLUME_INLINES
            else Stride2D(1, 1)
        )
        default_radius = 0.05 if default_stride == Stride2D(3, 3) else 0.035
        default_out = Path(f"outputs/f3_{split_name}_depth_compression_8d_tuned")

    feature_scale = (
        args.feature_scale
        if args.feature_scale is not None
        else ("global" if args.feature_mode == "envelope" else "column")
    )
    cfg = DepthCompressionConfig(
        out_dir=args.out_dir if args.out_dir is not None else default_out,
        device=args.device,
        seed=args.seed,
        stride_2d=(
            _parse_stride_2d(args.stride_2d)
            if args.stride_2d is not None
            else default_stride
        ),
        feature_mode=args.feature_mode,
        feature_scale=feature_scale,
        radius=args.radius if args.radius is not None else default_radius,
        width_8d=args.width_8d,
        ker_width_8d=args.ker_width_8d,
        depth_8d=args.depth_8d,
        nystrom=args.nystrom if args.nystrom > 0 else None,
        epochs=args.epochs,
    )
    return _LoadedDepthRun(volume=volume, config=cfg)


def main() -> None:
    loaded = _prepare_depth_run(_parse_args())
    summary = run_depth_compression(loaded.volume, loaded.config)
    print(f"Saved depth compression outputs to {loaded.config.out_dir}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
