"""CLI entrypoint to run the full NOFE seismic data reduction experiment suite."""

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
from nofe_seismic.experiments.evals import ExperimentConfig
from nofe_seismic.experiments.suite import run_all_experiments
from nofe_seismic.types import Stride3D, VolumeShape

_DENSE_ABLATION_MAX_INLINES = 300
_TRIPLE_LEN = 3
_DEFAULT_SYNTHETIC_STRIDE = Stride3D(1, 1, 1)
_DEFAULT_F3_STRIDE = Stride3D(4, 4, 2)


@dataclass(frozen=True)
class _LoadedExperiment:
    volume: SeismicVolume | LazySeismicVolume
    config: ExperimentConfig


def _parse_stride(values: Sequence[int]) -> Stride3D:
    if len(values) != _TRIPLE_LEN:
        raise ValueError(f"Expected 3 integers, got {len(values)}")
    return Stride3D(int(values[0]), int(values[1]), int(values[2]))


def _parse_shape(values: Sequence[int]) -> VolumeShape:
    if len(values) != _TRIPLE_LEN:
        raise ValueError(f"Expected 3 integers, got {len(values)}")
    return VolumeShape(int(values[0]), int(values[1]), int(values[2]))


def _auto_stride(
    shape: VolumeShape,
    *,
    target_train_nodes: int = 85_000,
    train_frac: float = 0.75,
) -> Stride3D:
    ni_train = max(8, round(shape.ni * train_frac))
    si = sx = sz = 1
    while ((ni_train + si - 1) // si) * ((shape.nx + sx - 1) // sx) * (
        (shape.nz + sz - 1) // sz
    ) > target_train_nodes:
        lens = (ni_train / si, shape.nx / sx, shape.nz / sz)
        idx = max(range(3), key=lambda k: lens[k])
        if idx == 0:
            si += 1
        elif idx == 1:
            sx += 1
        else:
            sz += 1
    return Stride3D(si, sx, sz)


def _load_volume_and_config(args: argparse.Namespace) -> _LoadedExperiment:
    if args.dataset == "synthetic":
        volume: SeismicVolume | LazySeismicVolume = make_synthetic_seismic_cube(
            _parse_shape(args.shape), seed=args.seed
        )
        default_out = Path("outputs/synthetic")
        default_radius, default_nystrom, default_epochs = 0.25, 0.25, 80
        default_slice_stride, default_slab = _DEFAULT_SYNTHETIC_STRIDE, 2
        include_dense = not args.skip_dense_ablation
    else:
        volumes = load_f3_directory(args.data_dir)
        split_name = (
            ("full" if "full" in volumes else next(iter(sorted(volumes))))
            if args.split is None
            else str(args.split)
        )
        if split_name not in volumes:
            raise KeyError(
                f"Split {split_name!r} not found; available splits: {sorted(volumes)}"
            )
        volume = volumes[split_name]
        default_out = Path(f"outputs/f3_{split_name}")
        default_radius, default_nystrom, default_epochs = 0.15, 0.15, 60
        default_slice_stride, default_slab = _DEFAULT_F3_STRIDE, 0
        include_dense = (
            False
            if args.skip_dense_ablation
            else volume.shape.ni < _DENSE_ABLATION_MAX_INLINES
        )

    nystrom_val = args.nystrom if args.nystrom is not None else default_nystrom
    cfg = ExperimentConfig(
        out_dir=args.out_dir if args.out_dir is not None else default_out,
        device=args.device,
        seed=args.seed,
        train_stride=(
            _parse_stride(args.stride)
            if args.stride is not None
            else _auto_stride(volume.shape)
        ),
        slice_stride=(
            _parse_stride(args.slice_stride)
            if args.slice_stride is not None
            else default_slice_stride
        ),
        slab_half_width=(
            args.slab_half_width if args.slab_half_width is not None else default_slab
        ),
        radius=args.radius if args.radius is not None else default_radius,
        nystrom=nystrom_val if nystrom_val > 0 else None,
        epochs=args.epochs if args.epochs is not None else default_epochs,
        include_dense_ablation=include_dense,
    )
    return _LoadedExperiment(volume=volume, config=cfg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "f3"), default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=Path("f3-full"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 64, 96))
    parser.add_argument("--stride", type=int, nargs=3, default=None)
    parser.add_argument("--slice-stride", type=int, nargs=3, default=None)
    parser.add_argument("--slab-half-width", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--nystrom", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-dense-ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    loaded = _load_volume_and_config(_parse_args())
    summary = run_all_experiments(loaded.volume, loaded.config)
    print(f"Saved outputs to {loaded.config.out_dir}")
    print(
        json.dumps(
            {
                "dataset": summary["dataset"],
                "volume_shape": summary["volume_shape"],
                "train_graph": summary["train_graph"],
                "heldout_3d_matched": summary["heldout_3d_matched"],
                "patch_stitching": summary["patch_stitching"],
                "resolution_consistency": summary["resolution_consistency"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
