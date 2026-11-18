"""CLI entrypoint to run the full NOFE seismic data reduction experiment suite."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nofe_seismic.data import (
    LazySeismicVolume,
    SeismicVolume,
    load_f3_directory,
    make_synthetic_seismic_cube,
)
from nofe_seismic.experiments import ExperimentConfig, run_all_experiments

_DENSE_ABLATION_MAX_INLINES = 300
_TRIPLE_LEN = 3


def _parse_triple(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != _TRIPLE_LEN:
        raise ValueError(f"Expected 3 integers, got {len(values)}")
    return int(values[0]), int(values[1]), int(values[2])


def _auto_stride(
    shape: tuple[int, int, int],
    *,
    target_train_nodes: int = 85_000,
    train_frac: float = 0.75,
) -> tuple[int, int, int]:
    ni, nx, nz = shape
    ni_train = max(8, round(ni * train_frac))
    si = sx = sz = 1
    while ((ni_train + si - 1) // si) * ((nx + sx - 1) // sx) * (
        (nz + sz - 1) // sz
    ) > target_train_nodes:
        lens = (ni_train / si, nx / sx, nz / sz)
        idx = max(range(3), key=lambda k: lens[k])
        if idx == 0:
            si += 1
        elif idx == 1:
            sx += 1
        else:
            sz += 1
    return si, sx, sz


def _load_volume_and_config(
    args: argparse.Namespace,
) -> tuple[SeismicVolume | LazySeismicVolume, ExperimentConfig]:
    if args.dataset == "synthetic":
        volume: SeismicVolume | LazySeismicVolume = make_synthetic_seismic_cube(
            _parse_triple(args.shape), seed=args.seed
        )
        default_out = Path("outputs/synthetic")
        default_radius = 0.25
        default_nystrom = 0.25
        default_epochs = 80
        default_slice_stride = (1, 1, 1)
        default_slab = 2
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
        default_radius = 0.15
        default_nystrom = 0.15
        default_epochs = 60
        default_slice_stride = (4, 4, 2)
        default_slab = 0
        include_dense = (
            False
            if args.skip_dense_ablation
            else volume.shape[0] < _DENSE_ABLATION_MAX_INLINES
        )

    stride = (
        _parse_triple(args.stride)
        if args.stride is not None
        else _auto_stride(volume.shape)
    )
    slice_stride = (
        _parse_triple(args.slice_stride)
        if args.slice_stride is not None
        else default_slice_stride
    )
    nystrom_val = args.nystrom if args.nystrom is not None else default_nystrom
    cfg = ExperimentConfig(
        out_dir=args.out_dir if args.out_dir is not None else default_out,
        device=args.device,
        seed=args.seed,
        train_stride=stride,
        slice_stride=slice_stride,
        slab_half_width=(
            args.slab_half_width if args.slab_half_width is not None else default_slab
        ),
        radius=args.radius if args.radius is not None else default_radius,
        nystrom=nystrom_val if nystrom_val > 0 else None,
        epochs=args.epochs if args.epochs is not None else default_epochs,
        include_dense_ablation=include_dense,
    )
    return volume, cfg


def main() -> None:
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
    args = parser.parse_args()

    volume, cfg = _load_volume_and_config(args)
    summary = run_all_experiments(volume, cfg)
    print(f"Saved outputs to {cfg.out_dir}")
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
