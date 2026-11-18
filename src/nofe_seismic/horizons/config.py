"""Configuration for NOFE-based boundary reconstruction and horizon tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nofe_seismic.types import Stride3D

_DEFAULT_TRAIN_STRIDE = Stride3D(11, 20, 6)


@dataclass(frozen=True)
class HorizonConfig:
    """Configuration for NOFE-based boundary reconstruction and horizon tracking."""

    out_dir: Path = Path("outputs/f3_horizon_tracking")
    device: str = "cpu"
    seed: int = 42
    train_inline_frac: float = 0.75
    train_stride: tuple[int, int, int] = _DEFAULT_TRAIN_STRIDE
    slice_lateral_stride: int = 4
    radius_train: float = 0.15
    radius_slice: float = 0.12
    width: int = 16
    ker_width: int = 48
    depth: int = 2
    out_dim: int = 8
    nystrom: float | None = 0.10
    chunk_size: int = 8192
    epochs: int = 80
    patience: int = 20
    lr: float = 1e-2
    contrast_windows: tuple[int, ...] = (4, 8, 14)
    rgt_w_dip: float = 10.0
    rgt_w_mono: float = 1.0
    max_slope: float = 10.0
    noise_sigma: float = 0.50
