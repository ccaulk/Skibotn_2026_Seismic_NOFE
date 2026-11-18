"""Shared value types and named structures used across ``nofe_seismic``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import numpy as np
from nofe import NOFE

RowDict: TypeAlias = dict[str, float | int | str | None]


class IndexRange(NamedTuple):
    """Half-open integer index interval ``[start, stop)``."""

    start: int
    stop: int


class Stride2D(NamedTuple):
    """2D subsampling stride ``(inline, crossline)``."""

    di: int
    dx: int


class Stride3D(NamedTuple):
    """3D subsampling stride ``(inline, crossline, depth)``."""

    di: int
    dx: int
    dz: int


class GridShape2D(NamedTuple):
    """2D grid dimensions ``(n_u, n_v)``."""

    nu: int
    nv: int


class VolumeShape(NamedTuple):
    """3D seismic volume dimensions ``(n_inline, n_crossline, n_depth)``."""

    ni: int
    nx: int
    nz: int


class Extent2D(NamedTuple):
    """Matplotlib ``imshow`` extent ``(xmin, xmax, ymax, ymin)``."""

    xmin: float
    xmax: float
    ymax: float
    ymin: float


class PlaneSurface(NamedTuple):
    """3D point coordinates and RGBA face colours for one cutaway plane."""

    locations: np.ndarray
    rgba: np.ndarray


@dataclass(frozen=True)
class NofeHyperparams:
    """Model architecture and training schedule for one NOFE run."""

    out_dim: int = 8
    width: int = 16
    ker_width: int = 48
    depth: int = 2
    nystrom: float | None = 0.10
    chunk_size: int = 8192
    device: str = "cpu"
    seed: int = 42
    epochs: int = 80
    patience: int = 20
    lr: float = 1e-2


@dataclass(frozen=True)
class TrainedNofe:
    """Trained NOFE operator along with its training history and timing."""

    model: NOFE
    history: list[dict[str, float]]
    train_embedding: np.ndarray
    elapsed_sec: float


def extent_from_locations(locs: np.ndarray) -> Extent2D:
    """Compute ``Extent2D`` from 2D ``(inline, crossline)`` point coordinates."""
    return Extent2D(
        xmin=float(locs[:, 1].min()),
        xmax=float(locs[:, 1].max()),
        ymax=float(locs[:, 0].max()),
        ymin=float(locs[:, 0].min()),
    )
