"""2D survey trace extraction and depth-log summary records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from ..types import GridShape2D, Stride2D
from .volumes import LazySeismicVolume, SeismicVolume

_MIN_FACIES_FOR_HIGH_START = 6


@dataclass(frozen=True)
class TraceLogs:
    """2D survey trace locations, depth-log features and optional facies summaries."""

    traces: np.ndarray
    locations: np.ndarray
    grid_shape: GridShape2D
    rms_amp: np.ndarray
    facies_fractions: np.ndarray | None
    dominant_facies: np.ndarray | None
    deepest_facies: np.ndarray | None
    salt_fraction: np.ndarray | None
    num_classes: int


def extract_trace_logs(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    inline_range: tuple[int, int] | None = None,
    crossline_range: tuple[int, int] | None = None,
    stride_2d: tuple[int, int] = Stride2D(1, 1),
    feature_mode: Literal["raw", "envelope"] = "raw",
) -> TraceLogs:
    """Extract 2D ``(inline, crossline)`` trace locations and depth logs ``[N, Z]``."""
    ni, nx, _ = volume.shape
    i0, i1 = inline_range if inline_range is not None else (0, ni)
    x0, x1 = crossline_range if crossline_range is not None else (0, nx)
    si, sx = stride_2d

    sl = (slice(i0, i1, si), slice(x0, x1, sx), slice(None))
    traces_3d = np.asarray(volume.amplitude[sl], dtype=np.float32)
    gi, gx, gz = traces_3d.shape
    traces_raw = traces_3d.reshape(gi * gx, gz)
    if feature_mode == "envelope":
        env = np.abs(hilbert(traces_raw.astype(np.float64), axis=1)).astype(np.float32)
        traces = gaussian_filter1d(env, sigma=2.0, axis=1).astype(np.float32)
    elif feature_mode == "raw":
        traces = traces_raw
    else:
        raise ValueError(f"Unsupported feature_mode {feature_mode!r}")

    ii, xx = np.meshgrid(
        np.arange(i0, i1, si, dtype=np.float32),
        np.arange(x0, x1, sx, dtype=np.float32),
        indexing="ij",
    )
    locations = np.stack([ii.ravel(), xx.ravel()], axis=1).astype(np.float32)
    rms_amp = np.sqrt(np.mean(traces_raw * traces_raw, axis=1)).astype(np.float32)
    grid_shape = GridShape2D(int(gi), int(gx))

    if volume.labels is None:
        return TraceLogs(
            traces=traces,
            locations=locations,
            grid_shape=grid_shape,
            rms_amp=rms_amp,
            facies_fractions=None,
            dominant_facies=None,
            deepest_facies=None,
            salt_fraction=None,
            num_classes=0,
        )

    lbl_2d = np.asarray(volume.labels[sl], dtype=np.int64).reshape(gi * gx, gz)
    num_classes = int(max(int(lbl_2d.max()) + 1, 1))
    fracs = np.stack(
        [(lbl_2d == c).mean(axis=1, dtype=np.float32) for c in range(num_classes)],
        axis=1,
    )
    high_start = (
        4 if num_classes >= _MIN_FACIES_FOR_HIGH_START else max(1, num_classes - 1)
    )
    return TraceLogs(
        traces=traces,
        locations=locations,
        grid_shape=grid_shape,
        rms_amp=rms_amp,
        facies_fractions=fracs,
        dominant_facies=np.argmax(fracs, axis=1).astype(np.int64),
        deepest_facies=lbl_2d[:, -1].astype(np.int64),
        salt_fraction=fracs[:, high_start:].sum(axis=1).astype(np.float32),
        num_classes=num_classes,
    )
