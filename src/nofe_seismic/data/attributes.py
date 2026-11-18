"""11-channel 3D seismic attribute extraction from raw amplitude cubes."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from scipy.signal import hilbert

_AMP_NDIM = 3
_MIN_AMP_SCALE = 1e-8
_SOBEL_NORM = 8.0

ATTRIBUTE_NAMES: tuple[str, ...] = (
    "amplitude",
    "envelope",
    "cos_phase",
    "sin_phase",
    "inst_freq",
    "grad_z",
    "grad_lateral",
    "rms_local",
    "variance_local",
    "lateral_coherence",
    "rms_macro",
)


class AttributeStack(NamedTuple):
    """Extracted 4D seismic attribute tensor ``[I, X, Z, F]`` and channel names."""

    features: np.ndarray
    names: tuple[str, ...]


def extract_seismic_attributes(
    amplitude: np.ndarray, *, scale: float | None = None
) -> AttributeStack:
    """Compute an 11-channel 3D seismic attribute stack from ``[I, X, Z]`` cubes."""
    amp = np.asarray(amplitude, dtype=np.float32)
    if amp.ndim != _AMP_NDIM:
        raise ValueError(f"Expected 3-D volume [I, X, Z], got {amp.shape}")

    if scale is None:
        amp_std = float(amp.std())
        scale = amp_std if amp_std > _MIN_AMP_SCALE else 1.0
    amp_norm = (amp / scale).astype(np.float32, copy=False)

    analytic = hilbert(amp_norm.astype(np.float64), axis=2)
    envelope = np.abs(analytic).astype(np.float32)
    phase = np.angle(analytic)

    dphase = np.clip(
        np.gradient(np.unwrap(phase, axis=2), axis=2).astype(np.float32),
        -np.pi,
        np.pi,
    )
    inst_freq = gaussian_filter(dphase, sigma=(0.5, 0.5, 1.0)).astype(np.float32)

    grad_z = np.gradient(amp_norm, axis=2).astype(np.float32)
    grad_i = sobel(amp_norm, axis=0, mode="nearest") / _SOBEL_NORM
    grad_x = sobel(amp_norm, axis=1, mode="nearest") / _SOBEL_NORM
    grad_lateral = np.hypot(grad_i, grad_x).astype(np.float32)

    sq = amp_norm * amp_norm
    mean_local = uniform_filter(amp_norm, size=(3, 3, 5), mode="nearest")
    mean_sq_local = uniform_filter(sq, size=(3, 3, 5), mode="nearest")
    rms_local = np.sqrt(np.maximum(mean_sq_local, 0.0)).astype(np.float32)
    variance_local = np.maximum(mean_sq_local - mean_local * mean_local, 0.0).astype(
        np.float32
    )

    lat_smooth = uniform_filter(amp_norm, size=(3, 3, 1), mode="nearest")
    lat_diff2 = uniform_filter(
        (amp_norm - lat_smooth) ** 2, size=(3, 3, 5), mode="nearest"
    )
    lateral_coherence = (1.0 / (1.0 + lat_diff2 / (mean_sq_local + 1e-4))).astype(
        np.float32
    )
    rms_macro = np.sqrt(
        np.maximum(uniform_filter(sq, size=(7, 7, 9), mode="nearest"), 0.0)
    ).astype(np.float32)

    stack = np.stack(
        [
            amp_norm,
            envelope,
            np.cos(phase).astype(np.float32),
            np.sin(phase).astype(np.float32),
            inst_freq,
            grad_z,
            grad_lateral,
            rms_local,
            variance_local,
            lateral_coherence,
            rms_macro,
        ],
        axis=-1,
    ).astype(np.float32)
    clean = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    return AttributeStack(features=clean, names=ATTRIBUTE_NAMES)
