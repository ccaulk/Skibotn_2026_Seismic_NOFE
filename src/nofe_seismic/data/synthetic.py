"""Synthetic 3D F3-like seismic cube generator with 5 geological facies."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .attributes import extract_seismic_attributes
from .volumes import SeismicVolume

_FACIES_HORIZONTAL = 0
_FACIES_CLINOFORM = 1
_FACIES_CHANNEL = 2
_FACIES_DEEP = 3
_FACIES_SALT = 4


def _ricker_wavelet(length: int = 21, f0: float = 0.14) -> np.ndarray:
    t = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    arg = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * arg) * np.exp(-arg)).astype(np.float32)


def make_synthetic_seismic_cube(
    shape: tuple[int, int, int] = (64, 64, 96),
    *,
    seed: int = 42,
    noise_std: float = 0.08,
) -> SeismicVolume:
    """Generate a synthetic 3D seismic cube with 5 geological facies regimes."""
    ni, nx, nz = shape
    rng = np.random.default_rng(seed)

    ii, xx, zz = np.meshgrid(
        np.arange(ni, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )

    fault_plane = 0.55 * ii + 0.45 * xx - 0.52 * (ni + nx) / 2.0
    fault_throw = np.where(fault_plane > 0.0, 6.0, 0.0).astype(np.float32)
    z_eff = zz - fault_throw

    h1 = 0.24 * nz + 0.08 * ii + 0.05 * xx + 1.8 * np.sin(ii / 10.0)
    h2 = 0.56 * nz + 0.14 * ii - 0.06 * xx + 2.2 * np.cos(xx / 11.0)

    dome_r2 = ((ii - 0.65 * ni) / (0.24 * ni)) ** 2 + (
        (xx - 0.35 * nx) / (0.24 * nx)
    ) ** 2
    in_dome = zz >= nz * (0.48 + 0.42 * np.minimum(dome_r2, 1.5))

    channel_center_i = 0.35 * ni + 4.0 * np.sin(xx / 7.5)
    channel_center_z = 0.40 * nz + 0.06 * xx + fault_throw
    in_channel = (
        ((ii - channel_center_i) / 4.5) ** 2 + ((z_eff - channel_center_z) / 4.2) ** 2
    ) <= 1.0

    labels = np.full(shape, _FACIES_DEEP, dtype=np.int64)
    labels[z_eff < h2] = _FACIES_CLINOFORM
    labels[z_eff < h1] = _FACIES_HORIZONTAL
    labels[in_channel] = _FACIES_CHANNEL
    labels[in_dome] = _FACIES_SALT

    dome_noise = gaussian_filter(
        rng.normal(size=shape).astype(np.float32), sigma=(0.8, 0.8, 1.1)
    )
    unit_fields = {
        _FACIES_HORIZONTAL: (
            1.0
            + 0.32 * np.sin(0.55 * (z_eff - 0.06 * ii))
            + 0.14 * np.cos(1.05 * z_eff)
        ),
        _FACIES_CLINOFORM: 1.45 + 0.48 * np.sin(0.75 * (z_eff + 0.22 * xx - 0.12 * ii)),
        _FACIES_CHANNEL: 2.65 + 0.65 * np.sin(0.95 * (z_eff - channel_center_z)),
        _FACIES_DEEP: 1.85 + 0.45 * np.sin(0.38 * (z_eff - 0.05 * xx)),
        _FACIES_SALT: 2.1 + 0.40 * dome_noise,
    }
    impedance = np.zeros(shape, dtype=np.float32)
    for facies_id, field in unit_fields.items():
        impedance = np.where(labels == facies_id, field, impedance)

    reflectivity = np.diff(impedance, axis=2, prepend=impedance[:, :, :1])
    wavelet = _ricker_wavelet(length=21, f0=0.14)
    seismic = np.apply_along_axis(
        lambda trace: np.convolve(trace, wavelet, mode="same"),
        axis=2,
        arr=reflectivity,
    ).astype(np.float32)
    seismic += rng.normal(scale=noise_std, size=shape).astype(np.float32)

    stack = extract_seismic_attributes(seismic)
    return SeismicVolume(
        amplitude=seismic,
        features=stack.features,
        feature_names=stack.names,
        labels=labels,
        name="synthetic_f3_like",
    )
