"""Seismic volume containers, point-cloud slicing and F3 directory loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import IndexRange, VolumeShape
from .attributes import ATTRIBUTE_NAMES, extract_seismic_attributes

_AMP_NDIM = 3
_FEAT_NDIM = 4
_MIN_AMP_SCALE = 1e-8
_MIN_TRIM_PLANES = 2
_ZERO_PLANE_STD = 1e-7


@dataclass(frozen=True)
class PointCloud:
    """Flattened 3D seismic point-cloud arrays extracted from a volume."""

    features: np.ndarray
    locations: np.ndarray
    amplitude: np.ndarray
    labels: np.ndarray | None


@dataclass(frozen=True)
class SeismicVolume:
    """In-memory 3D seismic cube with axes ``(inline, crossline, depth)``."""

    amplitude: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray | None = None
    name: str = "seismic"

    @property
    def shape(self) -> VolumeShape:
        ni, nx, nz = self.amplitude.shape
        return VolumeShape(int(ni), int(nx), int(nz))

    def __post_init__(self) -> None:
        if self.amplitude.ndim != _AMP_NDIM:
            raise ValueError(
                f"amplitude must be 3-D [I, X, Z], got {self.amplitude.shape}"
            )
        if self.features.ndim != _FEAT_NDIM or self.features.shape[:3] != self.shape:
            raise ValueError(
                f"features must be 4-D {(*self.shape, 'F')}, got {self.features.shape}"
            )
        if len(self.feature_names) != self.features.shape[3]:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) != "
                f"channels ({self.features.shape[3]})"
            )
        if self.labels is not None and self.labels.shape != self.shape:
            raise ValueError(
                f"labels shape {self.labels.shape} must match amplitude {self.shape}"
            )


def _grid_locations_3d(
    i_idx: np.ndarray,
    x_idx: np.ndarray,
    z_idx: np.ndarray,
) -> np.ndarray:
    ii, xx, zz = np.meshgrid(i_idx, x_idx, z_idx, indexing="ij")
    return np.stack([ii.ravel(), xx.ravel(), zz.ravel()], axis=1).astype(np.float32)


class LazySeismicVolume:
    """Memory-mapped 3D seismic volume computing attributes per subvolume."""

    def __init__(
        self,
        amplitude: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        z_offset: int = 0,
        z_stop: int | None = None,
        amp_scale: float | None = None,
        name: str = "f3_full",
        halo: int = 6,
    ) -> None:
        if amplitude.ndim != _AMP_NDIM:
            raise ValueError(f"amplitude must be 3-D [I, X, Z], got {amplitude.shape}")
        if labels is not None and labels.shape != amplitude.shape:
            raise ValueError(
                f"labels shape {labels.shape} must match amplitude {amplitude.shape}"
            )
        self._raw_amp = amplitude
        self._raw_labels = labels
        self.z_offset = int(z_offset)
        self.z_stop = amplitude.shape[2] if z_stop is None else int(z_stop)
        self.shape = VolumeShape(
            int(amplitude.shape[0]),
            int(amplitude.shape[1]),
            int(self.z_stop - self.z_offset),
        )
        self.feature_names = ATTRIBUTE_NAMES
        self.name = name
        self.halo = int(halo)
        if amp_scale is not None:
            self.amp_scale = float(amp_scale)
            return
        sample = np.asarray(
            amplitude[::4, ::4, self.z_offset : self.z_stop : 2],
            dtype=np.float32,
        )
        std = float(sample.std())
        self.amp_scale = std if std > _MIN_AMP_SCALE else 1.0

    @property
    def amplitude(self) -> np.ndarray:
        """Trimmed amplitude view ``[I, X, Z_valid]``."""
        return self._raw_amp[:, :, self.z_offset : self.z_stop]

    @property
    def labels(self) -> np.ndarray | None:
        if self._raw_labels is None:
            return None
        return self._raw_labels[:, :, self.z_offset : self.z_stop]

    def extract_points(
        self,
        *,
        inline_range: tuple[int, int] | None = None,
        crossline_range: tuple[int, int] | None = None,
        depth_range: tuple[int, int] | None = None,
        stride: tuple[int, int, int] = (1, 1, 1),
        inline_chunk: int = 48,
    ) -> PointCloud:
        """Compute native 11-channel attributes on haloed chunks and slice points."""
        ni, nx, nz = self.shape
        i0, i1 = inline_range if inline_range is not None else (0, ni)
        x0, x1 = crossline_range if crossline_range is not None else (0, nx)
        z0, z1 = depth_range if depth_range is not None else (0, nz)
        si, sx, sz = stride

        hx0, hx1 = max(0, x0 - self.halo), min(nx, x1 + self.halo)
        lx0, lx1 = x0 - hx0, x1 - hx0
        x_idx = np.arange(x0, x1, sx, dtype=np.float32)
        z_idx = np.arange(z0, z1, sz, dtype=np.float32)
        raw_labels = self.labels

        feat_parts: list[np.ndarray] = []
        amp_parts: list[np.ndarray] = []
        lbl_parts: list[np.ndarray] = []
        loc_parts: list[np.ndarray] = []

        step = max(si, (inline_chunk // si) * si)
        for ci0 in range(i0, i1, step):
            ci1 = min(i1, ci0 + step)
            hi0, hi1 = max(0, ci0 - self.halo), min(ni, ci1 + self.halo)
            slab_amp = np.asarray(
                self._raw_amp[hi0:hi1, hx0:hx1, self.z_offset : self.z_stop],
                dtype=np.float32,
            )
            slab_feats = extract_seismic_attributes(
                slab_amp, scale=self.amp_scale
            ).features

            li0, li1 = ci0 - hi0, ci1 - hi0
            sl_feat = (slice(li0, li1, si), slice(lx0, lx1, sx), slice(z0, z1, sz))
            sl_global = (slice(ci0, ci1, si), slice(x0, x1, sx), slice(z0, z1, sz))

            feat_parts.append(
                slab_feats[sl_feat]
                .reshape(-1, len(self.feature_names))
                .astype(np.float32)
            )
            amp_parts.append(
                np.asarray(self.amplitude[sl_global], dtype=np.float32).ravel()
            )
            if raw_labels is not None:
                lbl_parts.append(
                    np.asarray(raw_labels[sl_global], dtype=np.int64).ravel()
                )
            loc_parts.append(
                _grid_locations_3d(
                    np.arange(ci0, ci1, si, dtype=np.float32), x_idx, z_idx
                )
            )

        return PointCloud(
            features=(
                np.concatenate(feat_parts, axis=0)
                if feat_parts
                else np.empty((0, len(self.feature_names)), dtype=np.float32)
            ),
            locations=(
                np.concatenate(loc_parts, axis=0)
                if loc_parts
                else np.empty((0, _AMP_NDIM), dtype=np.float32)
            ),
            amplitude=(
                np.concatenate(amp_parts, axis=0)
                if amp_parts
                else np.empty(0, dtype=np.float32)
            ),
            labels=np.concatenate(lbl_parts, axis=0) if lbl_parts else None,
        )


def _detect_zero_plane_trim(amp: np.ndarray, metadata_path: Path | None) -> IndexRange:
    """Detect zero-padded first/last vertical planes (e.g. planes 0 and 254)."""
    nz = amp.shape[2]
    z_start, z_stop = 0, nz
    if metadata_path is not None and metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        zero_z = set(meta.get("amplitude", {}).get("all_zero_planes", [[], [], []])[2])
        zero_z.update(meta.get("suspect_vertical_planes", []))
        if 0 in zero_z and nz > _MIN_TRIM_PLANES:
            z_start = 1
        if (nz - 1) in zero_z and z_stop - z_start > _MIN_TRIM_PLANES:
            z_stop = nz - 1
        return IndexRange(z_start, z_stop)

    if float(np.std(amp[:, :, 0])) < _ZERO_PLANE_STD and nz > _MIN_TRIM_PLANES:
        z_start = 1
    if (
        float(np.std(amp[:, :, nz - 1])) < _ZERO_PLANE_STD
        and z_stop - z_start > _MIN_TRIM_PLANES
    ):
        z_stop = nz - 1
    return IndexRange(z_start, z_stop)


def _load_f3_full(data_dir: Path) -> LazySeismicVolume | None:
    full_dirs = (
        data_dir / "data" / "processed" / "f3" / "full",
        data_dir / "processed" / "f3" / "full",
        data_dir,
    )
    for fdir in full_dirs:
        seis_path = fdir / "seismic.npy"
        if not seis_path.exists():
            continue
        lbl_path = fdir / "labels.npy"
        meta_path = fdir / "metadata.json"
        amp = np.load(seis_path, mmap_mode="r", allow_pickle=False)
        labels = (
            np.load(lbl_path, mmap_mode="r", allow_pickle=False)
            if lbl_path.exists()
            else None
        )
        trim = _detect_zero_plane_trim(amp, meta_path if meta_path.exists() else None)
        return LazySeismicVolume(
            amp, labels, z_offset=trim.start, z_stop=trim.stop, name="f3_full"
        )
    return None


def _load_alaudah_splits(
    data_dir: Path,
) -> dict[str, SeismicVolume | LazySeismicVolume]:
    candidates = (
        (
            "train",
            data_dir / "train" / "train_seismic.npy",
            data_dir / "train" / "train_labels.npy",
        ),
        (
            "test1",
            data_dir / "test_once" / "test1_seismic.npy",
            data_dir / "test_once" / "test1_labels.npy",
        ),
        (
            "test2",
            data_dir / "test_once" / "test2_seismic.npy",
            data_dir / "test_once" / "test2_labels.npy",
        ),
        ("train", data_dir / "train_seismic.npy", data_dir / "train_labels.npy"),
        ("test1", data_dir / "test1_seismic.npy", data_dir / "test1_labels.npy"),
        ("test2", data_dir / "test2_seismic.npy", data_dir / "test2_labels.npy"),
    )
    volumes: dict[str, SeismicVolume | LazySeismicVolume] = {}
    for split_name, seis_path, lbl_path in candidates:
        if split_name in volumes or not seis_path.exists():
            continue
        amp = np.load(seis_path, mmap_mode="r", allow_pickle=False)
        labels = (
            np.load(lbl_path, mmap_mode="r", allow_pickle=False)
            if lbl_path.exists()
            else None
        )
        volumes[split_name] = LazySeismicVolume(amp, labels, name=f"f3_{split_name}")
    return volumes


def _load_npz_volumes(
    data_dir: Path,
) -> dict[str, SeismicVolume | LazySeismicVolume]:
    volumes: dict[str, SeismicVolume | LazySeismicVolume] = {}
    for npz_path in sorted(data_dir.rglob("*.npz")):
        with np.load(npz_path) as zf:
            keys = set(zf.files)
            amp_key = next(
                (k for k in ("amplitude", "seismic", "data", "volume") if k in keys),
                None,
            )
            if amp_key is None or zf[amp_key].ndim != _AMP_NDIM:
                continue
            feat_key = next(
                (k for k in ("features", "attributes", "x") if k in keys),
                None,
            )
            lbl_key = next((k for k in ("labels", "facies", "y") if k in keys), None)
            amp = zf[amp_key].astype(np.float32)
            labels = zf[lbl_key].astype(np.int64) if lbl_key else None
            if feat_key is not None and zf[feat_key].ndim == _FEAT_NDIM:
                feats = zf[feat_key].astype(np.float32)
                names = tuple(f"feat_{i}" for i in range(feats.shape[3]))
            else:
                stack = extract_seismic_attributes(amp)
                feats, names = stack.features, stack.names
            volumes[npz_path.stem] = SeismicVolume(
                amplitude=amp,
                features=feats,
                feature_names=names,
                labels=labels,
                name=npz_path.stem,
            )
    return volumes


def load_f3_directory(
    data_dir: Path,
) -> dict[str, SeismicVolume | LazySeismicVolume]:
    """Load an extracted F3 dataset directory (f3-full, Alaudah splits or .npz)."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    full_vol = _load_f3_full(data_dir)
    if full_vol is not None:
        return {"full": full_vol}

    alaudah = _load_alaudah_splits(data_dir)
    if alaudah:
        return alaudah

    npz_vols = _load_npz_volumes(data_dir)
    if npz_vols:
        return npz_vols

    raise FileNotFoundError(
        f"No supported seismic .npy/.npz volumes found under {data_dir}. "
        "Run scripts/inspect_data.py to inspect the archive contents."
    )


def volume_to_points(
    volume: SeismicVolume | LazySeismicVolume,
    *,
    inline_range: tuple[int, int] | None = None,
    crossline_range: tuple[int, int] | None = None,
    depth_range: tuple[int, int] | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> PointCloud:
    """Slice and flatten a seismic volume into a :class:`PointCloud`."""
    if isinstance(volume, LazySeismicVolume):
        return volume.extract_points(
            inline_range=inline_range,
            crossline_range=crossline_range,
            depth_range=depth_range,
            stride=stride,
        )

    ni, nx, nz = volume.amplitude.shape
    i0, i1 = inline_range if inline_range is not None else (0, ni)
    x0, x1 = crossline_range if crossline_range is not None else (0, nx)
    z0, z1 = depth_range if depth_range is not None else (0, nz)
    si, sx, sz = stride

    locations = _grid_locations_3d(
        np.arange(i0, i1, si, dtype=np.float32),
        np.arange(x0, x1, sx, dtype=np.float32),
        np.arange(z0, z1, sz, dtype=np.float32),
    )
    sl = (slice(i0, i1, si), slice(x0, x1, sx), slice(z0, z1, sz))
    features = (
        volume.features[sl].reshape(-1, volume.features.shape[3]).astype(np.float32)
    )
    amplitude = volume.amplitude[sl].ravel().astype(np.float32)
    labels = (
        volume.labels[sl].ravel().astype(np.int64)
        if volume.labels is not None
        else None
    )
    return PointCloud(
        features=features,
        locations=locations,
        amplitude=amplitude,
        labels=labels,
    )
