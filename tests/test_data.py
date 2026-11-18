"""Tests for seismic data loading, 11-channel attributes and synthetic 3D cube."""

from pathlib import Path

import numpy as np

from nofe_seismic.data import (
    ATTRIBUTE_NAMES,
    LazySeismicVolume,
    SeismicVolume,
    extract_seismic_attributes,
    load_f3_directory,
    make_synthetic_seismic_cube,
    volume_to_points,
)


def test_synthetic_cube_and_attributes() -> None:
    vol = make_synthetic_seismic_cube(shape=(24, 24, 32), seed=0)

    assert vol.amplitude.shape == (24, 24, 32)
    assert vol.features.shape == (24, 24, 32, len(ATTRIBUTE_NAMES))
    assert vol.labels is not None
    assert vol.labels.shape == (24, 24, 32)
    assert set(np.unique(vol.labels).tolist()).issubset({0, 1, 2, 3, 4})
    assert np.isfinite(vol.features).all()

    pts = volume_to_points(vol, stride=(2, 2, 2))

    assert pts.features.shape == (12 * 12 * 16, len(ATTRIBUTE_NAMES))
    assert pts.locations.shape == (12 * 12 * 16, 3)
    assert pts.amplitude.shape == (12 * 12 * 16,)
    assert pts.labels is not None
    assert pts.labels.shape == (12 * 12 * 16,)


def test_load_f3_directory_alaudah_and_npz(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    amp = np.random.default_rng(1).normal(size=(10, 12, 16)).astype(np.float32)
    lbl = np.zeros((10, 12, 16), dtype=np.int64)
    np.save(train_dir / "train_seismic.npy", amp)
    np.save(train_dir / "train_labels.npy", lbl)

    vols = load_f3_directory(tmp_path)

    assert "train" in vols
    assert vols["train"].amplitude.shape == (10, 12, 16)
    pts_train = volume_to_points(vols["train"])
    assert pts_train.features.shape == (10 * 12 * 16, 11)
    assert vols["train"].labels is not None

    npz_dir = tmp_path / "npz_case"
    npz_dir.mkdir()
    feats = np.random.default_rng(2).normal(size=(8, 8, 10, 5)).astype(np.float32)
    np.savez(npz_dir / "cube.npz", amplitude=amp[:8, :8, :10], features=feats)

    vols_npz = load_f3_directory(npz_dir)

    assert "cube" in vols_npz
    assert isinstance(vols_npz["cube"], SeismicVolume)
    assert vols_npz["cube"].features.shape == (8, 8, 10, 5)

    extracted, names = extract_seismic_attributes(amp)
    assert extracted.shape == (10, 12, 16, 11)
    assert len(names) == 11


def test_load_f3_full_layout_and_zero_plane_trim(tmp_path: Path) -> None:
    full_dir = tmp_path / "data" / "processed" / "f3" / "full"
    full_dir.mkdir(parents=True)
    amp = np.random.default_rng(3).normal(size=(12, 14, 10)).astype(np.float32)
    amp[:, :, 0] = 0.0
    amp[:, :, -1] = 0.0
    lbl = np.ones((12, 14, 10), dtype=np.uint8)
    np.save(full_dir / "seismic.npy", amp)
    np.save(full_dir / "labels.npy", lbl)

    vols = load_f3_directory(tmp_path)

    assert "full" in vols
    vol = vols["full"]
    assert isinstance(vol, LazySeismicVolume)
    assert vol.shape == (12, 14, 8)

    pts = volume_to_points(vol, inline_range=(2, 8), stride=(2, 2, 2))
    assert pts.features.shape == (3 * 7 * 4, 11)
    assert pts.labels is not None
    assert pts.labels.shape == (3 * 7 * 4,)
    assert np.isfinite(pts.features).all()

    pts_chunked = vol.extract_points(
        inline_range=(2, 8), stride=(2, 2, 2), inline_chunk=2
    )
    assert np.allclose(pts.features, pts_chunked.features, atol=1e-6)
    assert np.array_equal(pts.locations, pts_chunked.locations)
