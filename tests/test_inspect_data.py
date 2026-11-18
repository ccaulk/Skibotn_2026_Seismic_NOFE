"""Tests for scripts/inspect_data.py archive extraction and schema inspection."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


def test_inspect_data_extracts_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample.zip"
    extract_dir = tmp_path / "extracted"

    arr_path = tmp_path / "vol.npy"
    np.save(arr_path, np.zeros((4, 5, 6), dtype=np.float32))
    py_path = tmp_path / "loader.py"
    py_path.write_text("def load_cube():\n    return None\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(arr_path, arcname="vol.npy")
        zf.write(py_path, arcname="loader.py")

    proc = subprocess.run(  # noqa: S603 -- fixed in-repo script and tmp paths
        [
            sys.executable,
            "scripts/inspect_data.py",
            "--zip-path",
            str(zip_path),
            "--extract-dir",
            str(extract_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "vol.npy" in proc.stdout
    assert "shape=(4, 5, 6)" in proc.stdout
    assert "function load_cube" in proc.stdout
    assert (extract_dir / "vol.npy").exists()
