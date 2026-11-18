"""Extract and inspect f3-full-ready.zip or files under data/."""

from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path

import numpy as np


def _summarize_py(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    defs = [
        f"{type(node).__name__.replace('Def', '').lower()} {node.name}"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if defs:
        print(f"    py defs: {', '.join(defs[:15])}")


def inspect_directory(data_dir: Path) -> None:
    print(f"Inspecting directory: {data_dir.resolve()}")
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir():
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.relative_to(data_dir)}  ({size_mb:.2f} MiB)")
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            print(
                f"    npy shape={arr.shape}, dtype={arr.dtype}, "
                f"min={float(np.nanmin(arr)):.4g}, max={float(np.nanmax(arr)):.4g}"
            )
        elif path.suffix == ".npz":
            with np.load(path) as zf:
                for key in zf.files:
                    arr = zf[key]
                    print(f"    npz[{key!r}] shape={arr.shape}, dtype={arr.dtype}")
        elif path.suffix == ".py":
            _summarize_py(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("data/f3-full-ready.zip"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/f3-full-ready"))
    args = parser.parse_args()

    if args.zip_path.exists() and not args.extract_dir.exists():
        print(f"Extracting {args.zip_path} -> {args.extract_dir}")
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            zf.extractall(args.extract_dir)

    target = args.extract_dir if args.extract_dir.exists() else args.zip_path.parent
    if not target.exists():
        raise FileNotFoundError(
            f"Neither {args.zip_path} nor {args.extract_dir} exists yet."
        )
    inspect_directory(target)


if __name__ == "__main__":
    main()
