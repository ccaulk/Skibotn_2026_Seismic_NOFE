"""Baseline dimensionality reduction methods for seismic comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

_BARNES_HUT_MAX_DIM = 3


@dataclass(frozen=True)
class TsneResult:
    """Output of a (possibly subsampled) t-SNE projection."""

    embedding: np.ndarray
    indices: np.ndarray
    elapsed_sec: float


def run_pca(
    features: np.ndarray, *, out_dim: int = 3, seed: int = 42
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    model = PCA(n_components=out_dim, random_state=seed)
    emb = model.fit_transform(features).astype(np.float32)
    return emb, time.perf_counter() - t0


def run_umap(
    features: np.ndarray,
    *,
    out_dim: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    model = UMAP(
        n_components=out_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    emb = model.fit_transform(features).astype(np.float32)
    return emb, time.perf_counter() - t0


def run_tsne(
    features: np.ndarray,
    *,
    out_dim: int = 3,
    perplexity: float = 30.0,
    max_nodes: int = 15_000,
    seed: int = 42,
) -> TsneResult:
    """Run t-SNE (on a deterministic subsample if ``N > max_nodes``)."""
    n = features.shape[0]
    if n > max_nodes:
        idx = np.sort(
            np.random.default_rng(seed).choice(n, size=max_nodes, replace=False)
        )
        x_in = features[idx]
    else:
        idx = np.arange(n)
        x_in = features
    t0 = time.perf_counter()
    eff_perp = min(perplexity, max(5.0, (x_in.shape[0] - 1) / 3.0))
    method = "barnes_hut" if out_dim <= _BARNES_HUT_MAX_DIM else "exact"
    model = TSNE(
        n_components=out_dim,
        perplexity=eff_perp,
        method=method,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    emb = model.fit_transform(x_in).astype(np.float32)
    return TsneResult(embedding=emb, indices=idx, elapsed_sec=time.perf_counter() - t0)
