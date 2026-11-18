"""Quantitative evaluation metrics for structure and spatial preservation."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

_EPS = 1e-12


class _ArrayPair(NamedTuple):
    x: np.ndarray
    z: np.ndarray


class _EdgeMoments(NamedTuple):
    sum_dx2: float
    sum_dz2: float
    sum_dxdz: float


def norm_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Return RMSE between ``pred`` and ``ref`` normalised by ``std(ref)``."""
    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
    scale = float(np.std(ref)) or 1.0
    return rmse / scale


def _to_numpy(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def _rank_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    """Compute 1-based neighbour ranks per row (self-distance excluded)."""
    n = dist_matrix.shape[0]
    d = dist_matrix.copy()
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]
    return ranks


def _subsample_pair(
    x: np.ndarray, z: np.ndarray, *, sample_size: int, seed: int
) -> _ArrayPair:
    if x.shape[0] <= sample_size:
        return _ArrayPair(x, z)
    idx = np.random.default_rng(seed).choice(
        x.shape[0], size=sample_size, replace=False
    )
    return _ArrayPair(x[idx], z[idx])


def _kruskal_stress(dx: np.ndarray, dz: np.ndarray) -> float:
    tri = np.triu_indices(dx.shape[0], k=1)
    vx, vz = dx[tri], dz[tri]
    denom_z = float(np.dot(vz, vz))
    alpha = float(np.dot(vx, vz) / denom_z) if denom_z > _EPS else 0.0
    denom_x = float(np.dot(vx, vx))
    if denom_x <= _EPS:
        return 0.0
    return float(np.sqrt(np.sum((vx - alpha * vz) ** 2) / denom_x))


def neighborhood_metrics(
    features: np.ndarray | torch.Tensor,
    embedding: np.ndarray | torch.Tensor,
    *,
    k: int = 15,
    sample_size: int = 4000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Trustworthiness, Continuity, k-NN recall and global Stress."""
    x_raw, z_raw = _to_numpy(features), _to_numpy(embedding)
    if x_raw.shape[0] != z_raw.shape[0]:
        raise ValueError(
            f"Row mismatch: features={x_raw.shape[0]}, embedding={z_raw.shape[0]}"
        )

    x, z = _subsample_pair(x_raw, z_raw, sample_size=sample_size, seed=seed)
    n = x.shape[0]
    if n <= k + 1:
        raise ValueError(f"Need n > k + 1, got n={n}, k={k}")

    dx = cdist(x, x, metric="euclidean")
    dz = cdist(z, z, metric="euclidean")
    rx, rz = _rank_matrix(dx), _rank_matrix(dz)
    nn_x, nn_z = rx <= k, rz <= k

    trust_pen = np.sum((rx - k) * (nn_z & (~nn_x)))
    cont_pen = np.sum((rz - k) * (nn_x & (~nn_z)))
    norm = n * k * (2 * n - 3 * k - 1)
    trustworthiness = float(1.0 - (2.0 / norm) * trust_pen) if norm > 0 else 1.0
    continuity = float(1.0 - (2.0 / norm) * cont_pen) if norm > 0 else 1.0

    return {
        "trustworthiness": trustworthiness,
        "continuity": continuity,
        "knn_recall": float(np.sum(nn_x & nn_z) / (n * k)),
        "global_stress": _kruskal_stress(dx, dz),
        "eval_nodes": float(n),
    }


def _accumulate_edge_moments(
    x: np.ndarray,
    z: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    chunk_size: int,
) -> _EdgeMoments:
    sum_dx2 = 0.0
    sum_dz2 = 0.0
    sum_dxdz = 0.0
    num_edges = src.shape[0]
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        s, d = src[start:stop], dst[start:stop]
        dx = np.linalg.norm(x[s] - x[d], axis=1)
        dz = np.linalg.norm(z[s] - z[d], axis=1)
        sum_dx2 += float(np.dot(dx, dx))
        sum_dz2 += float(np.dot(dz, dz))
        sum_dxdz += float(np.dot(dx, dz))
    return _EdgeMoments(sum_dx2=sum_dx2, sum_dz2=sum_dz2, sum_dxdz=sum_dxdz)


def local_graph_metrics(
    features: np.ndarray | torch.Tensor,
    embedding: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
    *,
    chunk_size: int = 1_000_000,
) -> dict[str, float]:
    """Evaluate local scale-aligned Stress and spatial smoothness along edges."""
    x, z = _to_numpy(features), _to_numpy(embedding)
    ei = (
        edge_index.detach().cpu().numpy()
        if isinstance(edge_index, torch.Tensor)
        else edge_index
    )
    src, dst = ei[0], ei[1]
    num_edges = src.shape[0]
    if num_edges == 0:
        return {
            "local_stress": 0.0,
            "edge_embed_rms": 0.0,
            "relative_edge_variation": 0.0,
        }

    m = _accumulate_edge_moments(x, z, src, dst, chunk_size=chunk_size)
    alpha = m.sum_dxdz / m.sum_dz2 if m.sum_dz2 > _EPS else 0.0
    if m.sum_dx2 > _EPS:
        resid2 = max(0.0, m.sum_dx2 - 2.0 * alpha * m.sum_dxdz + (alpha**2) * m.sum_dz2)
        local_stress = float(np.sqrt(resid2 / m.sum_dx2))
    else:
        local_stress = 0.0

    edge_rms = float(np.sqrt(m.sum_dz2 / num_edges))
    global_rms = float(np.sqrt(np.mean(np.sum((z - z.mean(axis=0)) ** 2, axis=1))))
    rel_variation = edge_rms / global_rms if global_rms > _EPS else 0.0
    return {
        "local_stress": local_stress,
        "edge_embed_rms": edge_rms,
        "relative_edge_variation": rel_variation,
    }


def evaluate_clustering(
    embedding: np.ndarray | torch.Tensor,
    *,
    k_values: tuple[int, ...] = (4, 6, 8, 10),
    labels: np.ndarray | None = None,
    silhouette_sample: int = 10_000,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    """Run K-Means sweep on embeddings and compute internal + label scores."""
    z = _to_numpy(embedding)
    labels_arr = np.asarray(labels).reshape(-1) if labels is not None else None
    valid_mask = labels_arr >= 0 if labels_arr is not None else None
    has_valid_labels = (
        labels_arr is not None and valid_mask is not None and bool(np.any(valid_mask))
    )

    results: list[dict[str, float | int]] = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        pred = km.fit_predict(z)
        sil = float(
            silhouette_score(
                z,
                pred,
                sample_size=min(silhouette_sample, z.shape[0]),
                random_state=seed,
            )
        )
        row: dict[str, float | int] = {"k": int(k), "silhouette": sil}
        if has_valid_labels and labels_arr is not None and valid_mask is not None:
            row["ari"] = float(
                adjusted_rand_score(labels_arr[valid_mask], pred[valid_mask])
            )
            row["nmi"] = float(
                normalized_mutual_info_score(labels_arr[valid_mask], pred[valid_mask])
            )
        results.append(row)
    return results
