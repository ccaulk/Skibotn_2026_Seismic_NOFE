"""Seismic graph construction, standardisation and typed NOFE boundary helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import torch
from nofe import NOFE, PointGraph
from nofe.search import knn_edges, radius_edges
from torch_geometric.data import Data

from .types import NofeHyperparams, TrainedNofe


class ScaledGraph(NamedTuple):
    """Built PyG graph paired with its scaled feature and position arrays."""

    graph: Data
    features: np.ndarray
    locations: np.ndarray


class EdgeParameters(NamedTuple):
    """Edge search parameters returned by ``PointGraph.edge_parameters``."""

    radius: float
    k: int


class NeighbourPairs(NamedTuple):
    """Directed ``(source, target)`` index arrays from neighbour search."""

    src: np.ndarray
    dst: np.ndarray


@dataclass(frozen=True)
class SharedStandardizer:
    """Per-column z-score scaler fitted once and reused across train/eval graphs."""

    feat_mean: np.ndarray
    feat_std: np.ndarray
    pos_mean: np.ndarray
    pos_std: np.ndarray
    pos_weights: np.ndarray | None = None
    eps: float = 1e-8

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        locations: np.ndarray,
        *,
        feature_scale: Literal["column", "global"] = "column",
        pos_weights: tuple[float, ...] | np.ndarray | None = None,
        eps: float = 1e-8,
    ) -> SharedStandardizer:
        feat = np.asarray(features, dtype=np.float32)
        loc = np.asarray(locations, dtype=np.float32)
        weights = (
            None if pos_weights is None else np.asarray(pos_weights, dtype=np.float32)
        )
        if weights is not None and weights.shape != (loc.shape[1],):
            raise ValueError(
                f"pos_weights shape {weights.shape} must match "
                f"location dims ({loc.shape[1]},)"
            )
        if feature_scale == "global":
            feat_mean = np.full(feat.shape[1], float(feat.mean()), dtype=np.float32)
            feat_std = np.full(feat.shape[1], float(feat.std()) + eps, dtype=np.float32)
        elif feature_scale == "column":
            feat_mean = feat.mean(axis=0)
            feat_std = feat.std(axis=0) + eps
        else:
            raise ValueError(
                f"feature_scale must be 'column' or 'global', got {feature_scale!r}"
            )
        return cls(
            feat_mean=feat_mean,
            feat_std=feat_std,
            pos_mean=loc.mean(axis=0),
            pos_std=loc.std(axis=0) + eps,
            pos_weights=weights,
            eps=eps,
        )

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        feat = np.asarray(features, dtype=np.float32)
        return ((feat - self.feat_mean) / self.feat_std).astype(np.float32, copy=False)

    def transform_locations(self, locations: np.ndarray) -> np.ndarray:
        loc = np.asarray(locations, dtype=np.float32)
        scaled = (loc - self.pos_mean) / self.pos_std
        if self.pos_weights is not None:
            scaled = scaled * self.pos_weights
        return scaled.astype(np.float32, copy=False)


class SeismicPointGraph(PointGraph):
    """``PointGraph`` subclass for 2D/3D seismic point clouds."""

    def __init__(
        self,
        *,
        mode: Literal["radius", "knn"] = "radius",
        radius: float = 0.25,
        k: int = 16,
        normalize: bool = False,
        sample: int | None = None,
        seed: int = 0,
    ) -> None:
        if mode not in ("radius", "knn"):
            raise ValueError(f"mode must be 'radius' or 'knn', got {mode!r}")
        if mode == "radius" and radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if mode == "knn" and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        super().__init__(normalize=normalize, sample=sample, seed=seed)
        self.mode = mode
        self.radius = float(radius)
        self.k = int(k)

    def edge_parameters(self) -> EdgeParameters:
        return EdgeParameters(self.radius, self.k)

    def neighbour_pairs(
        self, points: np.ndarray, radius: float, k: int
    ) -> NeighbourPairs:
        if self.mode == "radius":
            src, dst = radius_edges(points, radius)
            return NeighbourPairs(src, dst)
        src, dst = knn_edges(points, k)
        return NeighbourPairs(src, dst)


def build_scaled_graph(
    builder: SeismicPointGraph,
    scaler: SharedStandardizer,
    features: np.ndarray,
    locations: np.ndarray,
) -> ScaledGraph:
    """Scale features and locations using ``scaler`` and construct a graph."""
    x_s = scaler.transform_features(features)
    p_s = scaler.transform_locations(locations)
    return ScaledGraph(graph=builder.build(x_s, p_s), features=x_s, locations=p_s)


def graph_num_nodes(data: Data) -> int:
    if data.num_nodes is None:
        raise ValueError("Graph is missing num_nodes")
    return int(data.num_nodes)


def graph_edge_index(data: Data) -> torch.Tensor:
    if data.edge_index is None:
        raise ValueError("Graph is missing edge_index")
    return data.edge_index


def embed_graph(model: NOFE, graph: Data) -> torch.Tensor:
    """Embed a single graph and narrow the return value to a tensor."""
    emb = model.embed(graph)
    if not isinstance(emb, torch.Tensor):
        raise TypeError(f"Expected a single embedding tensor, got {type(emb).__name__}")
    return emb


def embed_numpy(model: NOFE, graph: Data) -> np.ndarray:
    """Embed a single graph as a CPU NumPy array."""
    return embed_graph(model, graph).numpy()


def graph_loss(model: NOFE, graph: Data) -> float:
    """Evaluate scalar KL loss on a single graph."""
    value = model.loss(graph)
    if isinstance(value, list):
        raise TypeError("Expected scalar loss for a single graph, got list")
    return float(value)


def train_nofe(
    model: NOFE,
    graph: Data,
    *,
    epochs: int,
    patience: int,
    lr: float,
    seed: int,
) -> list[dict[str, float]]:
    """Train a NOFE model on a single graph and return its epoch history."""
    history = model.training(
        graph,
        epochs=epochs,
        patience=patience,
        lr=lr,
        verbose=False,
        seed=seed,
    )
    if history and isinstance(history[0], list):
        raise TypeError("Expected single-graph training history, got nested lists")
    return [
        {str(k): float(v) for k, v in row.items()}
        for row in history
        if isinstance(row, dict)
    ]


def fit_and_embed_nofe(graph: Data, params: NofeHyperparams) -> TrainedNofe:
    """Instantiate, train and embed a NOFE operator from ``NofeHyperparams``."""
    t0 = time.perf_counter()
    model = NOFE(
        out_dim=params.out_dim,
        width=params.width,
        ker_width=params.ker_width,
        depth=params.depth,
        nystrom=params.nystrom,
        chunk_size=params.chunk_size,
        device=params.device,
        seed=params.seed,
    )
    history = train_nofe(
        model,
        graph,
        epochs=params.epochs,
        patience=params.patience,
        lr=params.lr,
        seed=params.seed,
    )
    embedding = embed_numpy(model, graph)
    return TrainedNofe(
        model=model,
        history=history,
        train_embedding=embedding,
        elapsed_sec=time.perf_counter() - t0,
    )


def coord_map_2d(locs: np.ndarray) -> dict[tuple[float, float], int]:
    """Index 2D ``(inline, crossline)`` rows to their point positions."""
    return {(float(row[0]), float(row[1])): idx for idx, row in enumerate(locs)}


def coord_map_3d(locs: np.ndarray) -> dict[tuple[float, float, float], int]:
    """Index 3D ``(inline, crossline, depth)`` rows to their point positions."""
    return {
        (float(row[0]), float(row[1]), float(row[2])): idx
        for idx, row in enumerate(locs)
    }


def graph_summary(data: Data) -> dict[str, float | int]:
    """Return node, edge, degree and isolated-node statistics for a built graph."""
    num_nodes = graph_num_nodes(data)
    edge_index = graph_edge_index(data)
    num_edges = int(edge_index.shape[1])
    if num_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "mean_degree": 0.0,
            "min_degree": 0,
            "max_degree": 0,
            "isolated_nodes": 0,
        }
    deg = torch.bincount(edge_index[1], minlength=num_nodes)
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "mean_degree": float(deg.float().mean().item()),
        "min_degree": int(deg.min().item()),
        "max_degree": int(deg.max().item()),
        "isolated_nodes": int((deg == 0).sum().item()),
    }
