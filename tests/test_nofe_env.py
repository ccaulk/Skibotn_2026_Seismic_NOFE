"""Environment and device compatibility checks for NOFE on macOS."""

import numpy as np
import torch
from nofe import NOFE, PointGraph, backend
from nofe.search import radius_edges

from nofe_seismic.graph import embed_graph, graph_num_nodes, train_nofe


class SyntheticRadiusGraph(PointGraph):
    def __init__(
        self,
        radius: float = 0.6,
        *,
        normalize: bool = True,
        sample: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(normalize=normalize, sample=sample, seed=seed)
        self.radius = radius

    def edge_parameters(self) -> tuple[float, int]:
        return self.radius, 0

    def neighbour_pairs(
        self,
        points: np.ndarray,
        radius: float,
        k: int,  # noqa: ARG002 -- required by PointGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        return radius_edges(points, radius)


def _make_graph(
    num_nodes: int = 120, feat_dim: int = 6, pos_dim: int = 3, seed: int = 0
):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_nodes, feat_dim)).astype(np.float32)
    locations = rng.uniform(-1.0, 1.0, size=(num_nodes, pos_dim)).astype(np.float32)
    return SyntheticRadiusGraph(radius=0.8).build(features, locations)


def test_nofe_backend_is_valid() -> None:
    assert backend() in ("ckdtree", "faiss")


def test_nofe_dense_and_nystrom_cpu() -> None:
    graph = _make_graph()
    dense = NOFE(out_dim=3, width=8, ker_width=32, depth=3, device="cpu", seed=42)
    hist_dense = train_nofe(dense, graph, epochs=5, patience=5, lr=1e-2, seed=42)
    emb_dense = embed_graph(dense, graph)
    assert len(hist_dense) == 5
    assert tuple(emb_dense.shape) == (graph_num_nodes(graph), 3)
    assert torch.isfinite(emb_dense).all()

    nystrom = NOFE(
        out_dim=3,
        width=8,
        ker_width=32,
        depth=3,
        nystrom=0.25,
        chunk_size=32,
        device="cpu",
        seed=42,
    )
    hist_nys = train_nofe(nystrom, graph, epochs=5, patience=5, lr=1e-2, seed=42)
    emb_nys = embed_graph(nystrom, graph)
    assert len(hist_nys) == 5
    assert tuple(emb_nys.shape) == (graph_num_nodes(graph), 3)
    assert torch.isfinite(emb_nys).all()


def test_nofe_mps_if_available() -> None:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    graph = _make_graph()
    model = NOFE(out_dim=3, width=8, ker_width=32, depth=3, device="mps", seed=42)
    hist = train_nofe(model, graph, epochs=3, patience=3, lr=1e-2, seed=42)
    emb = embed_graph(model, graph)
    assert len(hist) == 3
    assert tuple(emb.shape) == (graph_num_nodes(graph), 3)
    assert torch.isfinite(emb).all()
