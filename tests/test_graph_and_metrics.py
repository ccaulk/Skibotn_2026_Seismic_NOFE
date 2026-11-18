"""Tests for seismic graph construction, standardisation, baselines and metrics."""

import numpy as np
import torch

from nofe_seismic.baselines import run_pca, run_tsne
from nofe_seismic.graph import SeismicPointGraph, SharedStandardizer, graph_summary
from nofe_seismic.metrics import (
    evaluate_clustering,
    local_graph_metrics,
    neighborhood_metrics,
)


def test_shared_standardizer_and_seismic_graph() -> None:
    rng = np.random.default_rng(7)
    feat_train = rng.normal(loc=5.0, scale=2.0, size=(80, 4)).astype(np.float32)
    loc_train = rng.uniform(0.0, 100.0, size=(80, 3)).astype(np.float32)
    scaler = SharedStandardizer.fit(feat_train, loc_train, pos_weights=(1.0, 1.0, 0.5))

    x_s = scaler.transform_features(feat_train)
    p_s = scaler.transform_locations(loc_train)
    assert np.allclose(x_s.mean(0), 0.0, atol=1e-5)
    assert np.allclose(p_s.std(0), np.array([1.0, 1.0, 0.5]), atol=1e-4)

    g = SeismicPointGraph(mode="radius", radius=1.2, normalize=False).build(x_s, p_s)
    stats = graph_summary(g)
    assert stats["num_nodes"] == 80
    assert stats["num_edges"] > 0
    assert g.edge_attr is not None
    assert g.edge_attr.shape[1] == 7


def test_metrics_identity_and_clustering() -> None:
    rng = np.random.default_rng(11)
    centers = np.array(
        [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-5.0, 5.0, -5.0]], dtype=np.float32
    )
    labels = np.repeat(np.arange(3), 30)
    x = centers[labels] + rng.normal(scale=0.3, size=(90, 3)).astype(np.float32)

    m = neighborhood_metrics(x, x, k=10, sample_size=90)
    assert np.isclose(m["trustworthiness"], 1.0, atol=1e-6)
    assert np.isclose(m["continuity"], 1.0, atol=1e-6)
    assert np.isclose(m["knn_recall"], 1.0, atol=1e-6)
    assert np.isclose(m["global_stress"], 0.0, atol=1e-6)

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    lm = local_graph_metrics(x, x * 2.5, edge_index)
    assert np.isclose(lm["local_stress"], 0.0, atol=1e-6)

    clust = evaluate_clustering(x, k_values=(3,), labels=labels)
    assert clust[0]["ari"] > 0.95
    assert clust[0]["nmi"] > 0.95

    pca_emb, _ = run_pca(x, out_dim=2)
    assert pca_emb.shape == (90, 2)
    tsne_res = run_tsne(x, out_dim=2, perplexity=10.0)
    assert tsne_res.embedding.shape == (90, 2)
