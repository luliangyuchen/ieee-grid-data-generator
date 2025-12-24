from types import SimpleNamespace

import numpy as np

import overview


def test_is_graph_connected():
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    outage = np.array([False, False])
    assert overview.is_graph_connected(3, edges, outage) is True

    outage = np.array([True, False])
    assert overview.is_graph_connected(3, edges, outage) is False


def test_compute_comb_space():
    total, target, truncated = overview.compute_comb_space(5, 2, 3)
    assert total == 10
    assert target == 3
    assert truncated is True


def test_build_connected_topology_set_non_truncated():
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    rng = np.random.default_rng(1)
    total, target, truncated = overview.compute_comb_space(2, 1, 10)
    connected = overview.build_connected_topology_set(
        num_nodes=3,
        edges_uv=edges,
        k=1,
        total_comb=total,
        target_size=target,
        truncated=truncated,
        rng=rng,
        sample_max_tries=None,
    )
    assert connected == []


def test_overview_smoke_case39():
    args = SimpleNamespace(
        case="IEEE39",
        seed=1,
        k=0,
        max_size=1,
        power_level=[1.0],
        samples_per_level=[1],
        sample_max_tries=0,
        preview_topologies=1,
    )
    result = overview.overview(args)
    assert result["case"] == "IEEE39"
    assert result["k"] == 0
    assert result["samples_per_topology"] == 1
