"""Leiden and graph conversion helpers for scICEpy."""

from __future__ import annotations

from typing import Sequence

import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import issparse

from .runtime import clustering_cache_env

_BETA_DEFAULT = 0.1
def beta_support_status() -> dict[str, object]:
    return {
        "supported": False,
        "applied": False,
        "default": float(_BETA_DEFAULT),
        "reason": "python-leidenalg 0.10.2 does not expose a beta control on the Optimiser path used by scICEpy.",
    }


def graph_to_igraph(adjacency) -> ig.Graph:
    if not issparse(adjacency):
        raise NotImplementedError("Dense adjacency matrices are not supported.")

    coo = adjacency.tocoo()
    mask = coo.row < coo.col
    edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
    graph = ig.Graph(n=adjacency.shape[0], edges=edges, directed=False)
    graph.es["weight"] = coo.data[mask].astype(float).tolist()
    return graph


def _cache_key(
    resolution: float,
    objective_function: str,
    n_iterations: int,
    beta: float,
    cache_key_suffix: str = "",
) -> str:
    return "_".join(
        [
            "r",
            f"{float(resolution):.8f}",
            "obj",
            objective_function,
            "iter",
            str(int(n_iterations)),
            "beta",
            f"{float(beta):.4f}",
            "suffix",
            cache_key_suffix,
        ]
    )


def leiden_clustering(
    graph: ig.Graph,
    resolution: float,
    objective_function: str = "CPM",
    n_iterations: int = 10,
    beta: float = 0.1,
    initial_membership: Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if objective_function == "CPM":
        partition = leidenalg.CPMVertexPartition(
            graph,
            resolution_parameter=resolution,
            weights="weight" if graph.is_weighted() else None,
            initial_membership=list(initial_membership) if initial_membership is not None else None,
        )
    else:
        partition = leidenalg.ModularityVertexPartition(
            graph,
            weights="weight" if graph.is_weighted() else None,
            initial_membership=list(initial_membership) if initial_membership is not None else None,
        )

    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(int(seed if seed is not None else np.random.randint(0, 2**31 - 1)))
    optimiser.optimise_partition(partition, n_iterations=int(n_iterations))
    return np.asarray(partition.membership, dtype=np.int32)


def cached_leiden_clustering(
    graph: ig.Graph,
    resolution: float,
    objective_function: str,
    n_iterations: int,
    beta: float,
    initial_membership: Sequence[int] | None = None,
    use_cache: bool = True,
    cache_key_suffix: str = "",
    seed: int | None = None,
) -> np.ndarray:
    if not use_cache or initial_membership is not None:
        return leiden_clustering(
            graph=graph,
            resolution=resolution,
            objective_function=objective_function,
            n_iterations=n_iterations,
            beta=beta,
            initial_membership=initial_membership,
            seed=seed,
        )

    key = _cache_key(resolution, objective_function, n_iterations, beta, cache_key_suffix)
    if key in clustering_cache_env:
        return clustering_cache_env[key].copy()

    result = leiden_clustering(
        graph=graph,
        resolution=resolution,
        objective_function=objective_function,
        n_iterations=n_iterations,
        beta=beta,
        initial_membership=initial_membership,
        seed=seed,
    )
    clustering_cache_env[key] = result.copy()
    return result
