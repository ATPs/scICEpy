"""
scICEpy - Single-cell Inconsistency-based Clustering Evaluation (Python implementation)

This module provides tools for evaluating clustering consistency in single-cell RNA-seq data
using the scICE algorithm. It integrates with scanpy/AnnData workflows.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
import warnings
from tqdm import tqdm
import leidenalg
import igraph as ig
from scipy.sparse import issparse
from multiprocessing import Pool, cpu_count
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _leiden_clustering(graph: ig.Graph, resolution: float, objective_function: str = "CPM",
                       n_iterations: int = 10, beta: float = 0.1,
                       initial_membership: Optional[List[int]] = None) -> np.ndarray:
    """
    Perform Leiden clustering on an igraph object.

    Parameters
    ----------
    graph : ig.Graph
        Input graph
    resolution : float
        Resolution parameter
    objective_function : str
        Either "CPM" or "modularity"
    n_iterations : int
        Number of iterations
    beta : float
        Beta parameter for Leiden
    initial_membership : Optional[List[int]]
        Initial cluster membership

    Returns
    -------
    np.ndarray
        Cluster assignments (0-based)
    """
    if objective_function == "CPM":
        partition_type = leidenalg.CPMVertexPartition
        partition = partition_type(graph, resolution_parameter=resolution,
                                    weights="weight" if graph.is_weighted() else None,
                                    initial_membership=initial_membership)
    else:  # modularity
        partition_type = leidenalg.ModularityVertexPartition
        partition = partition_type(graph, resolution_parameter=resolution,
                                    weights="weight" if graph.is_weighted() else None,
                                    initial_membership=initial_membership)

    # Optimize partition
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(np.random.randint(0, 2**31))
    diff = optimiser.optimise_partition(partition, n_iterations=n_iterations)

    return np.array(partition.membership, dtype=np.int16)


def _calculate_ecs(labels_a: np.ndarray, labels_b: np.ndarray, d: float = 0.9) -> Union[float, np.ndarray]:
    """
    Calculate Element-Centric Similarity (ECS) between two clusterings.

    This is a Python implementation of the ECS algorithm used in the Julia and R versions.

    Parameters
    ----------
    labels_a : np.ndarray
        First clustering labels
    labels_b : np.ndarray
        Second clustering labels
    d : float
        Damping factor (default: 0.9)

    Returns
    -------
    float
        Mean ECS score between the two clusterings
    """
    n = len(labels_a)
    unique_a = np.unique(labels_a)
    unique_b = np.unique(labels_b)

    # Create index mappings for clusters
    cluster_idx_a = {label: np.where(labels_a == label)[0] for label in unique_a}
    cluster_idx_b = {label: np.where(labels_b == label)[0] for label in unique_b}

    # Calculate cluster sizes
    c_size_a = d / np.array([len(cluster_idx_a[label]) for label in unique_a])
    c_size_b = d / np.array([len(cluster_idx_b[label]) for label in unique_b])

    # Calculate ECS for each element
    ecs = np.zeros(n)
    unique_ecs_vals = np.full((len(unique_a), len(unique_b)), np.nan)

    ppr1 = np.zeros(n)
    ppr2 = np.zeros(n)

    for i in range(n):
        i1 = np.where(unique_a == labels_a[i])[0][0]
        i2 = np.where(unique_b == labels_b[i])[0][0]

        if np.isnan(unique_ecs_vals[i1, i2]):
            nei1 = cluster_idx_a[labels_a[i]]
            nei2 = cluster_idx_b[labels_b[i]]
            all_idx = np.unique(np.concatenate([nei1, nei2]))

            # Calculate personalized PageRank vectors
            ppr1[nei1] = c_size_a[i1]
            ppr1[i] = 1.0 - d + c_size_a[i1]

            ppr2[nei2] = c_size_b[i2]
            ppr2[i] = 1.0 - d + c_size_b[i2]

            # Calculate ECS as L1 distance
            escore = np.sum(np.abs(ppr2[all_idx] - ppr1[all_idx]))
            ecs[i] = escore

            # Reset vectors
            ppr1[nei1] = 0.0
            ppr1[i] = 0.0
            ppr2[nei2] = 0.0
            ppr2[i] = 0.0

            unique_ecs_vals[i1, i2] = ecs[i]
        else:
            ecs[i] = unique_ecs_vals[i1, i2]

    # Convert to similarity (1 - normalized distance)
    return np.mean(1.0 - (1.0 / (2 * d)) * ecs)


def _extract_clustering_array(clustering_matrix: np.ndarray) -> Dict:
    """
    Extract unique clusterings and their probabilities.

    Parameters
    ----------
    clustering_matrix : np.ndarray
        Matrix where each row is a clustering result

    Returns
    -------
    Dict
        Dictionary with 'arr' (unique clusterings) and 'prob' (probabilities)
    """
    # Convert rows to tuples for hashing
    unique_clusterings = {}
    for row in clustering_matrix:
        key = tuple(row)
        unique_clusterings[key] = unique_clusterings.get(key, 0) + 1

    # Sort by frequency
    sorted_items = sorted(unique_clusterings.items(), key=lambda x: x[1], reverse=True)

    arr = [np.array(k, dtype=np.int16) for k, v in sorted_items]
    counts = np.array([v for k, v in sorted_items])
    prob = counts / counts.sum()

    return {'arr': arr, 'prob': prob}


def _calculate_ecs_pair(args):
    """
    Helper function to calculate ECS for a single pair of clusterings.
    Designed for parallel processing.

    Parameters
    ----------
    args : tuple
        (labels_a, labels_b) - two clustering label arrays

    Returns
    -------
    float
        ECS similarity score
    """
    labels_a, labels_b = args
    return _calculate_ecs(labels_a, labels_b)


def _calculate_ic_from_extracted(extracted: Dict, n_workers: int = 1) -> float:
    """
    Calculate Inconsistency Coefficient (IC) from extracted clusterings.

    Parameters
    ----------
    extracted : Dict
        Extracted clusterings with probabilities
    n_workers : int
        Number of parallel workers

    Returns
    -------
    float
        IC score (1/consistency, so lower is better, 1.0 is perfect)
    """
    if len(extracted['arr']) == 1:
        return 1.0  # Perfect consistency

    nu_mem = extracted['arr']
    prob_arr = extracted['prob']
    n_clusterings = len(nu_mem)

    # Calculate pairwise similarities
    similarities = np.zeros((n_clusterings, n_clusterings))
    np.fill_diagonal(similarities, 1.0)

    # Prepare pairs for parallel processing
    pairs = [(i, j) for i in range(n_clusterings) for j in range(i + 1, n_clusterings)]

    if n_workers > 1 and len(pairs) > 0:
        # Parallel computation
        args_list = [(nu_mem[i], nu_mem[j]) for i, j in pairs]
        # Context manager automatically handles pool.close() and pool.join() on exit
        with Pool(processes=n_workers) as pool:
            sim_results = pool.map(_calculate_ecs_pair, args_list)

        # Fill similarity matrix
        for idx, (i, j) in enumerate(pairs):
            similarities[i, j] = sim_results[idx]
            similarities[j, i] = sim_results[idx]
    else:
        # Sequential computation
        for i, j in pairs:
            sim = _calculate_ecs(nu_mem[i], nu_mem[j])
            similarities[i, j] = sim
            similarities[j, i] = sim

    # Calculate weighted consistency
    consistency = np.dot(similarities @ prob_arr, prob_arr)

    return 1.0 / consistency if consistency > 0 else np.inf


def _get_best_clustering(extracted: Dict) -> np.ndarray:
    """
    Get the best clustering from extracted results.

    The best clustering is the one with highest average similarity to all others.

    Parameters
    ----------
    extracted : Dict
        Extracted clusterings

    Returns
    -------
    np.ndarray
        Best clustering labels
    """
    if len(extracted['arr']) == 1:
        return extracted['arr'][0]

    nu_mem = extracted['arr']
    n_clusterings = len(nu_mem)

    # Calculate pairwise similarities
    similarities = np.zeros((n_clusterings, n_clusterings))
    np.fill_diagonal(similarities, 1.0)

    for i in range(n_clusterings):
        for j in range(i + 1, n_clusterings):
            sim = _calculate_ecs(nu_mem[i], nu_mem[j])
            similarities[i, j] = sim
            similarities[j, i] = sim

    # Choose clustering with highest average similarity
    avg_similarities = similarities.sum(axis=1)
    best_idx = np.argmax(avg_similarities)

    return nu_mem[best_idx]


def _calculate_mei_from_array(extracted: Dict) -> np.ndarray:
    """
    Calculate Mutual Element-wise Information (MEI) scores.

    Parameters
    ----------
    extracted : Dict
        Extracted clusterings

    Returns
    -------
    np.ndarray
        MEI scores for each element
    """
    if len(extracted['arr']) == 1:
        return np.ones(len(extracted['arr'][0]))

    nu_mem = extracted['arr']
    prob_arr = extracted['prob']
    n_clusterings = len(nu_mem)
    n_elements = len(nu_mem[0])

    mei_scores = np.zeros(n_elements)

    # Calculate element-wise similarities
    for i in range(n_clusterings):
        for j in range(i + 1, n_clusterings):
            # Simplified MEI calculation
            same_cluster = (nu_mem[i][:, None] == nu_mem[i]) & (nu_mem[j][:, None] == nu_mem[j])
            mei_contribution = same_cluster.sum(axis=1) / n_elements
            mei_scores += mei_contribution * (prob_arr[i] + prob_arr[j])

    return mei_scores / (n_clusterings - 1) if n_clusterings > 1 else np.ones(n_elements)


def _find_resolution_for_target(args):
    """
    Helper function to find resolution range for a single target cluster number.
    Designed for parallel processing with multiprocessing.Pool.

    Parameters
    ----------
    args : tuple
        (target_clusters, graph, start_g, end_g, objective_function,
         resolution_tolerance, seed)

    Returns
    -------
    tuple
        (target_clusters, (left_bound, right_bound))
    """
    (target_clusters, graph, start_g, end_g, objective_function,
     resolution_tolerance, seed) = args

    n_preliminary_trials = 15
    beta_preliminary = 0.01
    n_iter_preliminary = 5

    if seed is not None:
        np.random.seed(seed + target_clusters * 10)

    # Binary search for lower bound
    left, right = start_g, end_g
    max_iterations = 50
    iteration_count = 0
    effective_tolerance = resolution_tolerance / 10

    while iteration_count < max_iterations:
        if objective_function == "modularity":
            if abs(left - right) <= effective_tolerance:
                break
            mid = (left + right) / 2
            gamma_val = mid
        else:  # CPM
            if abs(np.exp(left) - np.exp(right)) <= effective_tolerance:
                break
            mid = (left + right) / 2
            gamma_val = np.exp(mid)

        # Test clustering
        cluster_results = []
        for _ in range(n_preliminary_trials):
            labels = _leiden_clustering(graph, gamma_val, objective_function,
                                       n_iter_preliminary, beta_preliminary)
            cluster_results.append(len(np.unique(labels)))

        n_clusters_obtained = np.median(cluster_results)

        if n_clusters_obtained < target_clusters:
            left = mid
        else:
            right = mid

        iteration_count += 1

    left_bound = right

    # Binary search for upper bound
    left, right = left_bound, end_g
    iteration_count = 0

    while iteration_count < max_iterations:
        if objective_function == "modularity":
            if abs(left - right) <= effective_tolerance:
                break
            mid = (left + right) / 2
            gamma_val = mid
        else:  # CPM
            if abs(np.exp(left) - np.exp(right)) <= effective_tolerance:
                break
            mid = (left + right) / 2
            gamma_val = np.exp(mid)

        # Test clustering
        cluster_results = []
        for _ in range(n_preliminary_trials):
            labels = _leiden_clustering(graph, gamma_val, objective_function,
                                       n_iter_preliminary, beta_preliminary)
            cluster_results.append(len(np.unique(labels)))

        n_clusters_obtained = np.median(cluster_results)

        if n_clusters_obtained > target_clusters:
            right = mid
        else:
            left = mid

        iteration_count += 1

    right_bound = left

    # Handle identical bounds
    min_cluster_range = 2  # Default assumption
    if left_bound == right_bound or np.isnan(left_bound) or np.isnan(right_bound):
        if objective_function == "CPM":
            center_val = np.exp((start_g + end_g) / 2) if np.isnan(left_bound) else left_bound
            cluster_offset = (target_clusters - min_cluster_range) * 0.05
            adjusted_center = center_val * (1 + cluster_offset)
            left_bound = max(np.exp(start_g), adjusted_center * 0.7)
            right_bound = min(np.exp(end_g), adjusted_center * 1.3)
        else:
            center_val = (start_g + end_g) / 2 if np.isnan(left_bound) else left_bound
            cluster_offset = (target_clusters - min_cluster_range) * 0.02
            adjusted_center = center_val + cluster_offset
            left_bound = max(start_g, adjusted_center - 0.15)
            right_bound = min(end_g, adjusted_center + 0.15)

    if objective_function == "CPM":
        return (target_clusters, (np.exp(left_bound), np.exp(right_bound)))
    else:
        return (target_clusters, (left_bound, right_bound))


def _find_resolution_ranges(
    graph: ig.Graph,
    cluster_range: List[int],
    start_g: float,
    end_g: float,
    objective_function: str,
    resolution_tolerance: float,
    n_workers: int,
    verbose: bool,
    seed: Optional[int] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Find resolution parameter ranges for each cluster number using binary search.

    Parameters
    ----------
    graph : ig.Graph
        Input graph
    cluster_range : List[int]
        Target cluster numbers
    start_g : float
        Start of search range
    end_g : float
        End of search range
    objective_function : str
        "CPM" or "modularity"
    resolution_tolerance : float
        Tolerance for resolution search
    n_workers : int
        Number of workers
    verbose : bool
        Whether to print progress
    seed : Optional[int]
        Random seed

    Returns
    -------
    Dict[int, Tuple[float, float]]
        Dictionary mapping cluster numbers to (min_resolution, max_resolution)
    """
    if verbose:
        logger.info("Starting resolution range search...")
        if n_workers > 1:
            logger.info(f"Using {n_workers} parallel workers")

    gamma_dict = {}

    # Prepare arguments for parallel processing
    args_list = [
        (target_clusters, graph, start_g, end_g, objective_function,
         resolution_tolerance, seed)
        for target_clusters in cluster_range
    ]

    # Use multiprocessing if n_workers > 1
    if n_workers > 1:
        # Context manager automatically handles pool.close() and pool.join() on exit
        with Pool(processes=n_workers) as pool:
            if verbose:
                # Use imap for progress bar support
                results = list(tqdm(
                    pool.imap(_find_resolution_for_target, args_list),
                    total=len(args_list),
                    desc="Resolution search"
                ))
            else:
                results = pool.map(_find_resolution_for_target, args_list)

        # Convert results to dictionary
        for target_clusters, bounds in results:
            gamma_dict[target_clusters] = bounds
    else:
        # Sequential processing for single worker
        for args in tqdm(args_list, desc="Resolution search", disable=not verbose):
            target_clusters, bounds = _find_resolution_for_target(args)
            gamma_dict[target_clusters] = bounds

    if verbose:
        logger.info(f"Found resolution ranges for {len(gamma_dict)} cluster numbers")

    return gamma_dict


def _optimize_clustering_wrapper(args):
    """
    Wrapper function for parallel processing of _optimize_clustering.

    Parameters
    ----------
    args : tuple
        All arguments needed for _optimize_clustering

    Returns
    -------
    Optional[Dict]
        Optimization results with cluster_number added
    """
    (graph, target_clusters, gamma_range, objective_function,
     n_trials, n_bootstrap, seed, beta, n_iterations, max_iterations,
     resolution_tolerance, verbose, n_workers) = args

    result = _optimize_clustering(
        graph, target_clusters, gamma_range, objective_function,
        n_trials, n_bootstrap, seed, beta, n_iterations, max_iterations,
        resolution_tolerance, verbose, n_workers
    )

    if result is not None:
        result['cluster_number'] = target_clusters

    return result


def _optimize_clustering(
    graph: ig.Graph,
    target_clusters: int,
    gamma_range: Tuple[float, float],
    objective_function: str,
    n_trials: int,
    n_bootstrap: int,
    seed: Optional[int],
    beta: float,
    n_iterations: int,
    max_iterations: int,
    resolution_tolerance: float,
    verbose: bool,
    n_workers: int = 1
) -> Optional[Dict]:
    """
    Optimize clustering within a resolution range.

    Parameters
    ----------
    graph : ig.Graph
        Input graph
    target_clusters : int
        Target number of clusters
    gamma_range : Tuple[float, float]
        Resolution parameter range
    objective_function : str
        "CPM" or "modularity"
    n_trials : int
        Number of trials per resolution
    n_bootstrap : int
        Number of bootstrap iterations
    seed : Optional[int]
        Random seed
    beta : float
        Beta parameter
    n_iterations : int
        Number of Leiden iterations
    max_iterations : int
        Maximum optimization iterations
    resolution_tolerance : float
        Resolution tolerance
    verbose : bool
        Whether to print progress
    n_workers : int
        Number of parallel workers for ECS calculations

    Returns
    -------
    Optional[Dict]
        Optimization results or None if failed
    """
    if seed is not None:
        cluster_seed = seed + target_clusters * 1000
        np.random.seed(cluster_seed)

    n_steps = 11
    delta_n = 2

    # Create gamma sequence
    if objective_function == "modularity":
        if gamma_range[0] != gamma_range[1]:
            gamma_sequence = np.linspace(gamma_range[0], gamma_range[1], n_steps)
        else:
            delta_g = resolution_tolerance
            gamma_sequence = np.linspace(gamma_range[0] - delta_g, gamma_range[0] + delta_g, n_steps)
    else:  # CPM
        if gamma_range[0] != gamma_range[1]:
            gamma_sequence = np.exp(np.linspace(np.log(gamma_range[0]), np.log(gamma_range[1]), n_steps))
        else:
            delta_g = resolution_tolerance
            gamma_sequence = np.exp(np.linspace(np.log(gamma_range[0]) - delta_g,
                                               np.log(gamma_range[0]) + delta_g, n_steps))

    # Test initial clustering for each gamma
    clustering_matrices = []
    mean_clusters = []

    for gamma_val in gamma_sequence:
        cluster_matrix = np.zeros((n_trials, graph.vcount()), dtype=np.int16)
        for trial in range(n_trials):
            labels = _leiden_clustering(graph, gamma_val, objective_function, n_iterations, beta)
            cluster_matrix[trial, :] = labels

        clustering_matrices.append(cluster_matrix)
        mean_clusters.append(np.median([len(np.unique(cluster_matrix[i, :]))
                                        for i in range(n_trials)]))

    mean_clusters = np.array(mean_clusters)

    # Filter for target cluster number
    valid_indices = np.where(mean_clusters == target_clusters)[0]

    if len(valid_indices) == 0:
        if verbose:
            logger.warning(f"No gammas produced target cluster count {target_clusters}")
        return None

    gamma_sequence = gamma_sequence[valid_indices]
    clustering_matrices = [clustering_matrices[i] for i in valid_indices]

    # Calculate IC for each gamma
    ic_scores = []
    for cluster_matrix in clustering_matrices:
        extracted = _extract_clustering_array(cluster_matrix)
        ic_result = _calculate_ic_from_extracted(extracted, n_workers)
        ic_scores.append(ic_result)

    ic_scores = np.array(ic_scores)

    # Find best gamma
    best_index = np.where(ic_scores == 1.0)[0]
    if len(best_index) > 0:
        best_index = best_index[0]
    else:
        best_index = np.argmin(ic_scores)

    best_gamma = gamma_sequence[best_index]
    best_clustering = clustering_matrices[best_index]
    k = n_iterations

    # Iterative improvement if not perfect
    if ic_scores[best_index] != 1.0 and len(gamma_sequence) > 1:
        current_matrices = clustering_matrices
        current_gammas = gamma_sequence
        current_ic = ic_scores

        ic_history = np.tile(current_ic, (10, 1)).T

        iteration_count = 0
        while k < max_iterations:
            k += delta_n
            iteration_count += 1

            # Update clustering results
            new_matrices = []
            new_ic = []

            for i, gamma_val in enumerate(current_gammas):
                current_matrix = current_matrices[i]

                new_clustering = np.zeros((n_trials, graph.vcount()), dtype=np.int16)
                for trial in range(n_trials):
                    init_membership = current_matrix[np.random.randint(n_trials), :]
                    labels = _leiden_clustering(graph, gamma_val, objective_function,
                                               delta_n, beta, init_membership.tolist())
                    new_clustering[trial, :] = labels

                new_matrices.append(new_clustering)

                extracted = _extract_clustering_array(new_clustering)
                ic_result = _calculate_ic_from_extracted(extracted, n_workers)
                new_ic.append(ic_result)

            new_ic = np.array(new_ic)

            # Update IC history
            ic_history = np.hstack([ic_history[:, 1:], new_ic[:, None]])

            # Check for convergence
            stable_indices = np.all(ic_history == ic_history[:, 0:1], axis=1)
            perfect_indices = np.where(new_ic == 1.0)[0]

            if len(perfect_indices) > 0:
                best_index = perfect_indices[0]
                best_gamma = current_gammas[best_index]
                best_clustering = new_matrices[best_index]
                break
            elif np.all(stable_indices):
                best_index = np.argmin(new_ic)
                best_gamma = current_gammas[best_index]
                best_clustering = new_matrices[best_index]
                break
            else:
                # Continue with best performing gammas
                keep_indices = (new_ic <= np.quantile(new_ic, 0.5)) | stable_indices
                keep_indices[np.argmin(new_ic)] = True

                if np.sum(keep_indices) == 1:
                    best_index = np.where(keep_indices)[0][0]
                    best_gamma = current_gammas[best_index]
                    best_clustering = new_matrices[best_index]
                    break

                current_gammas = current_gammas[keep_indices]
                current_matrices = [new_matrices[i] for i in np.where(keep_indices)[0]]
                ic_history = ic_history[keep_indices, :]
                current_ic = new_ic[keep_indices]

    # Bootstrap analysis
    ic_bootstrap = []
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(n_trials, n_trials, replace=True)
        bootstrap_matrix = best_clustering[sample_indices, :]
        extracted = _extract_clustering_array(bootstrap_matrix)
        ic_result = _calculate_ic_from_extracted(extracted, n_workers)
        ic_bootstrap.append(ic_result)

    ic_bootstrap = np.array(ic_bootstrap)
    ic_median = np.median(ic_bootstrap)

    # Extract best labels
    extracted_best = _extract_clustering_array(best_clustering)
    best_labels = _get_best_clustering(extracted_best)

    return {
        'gamma': best_gamma,
        'labels': extracted_best,
        'ic_median': ic_median,
        'ic_bootstrap': ic_bootstrap,
        'best_labels': best_labels,
        'n_iterations': k,
        'k': k
    }


def scICE_clustering(
    adata,
    graph_key: str = "connectivities",
    cluster_range: List[int] = None,
    n_workers: int = 10,
    n_trials: int = 15,
    n_bootstrap: int = 100,
    seed: Optional[int] = None,
    beta: float = 0.1,
    n_iterations: int = 10,
    max_iterations: int = 150,
    ic_threshold: float = np.inf,
    objective_function: str = "CPM",
    remove_threshold: float = 1.15,
    resolution_tolerance: float = 1e-8,
    verbose: bool = True,
    copy: bool = False
):
    """
    Single-cell Inconsistency-based Clustering Evaluation (scICE).

    Evaluates clustering consistency in single-cell RNA-seq data using the Leiden
    algorithm and Element-Centric Similarity (ECS).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must have a nearest neighbor graph computed
        (e.g., using sc.pp.neighbors())
    graph_key : str, optional (default: "connectivities")
        Key in adata.obsp containing the graph adjacency matrix
    cluster_range : List[int], optional (default: None)
        Range of cluster numbers to test. If None, defaults to range(2, 21)
    n_workers : int, optional (default: 10)
        Number of parallel workers for multiprocessing. Uses Python's multiprocessing.Pool
        for parallel execution of resolution search and clustering optimization across
        different cluster numbers. Set to 1 for sequential processing.
    n_trials : int, optional (default: 15)
        Number of clustering trials per resolution
    n_bootstrap : int, optional (default: 100)
        Number of bootstrap iterations
    seed : int, optional (default: None)
        Random seed for reproducibility
    beta : float, optional (default: 0.1)
        Beta parameter for Leiden clustering
    n_iterations : int, optional (default: 10)
        Number of Leiden iterations
    max_iterations : int, optional (default: 150)
        Maximum iterations for optimization
    ic_threshold : float, optional (default: np.inf)
        IC threshold for consistent clustering
    objective_function : str, optional (default: "CPM")
        Objective function for Leiden ("CPM" or "modularity")
    remove_threshold : float, optional (default: 1.15)
        Threshold for removing inconsistent results
    resolution_tolerance : float, optional (default: 1e-8)
        Tolerance for resolution parameter search
    verbose : bool, optional (default: True)
        Whether to print progress messages
    copy : bool, optional (default: False)
        Whether to return a copy of adata or modify in place

    Returns
    -------
    If copy=True, returns a copy of adata with scICE results in .uns['scICE'].
    Otherwise, modifies adata in place and returns None.

    The .uns['scICE'] dictionary contains:
        - 'gamma': Resolution parameters for each cluster number
        - 'ic': Inconsistency scores for each cluster number
        - 'ic_vec': Bootstrap IC distributions
        - 'n_cluster': Number of clusters tested
        - 'best_labels': Best clustering labels for each cluster number
        - 'mei': Mutual Element-wise Information scores
        - 'consistent_clusters': Cluster numbers meeting consistency threshold

    Examples
    --------
    >>> import scanpy as sc
    >>> import scICEpy
    >>> adata = sc.datasets.pbmc3k()
    >>> sc.pp.normalize_total(adata)
    >>> sc.pp.log1p(adata)
    >>> sc.pp.highly_variable_genes(adata)
    >>> sc.pp.pca(adata)
    >>> sc.pp.neighbors(adata)
    >>> scICEpy.scICE_clustering(adata, cluster_range=list(range(2, 11)))
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required. Install with: pip install scanpy")

    if copy:
        adata = adata.copy()

    if cluster_range is None:
        cluster_range = list(range(2, 21))

    if seed is not None:
        np.random.seed(seed)

    if verbose:
        logger.info("=" * 80)
        logger.info("Starting scICE clustering analysis...")
        logger.info(f"Testing cluster range: {min(cluster_range)}-{max(cluster_range)} ({len(cluster_range)} values)")
        logger.info(f"Parameters: n_trials={n_trials}, n_bootstrap={n_bootstrap}, objective={objective_function}")
        logger.info("=" * 80)

    # Check for graph
    if graph_key not in adata.obsp:
        raise ValueError(f"Graph '{graph_key}' not found in adata.obsp. "
                        f"Run sc.pp.neighbors() first. Available keys: {list(adata.obsp.keys())}")

    # Convert to igraph
    if verbose:
        logger.info("Converting graph to igraph format...")

    adjacency = adata.obsp[graph_key]
    if issparse(adjacency):
        adjacency = adjacency.tocoo()
        sources = adjacency.row
        targets = adjacency.col
        weights = adjacency.data

        # Create igraph
        n_obs = adata.n_obs
        edges = list(zip(sources.tolist(), targets.tolist()))
        graph = ig.Graph(n=n_obs, edges=edges, directed=False)
        graph.es['weight'] = weights.tolist()
    else:
        raise NotImplementedError("Dense adjacency matrices not yet supported")

    if verbose:
        logger.info(f"Graph: {graph.vcount()} vertices, {graph.ecount()} edges")

    # Determine resolution search bounds
    if objective_function == "modularity":
        start_g = -13
        end_g = 20
    else:  # CPM
        start_g = min(np.log(resolution_tolerance), -20)
        end_g = 20

    # Find resolution ranges
    gamma_dict = _find_resolution_ranges(
        graph, cluster_range, start_g, end_g, objective_function,
        resolution_tolerance, n_workers, verbose, seed
    )

    # Filter out problematic cluster numbers
    if verbose:
        logger.info("Filtering unstable cluster numbers...")

    excluded_numbers = []
    for cluster_num in tqdm(cluster_range, desc="Filtering", disable=not verbose):
        if cluster_num not in gamma_dict:
            excluded_numbers.append(cluster_num)
            continue

        gamma_range = gamma_dict[cluster_num]
        gamma_test = np.linspace(gamma_range[0], gamma_range[1], min(5, 10))

        ic_scores = []
        for gamma_val in gamma_test:
            cluster_results = np.zeros((10, graph.vcount()), dtype=np.int16)
            for i in range(10):
                labels = _leiden_clustering(graph, gamma_val, objective_function, 5, 0.01)
                cluster_results[i, :] = labels

            extracted = _extract_clustering_array(cluster_results)
            ic_result = _calculate_ic_from_extracted(extracted, n_workers)
            ic_scores.append(ic_result)

        if min(ic_scores) >= remove_threshold:
            excluded_numbers.append(cluster_num)

    valid_clusters = [c for c in cluster_range if c not in excluded_numbers]

    if verbose:
        if len(excluded_numbers) > 0:
            logger.info(f"Excluded {len(excluded_numbers)} cluster numbers due to instability: {excluded_numbers}")
        logger.info(f"Optimizing {len(valid_clusters)} cluster numbers...")

    if len(valid_clusters) == 0:
        logger.warning("No valid cluster numbers found!")
        results = {
            'gamma': np.array([]),
            'ic': np.array([]),
            'ic_vec': [],
            'n_cluster': np.array([]),
            'best_labels': [],
            'mei': [],
            'consistent_clusters': np.array([])
        }
    else:
        # Optimize clustering for each valid cluster number
        all_results = []

        # Prepare arguments for parallel processing
        opt_args_list = [
            (graph, cluster_num, gamma_dict[cluster_num], objective_function,
             n_trials, n_bootstrap, seed, beta, n_iterations, max_iterations,
             resolution_tolerance, False, 1)  # verbose=False for parallel, n_workers=1 for nested
            for cluster_num in valid_clusters if cluster_num in gamma_dict
        ]

        if n_workers > 1 and len(opt_args_list) > 1:
            # Parallel processing for multiple cluster numbers
            if verbose:
                logger.info(f"Optimizing {len(opt_args_list)} cluster numbers in parallel with {n_workers} workers")

            # Context manager automatically handles pool.close() and pool.join() on exit
            with Pool(processes=min(n_workers, len(opt_args_list))) as pool:
                if verbose:
                    results = list(tqdm(
                        pool.imap(_optimize_clustering_wrapper, opt_args_list),
                        total=len(opt_args_list),
                        desc="Optimizing"
                    ))
                else:
                    results = pool.map(_optimize_clustering_wrapper, opt_args_list)

            all_results = [r for r in results if r is not None]
        else:
            # Sequential processing or single cluster
            for cluster_num in tqdm(valid_clusters, desc="Optimizing", disable=not verbose):
                if cluster_num not in gamma_dict:
                    continue

                gamma_range = gamma_dict[cluster_num]
                result = _optimize_clustering(
                    graph, cluster_num, gamma_range, objective_function,
                    n_trials, n_bootstrap, seed, beta, n_iterations, max_iterations,
                    resolution_tolerance, verbose, n_workers
                )

                if result is not None:
                    result['cluster_number'] = cluster_num
                    all_results.append(result)

        # Compile results
        if len(all_results) > 0:
            results = {
                'gamma': np.array([r['gamma'] for r in all_results]),
                'ic': np.array([r['ic_median'] for r in all_results]),
                'ic_vec': [r['ic_bootstrap'] for r in all_results],
                'n_cluster': np.array([r['cluster_number'] for r in all_results]),
                'best_labels': [r['best_labels'] for r in all_results],
                'n_iter': np.array([r['n_iterations'] for r in all_results]),
                'k': np.array([r['k'] for r in all_results])
            }

            # Calculate MEI scores
            results['mei'] = [_calculate_mei_from_array(r['labels']) for r in all_results]

            # Determine consistent clusters
            consistent_indices = np.where(results['ic'] < ic_threshold)[0]
            results['consistent_clusters'] = results['n_cluster'][consistent_indices]

            if verbose:
                logger.info("=" * 80)
                logger.info(f"Analysis complete!")
                logger.info(f"Found {len(results['consistent_clusters'])} consistent cluster numbers: "
                          f"{list(results['consistent_clusters'])}")
                if len(excluded_numbers) > 0:
                    logger.info(f"Excluded {len(excluded_numbers)} cluster numbers: {excluded_numbers}")
        else:
            logger.warning("No successful optimizations!")
            results = {
                'gamma': np.array([]),
                'ic': np.array([]),
                'ic_vec': [],
                'n_cluster': np.array([]),
                'best_labels': [],
                'mei': [],
                'consistent_clusters': np.array([])
            }

    # Store results in adata
    adata.uns['scICE'] = results
    adata.uns['scICE']['cluster_range_tested'] = cluster_range

    if copy:
        return adata
    else:
        return None


def get_robust_labels(adata, threshold: float = 1.005, return_adata: bool = False):
    """
    Extract consistent clustering labels from scICE results.

    Parameters
    ----------
    adata : AnnData
        AnnData object with scICE results
    threshold : float, optional (default: 1.005)
        IC threshold for consistency
    return_adata : bool, optional (default: False)
        If True, returns AnnData object with labels added to .obs.
        If False, returns DataFrame with labels.

    Returns
    -------
    pd.DataFrame or AnnData
        If return_adata=False: DataFrame with cell barcodes and cluster labels
            for each consistent clustering
        If return_adata=True: AnnData object with labels added to .obs
            (columns named 'scICE_k_{n}')

    Examples
    --------
    >>> # Get labels as DataFrame
    >>> labels_df = get_robust_labels(adata, threshold=1.005)
    >>>
    >>> # Add labels directly to AnnData object
    >>> adata = get_robust_labels(adata, threshold=1.005, return_adata=True)
    >>> # Access labels: adata.obs['scICE_k_5'] for 5 clusters
    """
    if 'scICE' not in adata.uns:
        raise ValueError("No scICE results found. Run scICE_clustering() first.")

    results = adata.uns['scICE']
    valid_idx = results['ic'] < threshold

    if not np.any(valid_idx):
        warnings.warn(f"No clusterings found below IC threshold {threshold}")
        if return_adata:
            return adata
        else:
            return pd.DataFrame(index=adata.obs_names)

    label_dict = {}
    for i, cluster_num in enumerate(results['n_cluster'][valid_idx]):
        # Use consistent naming convention
        column_name = f'scICE_k_{int(cluster_num)}'
        label_dict[column_name] = results['best_labels'][i]

    if return_adata:
        # Add labels to AnnData object's .obs
        for col_name, labels in label_dict.items():
            adata.obs[col_name] = pd.Categorical(labels)
        return adata
    else:
        # Return as DataFrame
        df = pd.DataFrame(label_dict, index=adata.obs_names)
        return df


def plot_ic(adata, threshold: float = 1.005, figsize: Tuple[float, float] = (8, 6)):
    """
    Plot IC scores across cluster numbers.

    Parameters
    ----------
    adata : AnnData
        AnnData object with scICE results
    threshold : float, optional (default: 1.005)
        IC threshold line to plot
    figsize : Tuple[float, float], optional
        Figure size (width, height)

    Returns
    -------
    matplotlib figure and axis objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")

    if 'scICE' not in adata.uns:
        raise ValueError("No scICE results found. Run scICE_clustering() first.")

    results = adata.uns['scICE']

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for boxplot
    x_data = []
    y_data = []
    for i, cluster_num in enumerate(results['n_cluster']):
        ic_vec = results['ic_vec'][i]
        x_data.extend([cluster_num] * len(ic_vec))
        y_data.extend(ic_vec)

    # Create boxplot
    cluster_numbers = results['n_cluster']
    positions = range(len(cluster_numbers))

    bp = ax.boxplot([results['ic_vec'][i] for i in range(len(cluster_numbers))],
                     positions=positions, widths=0.6)

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

    # Formatting
    ax.set_xlabel('Number of clusters', fontsize=12)
    ax.set_ylabel('IC', fontsize=12)
    ax.set_title('scICE: Inconsistency Coefficient across cluster numbers', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels([int(k) for k in cluster_numbers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, ax
