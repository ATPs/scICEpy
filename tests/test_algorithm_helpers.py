import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scICEpy.optimization import (
    merge_small_clusters_to_neighbors,
    refine_gamma_candidates_by_raw_gap,
    select_gamma_admission,
)
from scICEpy.resolution_search import (
    clamp_gamma_range_to_raw_plateau,
    classify_resolution_search_state,
    count_effective_clusters,
    derive_shared_gamma_intervals,
    raw_cluster_guard_limits,
    raw_cluster_search_upper,
)
from scICEpy.results import finalize_cluster_range_results


def _candidate(
    cluster_number: int,
    source_target_cluster: int,
    gamma: float,
    ic: float,
    best_labels,
):
    labels = {"arr": [np.asarray(best_labels, dtype=np.int32)], "prob": np.asarray([1.0]), "parr": np.asarray([1.0])}
    return {
        "cluster_number": int(cluster_number),
        "source_target_cluster": int(source_target_cluster),
        "gamma": float(gamma),
        "labels": labels,
        "ic_median": float(ic),
        "ic_bootstrap": np.asarray([float(ic)], dtype=float),
        "best_labels": np.asarray(best_labels, dtype=np.int32),
        "effective_cluster_median": float(len(np.unique(best_labels))),
        "raw_cluster_median": float(len(np.unique(best_labels))),
        "final_cluster_median": float(len(np.unique(best_labels))),
        "admission_mode": "strict_soft",
        "best_labels_raw_cluster_count": int(len(np.unique(best_labels))),
        "best_labels_final_cluster_count": int(len(np.unique(best_labels))),
        "n_iterations": 10,
        "mei": np.ones(len(best_labels), dtype=float),
        "k": 10,
        "excluded": False,
        "exclusion_reason": "none",
        "result_status": "candidate",
        "phase1_primary_gamma_count": 2,
        "phase1_secondary_gamma_count": 1,
        "phase1_total_gamma_count": 3,
        "phase1_elapsed_sec": 0.1,
        "phase1_leiden_runs": 6,
        "secondary_phase1_used": True,
        "exact_hit_gamma_count": 1,
        "phase4_iterations": 1,
        "phase4_elapsed_sec": 0.1,
        "phase5_elapsed_sec": 0.1,
        "optimization_elapsed_sec": 0.3,
    }


def test_effective_cluster_count_and_raw_guards():
    labels = np.asarray([0, 0, 0, 1, 2], dtype=np.int32)
    assert count_effective_clusters(labels, min_cluster_size=1) == 3
    assert count_effective_clusters(labels, min_cluster_size=2) == 1
    assert raw_cluster_guard_limits(10) == {"soft": 13, "hard": 15}
    assert raw_cluster_search_upper(5) == 6
    assert raw_cluster_search_upper(20) == 22


def test_plateau_clamp_and_search_state():
    gamma_sequence = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    raw_cluster_medians = np.asarray([2.0, 3.0, 3.0, 4.0], dtype=float)
    clamped = clamp_gamma_range_to_raw_plateau(gamma_sequence, raw_cluster_medians, target_clusters=3)
    assert clamped["mode"] == "raw_exact"
    assert np.allclose(clamped["bounds"], np.asarray([2.0, 3.0], dtype=float))

    state = classify_resolution_search_state(
        raw_cluster_median=12,
        effective_cluster_median=8,
        target_clusters=8,
        min_cluster_size=3,
    )
    assert state["raw_class"] == "raw_above_soft"
    assert state["over_fragmented"] is False
    assert state["raw_guard_soft"] is False


def test_gamma_admission_and_raw_gap_refinement():
    admission = select_gamma_admission(
        strict_flags=np.asarray([False, True, False]),
        relaxed_flags=np.asarray([True, True, False]),
        soft_guard_flags=np.asarray([True, True, True]),
        hard_guard_flags=np.asarray([True, True, True]),
        raw_strict_flags=np.asarray([False, False, False]),
        raw_relaxed_flags=np.asarray([False, False, True]),
    )
    assert admission["mode"] == "strict_soft"
    assert admission["indices"] == [1]

    refined = refine_gamma_candidates_by_raw_gap(
        valid_indices=[0, 1],
        admission_mode="relaxed_soft",
        gamma_results=[
            {"raw_median_gap": 1.0, "mean_clusters": 4.0},
            {"raw_median_gap": 0.0, "mean_clusters": 5.0},
        ],
        target_clusters=5,
        min_cluster_size=2,
    )
    assert refined["indices"] == [1]
    assert refined["best_raw_gap"] == 0.0


def test_merge_small_clusters_to_neighbors():
    graph = csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.1, 0.1],
                [1.0, 0.0, 0.1, 0.8],
                [0.1, 0.1, 0.0, 0.2],
                [0.1, 0.8, 0.2, 0.0],
            ]
        )
    )
    labels = np.asarray([0, 0, 1, 2], dtype=np.int32)
    merged = merge_small_clusters_to_neighbors(labels, graph, min_cluster_size=2)
    assert np.array_equal(merged, np.asarray([0, 0, 0, 0], dtype=np.int32))


def test_finalize_cluster_range_results_rekeys_by_final_cluster():
    target_results = [
        _candidate(cluster_number=2, source_target_cluster=2, gamma=0.10, ic=1.20, best_labels=[0, 0, 1, 1]),
        _candidate(cluster_number=3, source_target_cluster=3, gamma=0.20, ic=1.05, best_labels=[0, 0, 1, 1]),
    ]
    final_results = finalize_cluster_range_results(
        target_results=target_results,
        requested_cluster_range=np.asarray([2, 3], dtype=int),
        searched_target_cluster_range=np.asarray([2, 3], dtype=int),
        search_coverage_complete=True,
        gamma_dict={2: (0.05, 0.15), 3: (0.15, 0.25)},
    )
    assert np.array_equal(final_results["n_cluster"], np.asarray([2], dtype=int))
    assert np.array_equal(final_results["source_target_cluster"], np.asarray([3.0]))
    assert len(final_results["target_diagnostics"]) == 2
    assert np.array_equal(final_results["uncovered_targets"], np.asarray([3], dtype=int))
    diag = final_results["target_diagnostics"].sort_values("requested_target_cluster").reset_index(drop=True)
    assert "returned_final_cluster" in diag.columns
    assert "superseded_by_source_target_cluster" in diag.columns
    assert bool(diag.loc[0, "returned_in_main_result"]) is False
    assert int(diag.loc[0, "superseded_by_source_target_cluster"]) == 3


def test_derive_shared_gamma_intervals_uses_objective_function_midpoint():
    probes = pd.DataFrame(
        {
            "gamma": np.asarray([1.0, 9.0], dtype=float),
            "final_cluster_count": np.asarray([2.0, 4.0], dtype=float),
            "raw_cluster_count": np.asarray([2.0, 4.0], dtype=float),
        }
    )
    cpm_state = derive_shared_gamma_intervals(
        probes,
        cluster_range=np.asarray([3], dtype=int),
        gamma_bounds=(1.0, 9.0),
        objective_function="CPM",
    )
    modularity_state = derive_shared_gamma_intervals(
        probes,
        cluster_range=np.asarray([3], dtype=int),
        gamma_bounds=(1.0, 9.0),
        objective_function="modularity",
    )
    assert np.any(np.isclose(cpm_state["target_gamma_seeds"]["3"], 3.0))
    assert np.any(np.isclose(modularity_state["target_gamma_seeds"]["3"], 5.0))


def test_derive_shared_gamma_intervals_keeps_raw_and_final_evidence():
    probes = pd.DataFrame(
        {
            "gamma": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
            "final_cluster_count": np.asarray([2.0, 2.0, 4.0, 4.0], dtype=float),
            "raw_cluster_count": np.asarray([2.0, 3.0, 3.0, 4.0], dtype=float),
        }
    )
    state = derive_shared_gamma_intervals(
        probes,
        cluster_range=np.asarray([3], dtype=int),
        gamma_bounds=(1.0, 4.0),
        objective_function="modularity",
    )
    detail = state["target_interval_details"]["3"]
    diagnostics = state["annotated_probes_df"]
    assert detail["optimization_ready"] is True
    assert detail["raw_interval_mode"] == "raw_exact"
    assert detail["mode"].startswith("final_bracket")
    assert "raw_exact_targets" in diagnostics.columns
    assert diagnostics["raw_exact_targets"].astype(str).str.contains("3").any()
