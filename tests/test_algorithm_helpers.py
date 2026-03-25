import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scICEpy.api import (
    _build_target_worker_budgets,
    _map_optimized_targets,
    _should_use_global_phase1_process_pool,
)
from scICEpy.optimization import (
    _evaluate_gamma,
    derive_gamma_admission_state,
    merge_small_clusters_to_neighbors,
    order_gamma_candidate_indices,
    optimize_clustering,
    refine_gamma_candidates_by_raw_gap,
    select_gamma_admission,
)
from scICEpy.resolution_search import (
    clamp_gamma_range_to_raw_plateau,
    classify_resolution_search_state,
    count_effective_clusters,
    discover_cpm_upper_gamma,
    derive_shared_gamma_intervals,
    find_resolution_ranges,
    raw_cluster_guard_limits,
    raw_cluster_search_upper,
    resolve_search_worker_capacity,
    resolve_search_probe_workers,
)
from scICEpy.results import finalize_cluster_range_results
from scICEpy.runtime import RuntimeContext, resolve_nested_worker_layout


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
    labels = np.asarray([10, 10, 10, 21, 35], dtype=np.int32)
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


def test_refine_gamma_candidates_preserves_multiple_exact_hits_for_non_raw_admission():
    refined = refine_gamma_candidates_by_raw_gap(
        valid_indices=[0, 1, 2],
        admission_mode="relaxed_unguarded",
        gamma_results=[
            {"hit_count": 1, "raw_median_gap": 60.0, "mean_clusters": 15.0},
            {"hit_count": 2, "raw_median_gap": 55.0, "mean_clusters": 15.5},
            {"hit_count": 0, "raw_median_gap": 1.0, "mean_clusters": 16.0},
        ],
        target_clusters=15,
        min_cluster_size=3,
    )
    assert refined["indices"] == [0, 1]
    assert refined["best_raw_gap"] == 55.0


def test_select_gamma_admission_extends_relaxed_family_with_exact_hit_support():
    admission = select_gamma_admission(
        strict_flags=np.asarray([False, False, False]),
        relaxed_flags=np.asarray([True, False, False]),
        soft_guard_flags=np.asarray([False, False, False]),
        hard_guard_flags=np.asarray([False, False, False]),
        raw_strict_flags=np.asarray([False, False, False]),
        raw_relaxed_flags=np.asarray([False, False, False]),
        exact_hit_flags=np.asarray([True, True, False]),
    )
    assert admission["mode"] == "exact_hit_supported"
    assert admission["indices"] == [0, 1]


def test_derive_gamma_admission_state_no_longer_uses_python_only_exact_hit_rescue():
    gamma_results = [
        {
            "gamma": 3.87e-6,
            "strict_valid": False,
            "relaxed_valid": False,
            "raw_strict_valid": False,
            "raw_relaxed_valid": False,
            "raw_guard_soft": False,
            "raw_guard_hard": False,
            "hit_count": 1,
            "raw_hit_count": 0,
            "final_cluster_median": 12.0,
            "median_gap": 2.0,
        },
        {
            "gamma": 5.16e-6,
            "strict_valid": False,
            "relaxed_valid": False,
            "raw_strict_valid": False,
            "raw_relaxed_valid": False,
            "raw_guard_soft": False,
            "raw_guard_hard": False,
            "hit_count": 0,
            "raw_hit_count": 0,
            "final_cluster_median": 15.5,
            "median_gap": 1.5,
        },
    ]
    state = derive_gamma_admission_state(
        gamma_results=gamma_results,
        target_clusters=14,
        min_cluster_size=3,
    )
    assert state["admission_mode"] == "none"
    assert state["valid_indices"] == []


def test_order_gamma_candidate_indices_prefers_right_shifted_exact_hits_for_high_k():
    order = order_gamma_candidate_indices(
        [
            {
                "gamma": 5.150685670830422e-06,
                "ic": 1.03269574924928,
                "hit_count": 1,
                "final_cluster_median": 15.0,
                "relaxed_valid": True,
            },
            {
                "gamma": 5.306894792842963e-06,
                "ic": 1.03315706591798,
                "hit_count": 2,
                "final_cluster_median": 15.0,
                "relaxed_valid": True,
            },
        ],
        target_clusters=15,
        exact_support_flags=np.asarray([True, True]),
        prefer_right_exact_hits=True,
    )
    assert order == [1, 0]


def test_order_gamma_candidate_indices_prioritizes_finalizable_candidates_before_exact_support_history():
    order = order_gamma_candidate_indices(
        [
            {
                "gamma": 6.24134990952689e-06,
                "ic": 1.028046,
                "hit_count": 1,
                "final_cluster_median": 19.0,
                "relaxed_valid": True,
            },
            {
                "gamma": 6.81119773797437e-06,
                "ic": 1.010501,
                "hit_count": 0,
                "final_cluster_median": np.nan,
                "relaxed_valid": False,
            },
        ],
        target_clusters=19,
        exact_support_flags=np.asarray([True, True]),
        finalizable_flags=np.asarray([True, False]),
        prefer_right_exact_hits=True,
    )
    assert order == [0, 1]


def test_order_gamma_candidate_indices_uses_conservative_order_within_finalizable_pool():
    order = order_gamma_candidate_indices(
        [
            {
                "gamma": 6.81119773797437e-06,
                "ic": 1.072079,
                "hit_count": 1,
                "final_cluster_median": 20.0,
                "strict_valid": True,
                "relaxed_valid": True,
            },
            {
                "gamma": 8.85234560513838e-06,
                "ic": 1.140429,
                "hit_count": 1,
                "final_cluster_median": 23.5,
                "strict_valid": False,
                "relaxed_valid": False,
            },
        ],
        target_clusters=20,
        exact_support_flags=np.asarray([True, True]),
        finalizable_flags=np.asarray([True, True]),
        prefer_right_exact_hits=True,
    )
    assert order == [0, 1]


def test_evaluate_gamma_retains_ic_for_exact_hit_supported_candidate(monkeypatch):
    class _Graph:
        def vcount(self):
            return 10

    trial_labels = [
        np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1, 2, 3, 3, 4, 4, 5], dtype=np.int32),
    ]
    call_state = {"idx": 0}

    def fake_leiden_clustering(**kwargs):
        idx = call_state["idx"]
        call_state["idx"] += 1
        return trial_labels[idx]

    monkeypatch.setattr("scICEpy.optimization.leiden_clustering", fake_leiden_clustering)
    monkeypatch.setattr("scICEpy.optimization.calculate_ic_from_extracted", lambda extracted, n_workers=1: 1.2345)
    result = _evaluate_gamma(
        graph=_Graph(),
        gamma_val=5.0e-06,
        target_clusters=3,
        objective_function="CPM",
        n_trials=2,
        beta=0.1,
        n_iterations=10,
        seed=123,
        snn_graph=None,
        min_cluster_size=1,
        worker_id="TEST",
        verbose=False,
        runtime_context=None,
    )
    assert int(result["hit_count"]) == 1
    assert bool(result["strict_valid"]) is False
    assert bool(result["relaxed_valid"]) is False
    assert float(result["ic"]) == pytest.approx(1.2345)
    assert result["matrix_ref"] is not None


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
    assert np.array_equal(final_results["source_target_cluster"], np.asarray([2.0]))
    assert len(final_results["target_diagnostics"]) == 2
    assert np.array_equal(final_results["uncovered_targets"], np.asarray([3], dtype=int))
    diag = final_results["target_diagnostics"].sort_values("requested_target_cluster").reset_index(drop=True)
    assert "returned_final_cluster" in diag.columns
    assert "superseded_by_source_target_cluster" in diag.columns
    assert bool(diag.loc[0, "returned_in_main_result"]) is True
    assert bool(diag.loc[1, "returned_in_main_result"]) is False
    assert str(diag.loc[1, "exclusion_reason"]) == "final_cluster_mismatch"


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


def test_derive_shared_gamma_intervals_expands_upper_tail_bounds():
    probes = pd.DataFrame(
        {
            "gamma": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
            "final_cluster_count": np.asarray([13.0, 15.0, 15.0, 16.0, 17.0], dtype=float),
            "raw_cluster_count": np.asarray([50.0, 60.0, 70.0, 80.0, 90.0], dtype=float),
        }
    )
    state = derive_shared_gamma_intervals(
        probes,
        cluster_range=np.asarray([13, 14, 15, 16, 17], dtype=int),
        gamma_bounds=(1.0, 5.0),
        objective_function="modularity",
    )
    detail = state["target_interval_details"]["15"]
    assert detail["mode"] == "final_exact"
    assert detail["gamma_left"] == pytest.approx(2.0)
    assert detail["gamma_right"] == pytest.approx(4.0)
    assert 4.0 in state["target_gamma_seeds"]["15"]


def test_derive_shared_gamma_intervals_keeps_raw_exact_as_seed_not_left_bound():
    probes = pd.DataFrame(
        {
            "gamma": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
            "final_cluster_count": np.asarray([13.0, 15.0, 15.0, 16.0, 17.0], dtype=float),
            "raw_cluster_count": np.asarray([15.0, 40.0, 55.0, 80.0, 95.0], dtype=float),
        }
    )
    state = derive_shared_gamma_intervals(
        probes,
        cluster_range=np.asarray([13, 14, 15, 16, 17], dtype=int),
        gamma_bounds=(1.0, 5.0),
        objective_function="modularity",
    )
    detail = state["target_interval_details"]["15"]
    assert detail["mode"] == "final_exact"
    assert detail["gamma_left"] == pytest.approx(2.0)
    assert detail["gamma_right"] == pytest.approx(4.0)
    assert 1.0 in state["target_gamma_seeds"]["15"]


def test_discover_cpm_upper_gamma_keeps_post_coverage_buffer(monkeypatch):
    def fake_probe_batch(
        graph,
        gamma_values,
        sweep_round,
        objective_function,
        n_iter_preliminary,
        beta_preliminary,
        requested_max,
        min_cluster_size,
        snn_graph,
        active_probe_workers,
        verbose,
        seed,
        probe_stage,
        discovery_round=None,
        coarse_probe_count=None,
        probe_metadata=None,
        runtime_context=None,
    ):
        gamma_values = np.asarray(gamma_values, dtype=float)
        return pd.DataFrame(
            {
                "gamma": gamma_values,
                "final_cluster_count": np.repeat(float(requested_max), gamma_values.size),
                "raw_cluster_count": np.repeat(float(requested_max + 25), gamma_values.size),
                "effective_cluster_count": np.repeat(float(requested_max), gamma_values.size),
                "degenerate_high_gamma": np.repeat(False, gamma_values.size),
            }
        )

    monkeypatch.setattr("scICEpy.resolution_search.global_resolution_search_probe_batch", fake_probe_batch)

    class _Graph:
        def vcount(self):
            return 245878

    state = discover_cpm_upper_gamma(
        graph=_Graph(),
        gamma_bounds=(1.0, 100.0),
        requested_max=20,
        n_iter_preliminary=3,
        beta_preliminary=0.01,
        min_cluster_size=3,
        snn_graph=None,
        active_probe_workers=2,
        verbose=False,
        seed=123,
    )
    assert state["upper_cap_stop_reason"] == "post_coverage_buffer"
    assert state["coverage_upper_gamma"] == pytest.approx(1.0)
    assert float(state["discovered_upper_gamma"]) > 4.0


def test_resolve_search_probe_workers_scale_with_targets_and_planned_probe_count():
    capacity_small = resolve_search_worker_capacity(
        requested_workers=40,
        n_vertices=245878,
        n_preliminary_trials=3,
        min_cluster_size=3,
        target_count=4,
        runtime_context=None,
    )
    capacity_large = resolve_search_worker_capacity(
        requested_workers=40,
        n_vertices=245878,
        n_preliminary_trials=3,
        min_cluster_size=3,
        target_count=19,
        runtime_context=None,
    )
    workers = resolve_search_probe_workers(
        requested_workers=40,
        n_vertices=245878,
        n_preliminary_trials=3,
        min_cluster_size=3,
        target_count=19,
        planned_probe_count=5,
        runtime_context=None,
    )
    assert capacity_large > capacity_small
    assert workers == 5


def test_find_resolution_ranges_uses_first_coverage_gamma_for_coarse_grid(monkeypatch):
    captured_gamma_batches: dict[str, list[np.ndarray]] = {"coarse": []}

    def fake_discover_upper_gamma(
        graph,
        gamma_bounds,
        requested_max,
        n_iter_preliminary,
        beta_preliminary,
        min_cluster_size,
        snn_graph,
        active_probe_workers,
        verbose,
        seed,
        runtime_context=None,
        target_count=1,
    ):
        return {
            "probe_results": pd.DataFrame(
                {
                    "gamma": np.asarray([1.0, 4.0, 10.0, 40.0, 100.0], dtype=float),
                    "final_cluster_count": np.asarray([2.0, 4.0, 20.0, 40.0, 80.0], dtype=float),
                    "raw_cluster_count": np.asarray([2.0, 4.0, 30.0, 60.0, 120.0], dtype=float),
                    "effective_cluster_count": np.asarray([2.0, 4.0, 20.0, 40.0, 80.0], dtype=float),
                    "degenerate_high_gamma": np.asarray([False, False, False, False, False], dtype=bool),
                }
            ),
            "discovered_upper_gamma": 100.0,
            "coverage_upper_gamma": 10.0,
            "upper_cap_stop_reason": "post_coverage_buffer",
        }

    def fake_probe_batch(
        graph,
        gamma_values,
        sweep_round,
        objective_function,
        n_iter_preliminary,
        beta_preliminary,
        requested_max,
        min_cluster_size,
        snn_graph,
        active_probe_workers,
        verbose,
        seed,
        probe_stage,
        coarse_probe_count=None,
        discovery_round=None,
        probe_metadata=None,
        runtime_context=None,
    ):
        gamma_values = np.asarray(gamma_values, dtype=float)
        if probe_stage == "coarse":
            captured_gamma_batches["coarse"].append(gamma_values.copy())
        final_counts = np.linspace(2.0, 20.0, num=max(1, gamma_values.size), dtype=float)
        return pd.DataFrame(
            {
                "gamma": gamma_values,
                "final_cluster_count": final_counts,
                "raw_cluster_count": final_counts + 10.0,
                "effective_cluster_count": final_counts,
                "degenerate_high_gamma": np.repeat(False, gamma_values.size),
            }
        )

    monkeypatch.setattr("scICEpy.resolution_search.discover_cpm_upper_gamma", fake_discover_upper_gamma)
    monkeypatch.setattr("scICEpy.resolution_search.global_resolution_search_probe_batch", fake_probe_batch)

    class _Graph:
        def vcount(self):
            return 245878

    state = find_resolution_ranges(
        graph=_Graph(),
        cluster_range=np.asarray([14, 20], dtype=int),
        start_g=np.log(1.0),
        end_g=np.log(100.0),
        objective_function="CPM",
        resolution_tolerance=1e-6,
        n_workers=40,
        verbose=False,
        seed=123,
        snn_graph=None,
        min_cluster_size=3,
    )
    assert captured_gamma_batches["coarse"]
    assert float(np.max(captured_gamma_batches["coarse"][0])) <= 10.0 + 1e-12
    assert "_attrs" in state
    assert float(state["_attrs"]["coarse_upper_gamma"]) == pytest.approx(10.0)


def test_resolve_nested_worker_layout_biases_large_graphs_toward_outer_parallelism():
    runtime_context = RuntimeContext(
        spill_threshold_bytes=float(1e9),
        memory_budget_bytes=float(512 * 1024**3),
        scratch_root="/tmp",
        runtime_dir="/tmp",
    )
    layout = resolve_nested_worker_layout(
        total_workers=40,
        task_count=19,
        n_cells=245878,
        n_trials=4,
        n_bootstrap=20,
        runtime_context=runtime_context,
        outer_workers=None,
        inner_workers=None,
        expected_gamma_count=11,
    )
    assert int(layout["outer_workers"]) == 19
    assert int(layout["inner_workers"]) == 1
    assert int(layout["unused_worker_capacity"]) == 21


def test_build_target_worker_budgets_frontloads_extra_workers_to_expensive_targets():
    class _Graph:
        def vcount(self):
            return 245878

    scheduled_clusters = list(range(20, 1, -1))
    state = {
        "graph": _Graph(),
        "n_trials": 4,
        "n_bootstrap": 20,
        "n_workers": 1,
        "total_workers_requested": 40,
    }
    budgets = _build_target_worker_budgets(
        scheduled_clusters=scheduled_clusters,
        state=state,
        active_workers=19,
    )
    assert sum(int(budgets[k]) for k in scheduled_clusters[:19]) == 40
    assert max(int(budgets[k]) for k in scheduled_clusters[:19]) == 3
    assert min(int(budgets[k]) for k in scheduled_clusters[:19]) == 2


def test_should_use_global_phase1_process_pool_prefers_large_graph_multi_target_runs():
    class _Graph:
        def vcount(self):
            return 245878

    state = {
        "graph": _Graph(),
        "n_trials": 4,
        "n_workers": 1,
        "total_workers_requested": 40,
    }
    assert _should_use_global_phase1_process_pool([14, 15, 16], state, active_workers=3) is True

    state["total_workers_requested"] = 3
    assert _should_use_global_phase1_process_pool([14, 15, 16], state, active_workers=3) is False


def test_map_optimized_targets_wires_precomputed_phase1_by_target(monkeypatch):
    class _Graph:
        def vcount(self):
            return 245878

    seen: dict[int, dict[str, object] | None] = {}

    monkeypatch.setattr("scICEpy.api._should_use_global_phase1_process_pool", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "scICEpy.api._build_global_phase1_precomputed",
        lambda scheduled_clusters, state: {
            int(cluster_num): {"marker": f"k{int(cluster_num)}"}
            for cluster_num in scheduled_clusters
        },
    )

    def fake_optimize_target(cluster_num: int, state: dict[str, object]) -> dict[str, object]:
        seen[int(cluster_num)] = state.get("precomputed_phase1_by_target", {}).get(int(cluster_num))
        return {
            "cluster_number": int(cluster_num),
            "source_target_cluster": int(cluster_num),
        }

    monkeypatch.setattr("scICEpy.api._optimize_target_cluster_impl", fake_optimize_target)
    results = _map_optimized_targets(
        valid_clusters=[2, 3],
        state={
            "graph": _Graph(),
            "gamma_dict": {2: (0.1, 0.2), 3: (0.2, 0.3)},
            "target_interval_details": {},
            "n_trials": 4,
            "n_bootstrap": 20,
            "n_workers": 1,
            "total_workers_requested": 40,
            "verbose": False,
        },
        active_workers=1,
    )
    assert [int(item["cluster_number"]) for item in results] == [2, 3]
    assert seen == {2: {"marker": "k2"}, 3: {"marker": "k3"}}


def test_optimize_clustering_can_reuse_precomputed_phase1_without_local_gamma_threads(monkeypatch):
    class _Graph:
        def vcount(self):
            return 4

    labels = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    phase1_result = {
        "gamma": 0.15,
        "ic": 1.0,
        "matrix_ref": {"type": "memory", "matrix": labels.copy()},
        "mean_clusters": 2.0,
        "median_effective_clusters": 2.0,
        "effective_cluster_median": 2.0,
        "raw_cluster_median": 2.0,
        "final_cluster_median": 2.0,
        "median_gap": 0.0,
        "raw_median_gap": 0.0,
        "within_median_window": True,
        "strict_valid": True,
        "relaxed_valid": True,
        "raw_strict_valid": True,
        "raw_relaxed_valid": True,
        "hit_count": 2,
        "raw_hit_count": 2,
        "raw_guard_soft": True,
        "raw_guard_hard": True,
        "effective_hit_count": 2,
        "hit_trials": [0, 1],
        "_gamma_batch": "Primary Phase 1",
        "_phase_name": "phase1_primary",
    }

    def fail_evaluate_gamma(*args, **kwargs):
        raise AssertionError("local phase1 gamma evaluation should be skipped when precomputed_phase1 is provided")

    monkeypatch.setattr("scICEpy.optimization._evaluate_gamma", fail_evaluate_gamma)
    result = optimize_clustering(
        graph=_Graph(),
        target_clusters=2,
        gamma_range=(0.1, 0.2),
        objective_function="CPM",
        n_trials=2,
        n_bootstrap=1,
        seed=123,
        beta=0.1,
        n_iterations=10,
        max_iterations=10,
        resolution_tolerance=1e-8,
        n_workers=2,
        snn_graph=None,
        gamma_seed_values=None,
        min_cluster_size=1,
        verbose=False,
        precomputed_phase1={
            "primary_gamma_sequence": np.asarray([0.15], dtype=float),
            "secondary_gamma_sequence": np.asarray([], dtype=float),
            "gamma_seed_table": None,
            "phase1_expected_runs": 2,
            "primary_phase1": {
                "results": [phase1_result],
                "elapsed_sec": 1.25,
                "gamma_count": 1,
                "leiden_runs": 2,
                "nested_workers": 1,
            },
            "secondary_phase1": {
                "results": [],
                "elapsed_sec": 0.0,
                "gamma_count": 0,
                "leiden_runs": 0,
                "nested_workers": 1,
            },
            "secondary_phase1_used": False,
            "phase1_pool_workers": 8,
        },
    )
    assert bool(result["success"]) is True
    assert float(result["gamma"]) == pytest.approx(0.15)
    assert int(result["phase1_primary_gamma_count"]) == 1
    assert int(result["phase1_secondary_gamma_count"]) == 0
    diagnostics = result["optimization_diagnostics"]
    assert "phase1_primary" in diagnostics["phase"].values
