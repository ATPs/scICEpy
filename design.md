# scICEpy Design Document

## 1. Scope

This document describes the current implementation of `scICE_clustering()` in
**scICEpy**, the Python/AnnData port of the updated **scICER** workflow.

It is intended to answer four questions:

- what the public AnnData-facing API does,
- how cluster-range mode and manual-resolution mode run end-to-end,
- where each major algorithm stage lives in the Python codebase,
- which parts are already aligned to current scICER semantics and which parts
  are still known parity caveats.

This description matches the code under `scICEpy/scICEpy/` on 2026-03-23 and is
written against the updated scICER design that includes:

- raw-cluster-aware gamma admission,
- final-merged-cluster-keyed result semantics,
- shared resolution sweep diagnostics,
- manual `resolution` mode deduplication,
- `target_diagnostics` / `resolution_search_diagnostics` style reporting.

The document describes the Python implementation as it exists today. It does
not assume perfect parity where the code still differs from scICER.

## 1.1 Current Source Layout

The implementation is split across focused modules:

- `scICEpy/api.py`: public entry point, input validation, graph extraction,
  top-level mode selection, and orchestration.
- `scICEpy/resolution_search.py`: shared gamma sweep, preliminary probe
  evaluation, count stabilization, interval derivation, and search diagnostics.
- `scICEpy/optimization.py`: per-target optimization, gamma batching and
  admission, Phase 4 iterative refinement, Phase 5 bootstrap finalization, and
  small-cluster merge.
- `scICEpy/results.py`: result assembly, final-count rekeying,
  `target_diagnostics`, and summary field attachment.
- `scICEpy/metrics.py`: ECS, IC, MEI, and representative clustering selection.
- `scICEpy/leiden_wrapper.py`: sparse adjacency to `python-igraph` conversion,
  low-level Leiden execution, and simple clustering cache.
- `scICEpy/runtime.py`: worker budgeting, thread/process helpers, heartbeat
  logging, and optional matrix spill-to-disk.
- `scICEpy/visualization.py`: `plot_ic()` and `get_robust_labels()`.
- `scripts/qs_to_h5ad.R`: Seurat `.qs` to `.h5ad` conversion, including graph
  aliasing for Python benchmarks.

## 1.2 AnnData / Conversion Model

scICEpy assumes an AnnData-like object with:

- graph(s) in `adata.obsp`,
- cell names in `adata.obs_names`,
- result storage in `adata.uns`.

For Seurat-to-AnnData parity benchmarking, `scripts/qs_to_h5ad.R` converts a
Seurat object into `.h5ad` and copies the chosen Seurat graph to
`adata.obsp["connectivities"]`. It also preserves conversion metadata in
`adata.uns["scICEpy_source"]`.

This lets Python benchmarks use either:

- the original graph key such as `RNA_snn`, or
- the canonical alias `connectivities`.

## 1.3 Current Parity Caveats

Three caveats matter when comparing scICEpy to scICER:

- The public `beta` parameter is carried through the Python API, logging, and
  cache keys, but the current low-level `leidenalg.Optimiser()` call path does
  not apply an explicit beta term. The implementation now reports this in
  result metadata as `beta_supported = FALSE`, `beta_applied = FALSE`, and
  `beta_support_reason`.
- The Python result object stores `graph_key` in `adata.uns["scICE"]`, while
  older docs and R outputs often refer to `graph_name`.
- Shared interval derivation in `resolution_search.py` is driven primarily by
  stabilized final merged counts plus exact/near probe seeds. The module also
  contains raw-plateau helpers, but those helpers are not yet the main interval
  selector the way current scICER design emphasizes raw-cluster-aware search.

## 2. Quick Mental Model

`scICE_clustering()` is a multi-stage pipeline:

1. Validate inputs and normalize either `cluster_range` or `resolution`.
2. Resolve worker layout and create a runtime context.
3. Extract a sparse graph from `adata.obsp[graph_key]`.
4. Convert the graph to `python-igraph`.
5. Choose one of two entry modes:
   - `cluster_range` mode:
     run one shared gamma sweep, derive per-target gamma intervals, optionally
     filter unstable targets, then optimize each target.
   - `resolution` mode:
     skip shared search and evaluate each supplied gamma directly.
6. For each kept target or manual gamma:
   - run repeated Leiden trials,
   - measure effective/raw/final cluster medians,
   - compute IC on admitted gamma values,
   - optionally refine with more iterations,
   - bootstrap IC,
   - compute MEI,
   - choose one representative `best_labels`,
   - apply the final small-cluster merge once to `best_labels`.
7. Build `adata.uns["scICE"]`, re-keying cluster-range results by the true final
   merged cluster count.

The key semantic point is the same as updated scICER:

- `requested target cluster` means the target searched during optimization.
- `final merged cluster count` means the number of clusters in returned
  `best_labels` after the final small-cluster merge.

The public main result is keyed by the second quantity, not the first.

## 2.1 Final-Count-Keyed Results

In cluster-range mode:

- `n_cluster` is the returned final merged cluster count.
- `source_target_cluster` records which requested target produced that result.
- `coverage_complete` is `False` if some requested targets do not appear in the
  final public result after rekeying and deduplication.
- `search_coverage_complete` is a stricter property of the shared gamma search
  itself: it reports whether the search found optimization-ready intervals for
  all requested targets before optimization and rekeying.

This is why `search_coverage_complete = TRUE` and `coverage_complete = FALSE`
can happen at the same time. A requested target can optimize successfully and
still disappear from the public result if its final merged labels collapse onto
another returned final cluster number with lower IC.

## 3. Public API

### 3.1 `scICE_clustering()`

```python
scICE_clustering(
    adata,
    graph_key: str = "connectivities",
    cluster_range=None,
    n_workers: int = 10,
    outer_workers: int | None = None,
    inner_workers: int | None = None,
    n_trials: int = 15,
    n_bootstrap: int = 100,
    seed: int | None = None,
    beta: float = 0.1,
    n_iterations: int = 10,
    max_iterations: int = 150,
    ic_threshold: float = np.inf,
    objective_function: str = "CPM",
    remove_threshold: float = 1.15,
    min_cluster_size: int = 2,
    resolution_tolerance: float = 1e-8,
    verbose: bool = True,
    resolution=None,
    copy: bool = False,
)
```

Parameter summary:

- `adata`: AnnData-like object. Must expose `.obsp`, `.obs_names`, and `.uns`.
- `graph_key`: graph to cluster from `adata.obsp`.
- `cluster_range`: requested final merged cluster targets in cluster-range mode.
- `resolution`: manual gamma values. When present, cluster-range search is
  skipped and duplicate gamma values are removed before evaluation.
- `n_workers`: requested parallel worker budget. On Linux this drives outer
  multiprocessing plus inner thread pools; on Windows the outer process pools
  are disabled.
- `outer_workers`: optional explicit cap for target-level or resolution-level
  multiprocessing.
- `inner_workers`: optional explicit cap for per-target or per-resolution trial
  work.
- `n_trials`: repeated Leiden runs per gamma.
- `n_bootstrap`: bootstrap repetitions for final IC estimation.
- `seed`: base seed for deterministic per-gamma / per-trial seed derivation.
- `beta`: retained for API parity and logging. Current wrapper caveat noted in
  Section 1.3 applies.
- `n_iterations`: Leiden iterations used in Phase 1.
- `max_iterations`: upper bound for Phase 4 refinement.
- `ic_threshold`: used only when attaching summary fields such as
  `consistent_clusters`.
- `objective_function`: `"CPM"` or `"modularity"`.
- `remove_threshold`: optional pre-optimization filter. `Inf` skips filtering.
- `min_cluster_size`: controls effective-cluster counting and the final merge
  applied to `best_labels`.
- `resolution_tolerance`: search tolerance used in cluster-range mode.
- `verbose`: enables detailed logger output and heartbeat messages.
- `copy`: when `True`, return a modified copy of AnnData; otherwise return
  `None` and write results in place.

### 3.2 `plot_ic()` and `get_robust_labels()`

`visualization.py` exposes two AnnData-facing helpers:

- `plot_ic()` consumes either an AnnData object with `adata.uns["scICE"]` or a
  raw result dictionary.
- `get_robust_labels()` extracts returned `best_labels` into a DataFrame or
  annotates an AnnData copy.

The plotting logic follows the updated R behavior by supporting
two-line x-axis labels with gamma values when `show_gamma=True`.

## 4. Data and Result Model

## 4.1 Graph Extraction

`api._extract_graph()`:

- reads `adata.obsp[graph_key]`,
- logs graph shape, sparsity, and weight range,
- converts the sparse adjacency matrix into an undirected `igraph.Graph`.

`leiden_wrapper.graph_to_igraph()` keeps only the strict upper triangle
(`row < col`) so each undirected edge is represented once.

## 4.2 Result Object

The main result is stored in `adata.uns["scICE"]` and is a nested Python
dictionary containing:

- core arrays:
  `gamma`, `ic`, `ic_vec`, `n_cluster`, `best_labels`,
  `effective_cluster_median`, `raw_cluster_median`,
  `final_cluster_median`, `admission_mode`,
  `best_labels_raw_cluster_count`, `best_labels_final_cluster_count`,
  `source_target_cluster`, `result_status`.
- diagnostics:
  `target_diagnostics`, `resolution_search_diagnostics`,
  `resolution_diagnostics`.
- summary fields:
  `best_cluster`, `best_resolution`, `consistent_clusters`,
  `coverage_complete`, `search_coverage_complete`,
  `uncovered_targets`, `search_uncovered_targets`.
- metadata:
  `analysis_mode`, `resolution_input`, `graph_key`, `graph_name`, `beta`,
  `beta_supported`, `beta_applied`, `beta_support_reason`,
  `parallel_layout`, `min_cluster_size`, `cell_names`, `cluster_range_tested`.

## 4.3 Effective, Raw, and Final Counts

When `min_cluster_size > 1`:

- effective cluster count:
  number of clusters with size `>= min_cluster_size`.
- raw cluster count:
  number of unique labels before the final merge.
- final cluster count:
  number of clusters after applying the small-cluster merge to `best_labels`.

Resolution search records all three quantities for each probe. Optimization also
records all three medians for each gamma candidate.

## 5. Cluster-Range Workflow

## 5.1 Top-Level Orchestration

`api._run_cluster_range_mode()`:

- chooses shared search bounds,
- calls `find_resolution_ranges()`,
- optionally filters requested targets with `_filter_cluster_targets()`,
- allocates worker budgets for target optimization,
- calls `optimize_clustering()` per retained target,
- finalizes everything through `results.finalize_cluster_range_results()`.

## 5.2 Shared Resolution Search

`resolution_search.find_resolution_ranges()` implements the shared sweep.

Important details:

- For CPM, visible search bounds are `exp(start_g)` to `exp(end_g)`, with
  `start_g = max(log(resolution_tolerance), -20)` and `end_g = 20`.
- For large graphs, search uses fewer preliminary settings:
  `n_preliminary_trials = 3` and `n_iter_preliminary = 3` for
  `n_vertices >= 200000`.
- Probe worker count is memory-capped with `cap_workers_by_memory()`.
- The coarse probe count is `min(max(3 * active_probe_workers, 12), 30)`.

The search itself has four stages:

1. CPM upper-cap discovery:
   probe adaptive geometric batches until the requested maximum is covered, the
   high-gamma region becomes degenerate, or the hard cap is reached.
2. Coarse sweep:
   probe the full interval with a shared gamma grid.
3. Refinement rounds:
   add internal points inside unresolved intervals, with up to 8 points per
   interval when worker budget exceeds the number of unresolved intervals.
4. Interval derivation:
   derive per-target intervals and seed tables from the stabilized probe table.

Each probe currently runs one representative clustering and records:

- `effective_cluster_count`,
- `raw_cluster_count`,
- `final_cluster_count`,
- probe timing and pid,
- coarse / discovery / refinement metadata.

The result carries a full `resolution_search_diagnostics` DataFrame.

## 5.3 Interval Derivation

`derive_shared_gamma_intervals()`:

- sorts probes by gamma,
- stabilizes final and raw cluster-count curves using a monotone cumulative max,
- marks exact final-count hits,
- otherwise falls back to final-count brackets or near-target regions,
- stores per-target seed gamma values and interval details for optimization.

This is already structurally close to scICER, but Section 1.3 caveat still
applies: the Python implementation currently makes final-count-driven decisions
first, whereas current scICER design puts more emphasis on raw-count-aware
interval selection.

## 5.4 Optional Filtering

`_filter_cluster_targets()` is skipped when `remove_threshold = Inf`.

When enabled, it:

- samples 5 gamma values inside each target interval,
- runs 10 short Leiden trials per gamma (`n_iterations = 5`, `beta = 0.01`),
- computes IC,
- excludes targets whose best sampled IC is still above `remove_threshold`.

## 5.5 Per-Target Optimization

`optimization.optimize_clustering()` is the main per-target engine.

Phase 1:

- Build gamma batches from the search interval.
- Current Python defaults use a primary budget of 8 gamma values and a
  secondary budget of 4 gamma values.
- Seed values come from:
  interval endpoints, selected midpoint seed, exact probe values, near probe
  values, and generic search seeds.
- `_evaluate_gamma()` runs repeated Leiden trials for each gamma and records:
  effective/raw/final medians, hit counts, guard flags, and IC.

Gamma admission:

- `select_gamma_admission()` uses the ordered ladder:
  `raw_strict_soft -> strict_soft -> relaxed_soft -> strict_hard ->
  relaxed_hard -> relaxed_unguarded -> raw_relaxed_soft ->
  raw_relaxed_hard -> raw_relaxed_unguarded`.
- `refine_gamma_candidates_by_raw_gap()` breaks ties by minimum raw-median gap
  when `min_cluster_size > 1`.

Phase 4:

- If multiple viable gamma values remain and the best IC is not already good
  enough, the code re-runs Leiden with extra iterations and seeded initial
  memberships.
- The retained gamma set is progressively pruned by IC and stability.

Phase 5:

- Bootstrap IC is computed from the selected candidate matrix.
- `calculate_mei_from_array()` computes per-cell stability.
- `get_best_clustering()` chooses the representative clustering.
- If `preferred_trial_indices` exist, exact final-hit trials are preferred
  during representative-label selection.
- `merge_small_clusters_to_neighbors()` is applied once to `best_labels`.

## 5.6 Result Finalization

`results.finalize_cluster_range_results()`:

- rekeys successful target results by final merged cluster count,
- keeps only the lowest-IC result for each returned final cluster number,
- preserves one row per requested target in `target_diagnostics`,
- concatenates per-target `optimization_diagnostics`,
- computes `coverage_complete`,
- keeps the shared search diagnostics attached to the final object.

## 6. Manual Resolution Workflow

When `resolution` is supplied:

- input gamma values are normalized and de-duplicated,
- cluster-range search is skipped entirely,
- each remaining gamma is evaluated through
  `optimization.evaluate_fixed_resolution()`,
- the public main result is deduplicated by final cluster count, keeping the
  lowest-IC gamma for each final cluster number,
- all per-gamma rows remain available in `resolution_diagnostics`.

This matches the updated scICER semantics:

- manual resolution mode is fixed-gamma evaluation,
- it is not a one-row-per-input-gamma replay of cluster-range mode.

## 7. Parallelism, Memory, and Spill

## 7.1 Parallel Layout

scICEpy uses two layers of parallelism:

- outer multiprocessing on Unix for:
  search probes, target optimization, and manual resolution evaluation.
- inner thread pools for:
  repeated Leiden trials or gamma batches inside a target.

`resolve_effective_workers()` caps the requested worker count to
`os.cpu_count() - 1` on Unix.

## 7.2 Memory Budgeting

`runtime.cap_workers_by_memory()` uses:

- `estimate_trial_matrix_bytes()`,
- a runtime memory budget derived from `/proc/meminfo` or
  `SCICEPY_MEMORY_BUDGET_BYTES`.

The intent is not exact peak-RSS prediction; it is a practical upper bound for
worker allocation based on matrix footprints.

## 7.3 Spill Mode

Cluster matrices are stored through `store_cluster_matrix()`:

- in memory when small enough,
- as `.npy` spill files when estimated size exceeds the spill threshold.

Defaults:

- spill threshold: 2 GiB,
- spill directory: temporary directory created on demand,
- cleanup: automatic in the `finally` block at the end of `scICE_clustering()`.

## 8. Metrics Layer

`metrics.py` is currently a pure-Python implementation of:

- ECS,
- IC from extracted clustering arrays,
- representative-clustering selection,
- MEI.

This is faithful to the scICER concepts but is also one of the main runtime
hotspots for large real datasets, especially during Phase 5 bootstrap.

## 9. Practical Benchmarking Notes

For real-data parity testing against scICER:

- small dataset:
  `/XCLabServer004_fastIO/sim_Tcell_13.qs`
- large dataset:
  `/data1/xlab/researches/20250709_scICE/20260209_reviews/20260305_pancreatic/pancreatic_harmony1.qs`

Current recommended Python preparation flow:

1. convert Seurat `.qs` to `.h5ad` with `scripts/qs_to_h5ad.R`,
2. for very large objects, create a lightweight `.h5ad` that keeps one feature
   plus all relevant `obsp` graphs,
3. benchmark scICEpy against stored scICER outputs on the same graph and
   matching parameters.

## 10. Summary

The current scICEpy implementation already mirrors the modern scICER workflow
at a structural level:

- shared resolution search,
- raw/effective/final count tracking,
- final-count-keyed result semantics,
- manual-resolution deduplication,
- detailed diagnostics tables,
- outer-process plus inner-thread execution.

The remaining work is mostly about closing the last parity and performance gaps,
not redesigning the pipeline from scratch.
