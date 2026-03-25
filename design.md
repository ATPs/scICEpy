# scICEpy Design Document

## 1. Scope

This document describes the current implementation of `scICE_clustering()` in
**scICEpy**, the Python/AnnData port of the updated **scICER** workflow.
Practically, scICEpy should be read as an AnnData-native implementation of the
modern scICER semantics rather than as a line-by-line port of the original
Julia `scICE` code.

It is intended to answer five questions:

- what the public AnnData-facing API does,
- how cluster-range mode and manual-resolution mode run end-to-end,
- where each major algorithm stage lives in the Python codebase,
- how scICEpy compares to current scICER and to the original Julia scICE,
- which parts are already aligned to current scICER semantics and which
  practical differences still remain.

This description matches the code under `scICEpy/scICEpy/` on 2026-03-25 and is
written against the updated scICER design that includes:

- raw-cluster-aware gamma admission,
- final-merged-cluster-keyed result semantics,
- shared resolution sweep diagnostics,
- manual `resolution` mode deduplication,
- `target_diagnostics` / `resolution_search_diagnostics` style reporting.

The document describes the Python implementation as it exists today. The
workflow-level behavior is now intentionally aligned with scICER for the main
user-visible semantics, while a few backend and metadata differences still
remain.

For the Seurat/R counterpart, see [**scICER**](https://github.com/ATPs/scICER).

## 1.1 Current Source Layout

The implementation is split across focused modules:

- `scICEpy/scICEpy.py`: concrete `scICE_clustering()` entry implementation.
- `scICEpy/clustering_inputs.py`: input validation, cluster-range
  normalization, resolution normalization, graph extraction, and compact
  cluster-value formatting helpers.
- `scICEpy/clustering_reporting.py`: final result summary logging for the
  public clustering entry point.
- `scICEpy/clustering_dispatch.py`: target filtering, manual-resolution
  dispatch, target optimization dispatch, per-target worker budgeting, and
  shared Phase 1 process-pool helpers.
- `scICEpy/clustering_modes.py`: cluster-range-mode and manual-resolution-mode
  orchestration.
- `scICEpy/cluster_utils.py`: low-level cluster count helpers, raw guard
  helpers, and the final small-cluster merge routine shared by search and
  optimization code.
- `scICEpy/resolution_search.py`: shared gamma sweep, preliminary probe
  evaluation, count stabilization, interval derivation, and search diagnostics.
- `scICEpy/search_bounds.py`: shared interval-bound, probe-plan, and
  target-interval derivation helpers used by resolution search.
- `scICEpy/gamma_candidates.py`: gamma batching, seed normalization,
  admission scoring, recovery-point generation, and candidate ordering helpers.
- `scICEpy/gamma_execution.py`: low-level gamma evaluation, trial-matrix
  summaries, diagnostics flattening, and final clustering selection.
- `scICEpy/target_optimizer.py`: per-target optimization, fixed-resolution
  evaluation, Phase 4 iterative refinement, and Phase 5 bootstrap finalization.
- `scICEpy/results.py`: result assembly, lightweight target-result helpers,
  final-count rekeying,
  `target_diagnostics`, and summary field attachment.
- `scICEpy/metrics.py`: ECS, IC, MEI, and representative clustering selection.
- `scICEpy/leiden_wrapper.py`: sparse adjacency to `python-igraph` conversion,
  low-level Leiden execution, and simple clustering cache.
- `scICEpy/runtime.py`: worker budgeting, shared process-pool helpers,
  heartbeat logging, and optional matrix spill-to-disk.
- `scICEpy/visualization.py`: `plot_ic()` and `get_robust_labels()`.
- `scICEpy/large_h5ad.py`: helpers for creating lightweight `.h5ad` copies,
  running scICEpy on them, and writing results back to the original file.
- repository-root `__init__.py`: import shim that forwards
  `import scICEpy` from the repository parent directory to the actual
  packaged implementation under `scICEpy/`.
- `tests/test_algorithm_helpers.py`, `tests/test_api.py`, and
  `tests/test_large_h5ad.py`: fast pytest coverage for helper behavior,
  public API contracts, and large-H5AD helper workflows.
- `tests/test_smoke.py`: slow end-to-end pytest smoke coverage for package
  import, Scanpy preprocessing, clustering, plotting, and manual-resolution
  execution.
- `scripts/qs_to_h5ad.R`: Seurat `.qs` to `.h5ad` conversion, including graph
  aliasing for Python benchmarks.
- `scripts/make_light_h5ad.py`: create a smaller `.h5ad` that preserves the
  graph and AnnData metadata needed for repeated scICEpy runs.
- `scripts/run_large_h5ad_scice.py`: convenience wrapper that creates a light
  `.h5ad`, runs `scICE_clustering()`, and writes `uns["scICE"]` back to the
  original input file.

The split modules intentionally use explicit one-way imports. The public entry
modules are allowed to depend on helper modules, but helper modules must not
depend back on the entry modules via wildcard imports because that leaves
runtime globals unbound during package initialization. `scICE_clustering` is
imported directly from `scICEpy.py` by the package root, the shared entry
helpers live in the `clustering_*` modules listed above, and the optimization
stack is split across `gamma_candidates.py`, `gamma_execution.py`, and
`target_optimizer.py`.

Pytest discovery is now anchored to `tests/` via `pyproject.toml`. Routine
validation runs through `python -m pytest -q -m "not slow"`, while the
end-to-end Scanpy smoke coverage runs separately through
`python -m pytest -q -m slow tests/test_smoke.py`.

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

For large H5AD inputs, `large_h5ad.py` implements a wrapper workflow:

- create a lightweight copy that keeps `obs`, `obsm`, `obsp`, `uns`, and only
  the first `n_vars` columns of matrix/variable-aligned content,
- run `scICE_clustering()` on that light copy,
- reopen the original input with `anndata.read_h5ad(..., backed="r+")`,
- verify that `obs_names` still match exactly,
- write only `adata.uns["scICE"]` back to the original file and persist it.

Before writing, large-H5AD helpers encode variable-length result sequences
such as stored label collections and bootstrap vectors into an H5AD-safe
nested mapping form. Public helper readers such as `plot_ic()` and
`get_robust_labels()` decode that form transparently after reload.

This wrapper does not auto-populate `adata.obs["scICE_k_*"]` in the original
file; label extraction remains an explicit post-processing step via
`get_robust_labels()`.

## 1.3 Current Alignment Status and Known Differences

At the algorithm/workflow level, scICEpy is already aligned with the current
scICER design on the behaviors that most affect user-visible results:

- shared coarse-to-refine resolution search,
- raw/effective/final cluster counting,
- raw-cluster-aware gamma admission and raw-gap tie-breaking,
- final-merged-cluster-keyed result semantics,
- manual `resolution` mode deduplication by final cluster number,
- detailed `target_diagnostics`, `resolution_search_diagnostics`,
  `optimization_diagnostics`, and `resolution_diagnostics`.

The remaining concrete differences are mostly backend- or ecosystem-level:

- The public `beta` parameter is carried through the Python API, logging, and
  cache keys, but the current low-level `leidenalg.Optimiser()` call path does
  not apply an explicit beta term. In practice, scICEpy can seed and optimize a
  prepared partition, but unlike scICER's `igraph::cluster_leiden(...)`
  wrapper it does not expose the same beta control on the execution path used
  here. The implementation reports this in result metadata as
  `beta_supported = FALSE`, `beta_applied = FALSE`, and `beta_support_reason`.
- scICEpy operates on AnnData and writes a nested result dictionary to
  `adata.uns["scICE"]`, while scICER operates on Seurat objects and returns an
  R `scICE` result object.
- Exact numeric IC/gamma outcomes can still differ from scICER and from the
  original Julia scICE because the execution backends differ
  (`python-igraph`/`leidenalg` plus pure-Python metrics in scICEpy, R
  `igraph` plus ClustAssess in scICER, and Julia + PyCall machinery in
  original scICE).
- `cluster_range_tested` currently mirrors the returned public `n_cluster`
  values. For authoritative requested/search targets, use
  `requested_cluster_range` and `searched_target_cluster_range`.

## 1.4 Positioning Relative to scICE and scICER

The shortest accurate positioning statement is:

- relative to **scICER**, scICEpy is a Python/AnnData port with mostly matched
  workflow semantics;
- relative to **scICE** (Julia), scICEpy intentionally inherits the newer
  scICER behavior rather than preserving the historical Julia behavior exactly.

This means that, when parity matters:

- compare scICEpy primarily against **scICER** for current algorithm semantics,
- compare scICEpy against **scICE** mainly to understand how the workflow has
  evolved since the original Julia implementation.

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

## 2.2 Resolution-Mode Summary

Manual `resolution` mode is intentionally a fixed-gamma evaluation path, not a
replay of the `cluster_range` optimizer.

When `resolution` is supplied, scICEpy:

1. normalizes the input gamma values and removes duplicates while preserving
   input order,
2. skips the shared gamma-search stage entirely,
3. evaluates each remaining gamma independently with repeated Leiden trials,
4. computes per-gamma Phase 1 IC and bootstrap IC summaries,
5. groups evaluated gamma values by `best_labels_final_cluster_count` and keeps
   only the lowest-IC gamma for each final cluster number in the public main
   result,
6. keeps the full per-gamma trace in `resolution_diagnostics`.

Two consequences matter for interpretation:

- `resolution = old_results["gamma"]` is not guaranteed to reproduce the
  public output of an earlier `cluster_range` run.
- `resolution` mode does not use the multi-gamma admission ladder or Phase 4
  iterative refinement used by target optimization.

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
    scratch_dir: str | None = None,
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
- `scratch_dir`: optional root directory for scICEpy runtime temporary files
  and spill storage.

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
  `optimization_diagnostics`,
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

This is intentionally aligned with the modern scICER search design: stabilized
final merged counts determine optimization readiness, while raw-count plateaus,
raw exact/near hits, and raw bracket annotations constrain and annotate the
candidate interval instead of being ignored.

## 5.4 Optional Filtering

`_filter_cluster_targets()` is skipped when `remove_threshold = Inf`.

When enabled, it:

- samples 5 gamma values inside each target interval,
- runs 10 short Leiden trials per gamma (`n_iterations = 5`, `beta = 0.01`),
- computes IC,
- excludes targets whose best sampled IC is still above `remove_threshold`.

## 5.5 Per-Target Optimization

`target_optimizer.optimize_clustering()` is the main per-target engine.

Phase 1:

- Build gamma batches from the search interval.
- Current Python defaults use a primary budget of 8 gamma values and a
  secondary budget of 4 gamma values.
- Seed values come from:
  interval endpoints, selected midpoint seed, exact probe values, near probe
  values, and generic search seeds.
- Large graphs (`graph.vcount() >= 200000`) now force Phase 1 onto a shared
  process pool across targets instead of relying on nested per-target thread
  pools.
- `_evaluate_gamma()` builds one `TrialMatrixSummary` per gamma and records:
  effective/raw/final medians, hit counts, guard flags, reusable
  `final_cluster_counts`, and IC.

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
- The extra-iteration matrices reuse the same `TrialMatrixSummary` path as
  Phase 1, so final-hit trial selection and final-count bookkeeping are
  computed once per matrix.
- The retained gamma set is progressively pruned by IC and stability.

Phase 5:

- Bootstrap IC is computed from the selected candidate matrix.
- `calculate_mei_from_array()` computes per-cell stability.
- `get_best_clustering()` chooses the representative clustering.
- If `preferred_trial_indices` exist, exact final-hit trials are preferred
  during representative-label selection.
- If `final_cluster_counts` are already available from earlier phases, final
  selection reuses them instead of recomputing full-trial final counts.
- `merge_small_clusters_to_neighbors()` is applied once to `best_labels`.

## 5.6 Result Finalization

`results.finalize_cluster_range_results()`:

- builds per-target rows with `build_target_result_record()` and lightweight
  dict helpers,
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
  `target_optimizer.evaluate_fixed_resolution()`,
- the public main result is deduplicated by final cluster count through the
  same final-cluster rekey helper used in cluster-range mode, keeping the
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
- a shared large-graph Phase 1 process pool for:
  per-gamma trial evaluation across multiple targets.
- inner thread pools for:
  small/medium fixed-resolution work, small/medium target-local Phase 1, and
  bootstrap/finalize steps.

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

## 10. Comparison to scICE and scICER

## 10.1 scICEpy vs scICER

For current behavior, scICEpy is much closer to scICER than to the original
Julia scICE. The important comparison points are:

| Aspect | scICEpy | scICER |
|--------|---------|--------|
| Host ecosystem | AnnData / Scanpy-style workflows | Seurat / R workflows |
| Entry modes | `cluster_range` plus manual `resolution` | `cluster_range` plus manual `resolution` |
| Search semantics | Shared upper-cap discovery + coarse sweep + refinement | Same overall structure |
| Optimization semantics | Raw/effective/final counts, gamma-admission ladder, final-count rekeying | Same overall structure |
| Result semantics | Main result keyed by final merged cluster count in `best_labels` | Same overall structure |
| Diagnostics | `target_diagnostics`, `resolution_search_diagnostics`, `optimization_diagnostics`, `resolution_diagnostics` | Same diagnostic families |
| Large-object workflow | Lightweight `.h5ad` helpers | Lightweight Seurat / `qs` workflow |
| Main remaining gap | `beta` retained but not applied by current Python backend; some metadata fields differ | `beta` applied by R backend |

In practical terms, a user who already understands modern scICER should read
scICEpy as the same clustering workflow expressed in Python and attached to
AnnData, not as a separate algorithm with different result semantics.

## 10.2 scICEpy vs scICE (Julia)

Relative to the original Julia implementation, scICEpy follows the newer
scICER-style workflow rather than the historical scICE behavior:

| Aspect | scICEpy | Original `scICE` |
|--------|---------|------------------|
| Input model | AnnData graph in `adata.obsp` | Julia dictionary / scLENS-centered pipeline |
| Search architecture | Shared global gamma sweep with refinement | Per-target binary search with repeated midpoint probing |
| Cluster-count model | Effective count + raw count + final merged count | Raw count only |
| Meaning of `cluster_range` | Requested final merged cluster counts | Requested raw cluster counts |
| Candidate admission | Multi-level admission ladder with raw-guarded fallback | Exact median-count equality filter |
| Final label handling | One final small-cluster merge on `best_labels`, then rekey by final cluster count | No final small-cluster merge |
| Output truthfulness | Returned `n_cluster` matches the final merged `best_labels` | Returned target/filter count can differ from raw final labels |
| Manual `resolution` mode | Supported | No dedicated public manual-resolution path |
| Diagnostics | Explicit search/optimization/target diagnostics tables | No comparable diagnostic tables |
| `beta` behavior | Exposed but not currently applied by the Python backend | Applied in Julia Leiden calls |

So the most important takeaway is not "scICEpy reproduces Julia scICE exactly".
The more accurate statement is:

- scICEpy preserves the **problem domain** of scICE,
- but adopts the **modern semantics and auditability model** established by
  scICER.

Users comparing outputs across generations should therefore expect the largest
differences in:

- target-coverage behavior when no gamma matches a target exactly,
- the meaning of returned cluster numbers,
- small-cluster handling in final labels,
- availability of manual-resolution diagnostics and per-target audit trails.

## 11. Summary

The current scICEpy implementation already mirrors the modern scICER workflow
at a structural level:

- shared resolution search,
- raw/effective/final count tracking,
- final-count-keyed result semantics,
- manual-resolution deduplication,
- detailed diagnostics tables,
- outer-process plus inner-thread execution.

Relative to scICER, the remaining gaps are mainly backend-specific rather than
algorithmic. Relative to the original Julia scICE, scICEpy should be understood
as an evolved implementation with modern scICER semantics, not as a literal
behavior-preserving port.
