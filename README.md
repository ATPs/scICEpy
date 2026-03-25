# scICEpy

**Single-cell Inconsistency-based Clustering Evaluation for Python**

scICEpy evaluates clustering stability on precomputed single-cell graphs stored
in AnnData objects. It is designed for Scanpy-style workflows and writes its
results back to `adata.uns["scICE"]`.

## Overview

scICEpy currently supports two analysis modes:

- **Cluster-range mode**: run one shared resolution search, derive per-target
  gamma intervals, then optimize each requested target cluster.
- **Manual resolution mode**: skip shared search and evaluate the supplied
  gamma values directly. Repeated gamma values are deduplicated before
  evaluation.

Important current result semantics:

- Returned public results are keyed by the **final merged cluster count** in
  `best_labels`.
- In cluster-range mode, `source_target_cluster` records which requested target
  produced each returned final cluster result.
- `min_cluster_size` affects both effective-cluster counting during search and
  optimization, and the final merge applied to `best_labels`.
- The public `beta` parameter is retained for API compatibility and metadata,
  but the current Python backend reports `beta_supported = False` and
  `beta_applied = False`.

## Installation

### Install from source

```bash
cd scICEpy
pip install -e .
```

### Development install

```bash
cd scICEpy
pip install -e ".[dev]"
```

### Core dependencies

- Python >= 3.8
- numpy
- pandas
- scipy
- scanpy
- anndata
- matplotlib
- python-igraph
- leidenalg

## Verification

Quick import check:

```bash
python - <<'PY'
import scICEpy
print(scICEpy.__version__)
print(scICEpy.scICE_clustering)
PY
```

Automated tests:

```bash
python -m pytest -q
python test_scICEpy.py
```

Development note:

- The implementation logic lives in `clustering_dispatch.py`,
  `clustering_inputs.py`, `clustering_modes.py`, `clustering_reporting.py`,
  `gamma_candidates.py`, `gamma_execution.py`, `target_optimizer.py`,
  `cluster_utils.py`, and `scICEpy.py`.
- The split helper modules now use explicit one-way imports rather than
  circular `import *` chains.

## Quick Start

```python
import scanpy as sc
import scICEpy

adata = sc.read_h5ad("your_data.h5ad")

# If the graph is not already present, compute neighbors first.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.pca(adata)
sc.pp.neighbors(adata)

scICEpy.scICE_clustering(
    adata,
    graph_key="connectivities",
    cluster_range=list(range(2, 21)),
    n_trials=15,
    n_bootstrap=100,
    seed=42,
    verbose=True,
)

results = adata.uns["scICE"]
print(results["analysis_mode"])
print(results["n_cluster"])
print(results["ic"])
print(results["best_cluster"], results["best_resolution"])
```

Manual resolution mode:

```python
scICEpy.scICE_clustering(
    adata,
    resolution=[0.05, 0.05, 0.10, 0.20],
    n_trials=15,
    n_bootstrap=100,
    seed=42,
    verbose=True,
)

results = adata.uns["scICE"]
print(results["analysis_mode"])      # "resolution"
print(results["resolution_input"])   # deduplicated gamma values
```

## Accessing Results

After `scICE_clustering()` finishes, scICEpy stores a nested result dictionary
in `adata.uns["scICE"]`.

```python
results = adata.uns["scICE"]

print(results["analysis_mode"])
print(results["n_cluster"])              # returned final merged cluster counts
print(results["source_target_cluster"])  # originating requested targets
print(results["gamma"])
print(results["ic"])
print(results["consistent_clusters"])
print(results["best_cluster"], results["best_resolution"])
print(results["coverage_complete"], results["search_coverage_complete"])
print(results["parallel_layout"])
```

Key fields to know:

- `analysis_mode`: `"cluster_range"` or `"resolution"`.
- `n_cluster`: returned final merged cluster counts in the public result.
- `source_target_cluster`: requested target that produced each returned result
  in cluster-range mode.
- `gamma`, `ic`, `ic_vec`, `best_labels`: main per-result outputs.
- `consistent_clusters`, `best_cluster`, `best_resolution`: summary fields
  derived from `ic_threshold`.
- `coverage_complete`: whether every requested target survives the final
  public result after optimization and final-count rekeying.
- `search_coverage_complete`: whether shared search found optimization-ready
  intervals for all requested targets before optimization.
- `target_diagnostics`, `resolution_search_diagnostics`,
  `optimization_diagnostics`, `resolution_diagnostics`: pandas DataFrames with
  search, optimization, and manual-resolution diagnostics.
- `parallel_layout`: resolved outer/inner worker layout used for the run.
  Large graphs now bias Phase 1 toward shared process-pool execution and often
  resolve to `inner_workers = 1`.
- `min_cluster_size`: value used for effective counting and final label merge.
- `beta_supported`, `beta_applied`, `beta_support_reason`: backend capability
  metadata for the `beta` parameter.
- `cluster_range_tested`: summary array retained in the public result; it
  currently mirrors the returned `n_cluster` values.

## Labels and Plotting

Extract returned labels as a DataFrame:

```python
labels_df = scICEpy.get_robust_labels(adata, threshold=1.005)
print(labels_df.head())
```

Or add them directly to `adata.obs`:

```python
adata = scICEpy.get_robust_labels(adata, threshold=1.005, return_adata=True)
```

Plot the IC distributions:

```python
fig, ax = scICEpy.plot_ic(adata, threshold=1.005, show_gamma=True)
```

`plot_ic()` and `get_robust_labels()` can also consume a raw result dictionary
instead of an AnnData object, as long as cell names are available when labels
need to be returned.

## API Summary

Main entry point:

```python
scICEpy.scICE_clustering(
    adata,
    graph_key="connectivities",
    cluster_range=None,
    n_workers=10,
    outer_workers=None,
    inner_workers=None,
    n_trials=15,
    n_bootstrap=100,
    seed=None,
    beta=0.1,
    n_iterations=10,
    max_iterations=150,
    ic_threshold=float("inf"),
    objective_function="CPM",
    remove_threshold=1.15,
    min_cluster_size=2,
    resolution_tolerance=1e-8,
    verbose=True,
    resolution=None,
    copy=False,
    scratch_dir=None,
)
```

Behavior notes for the most important options:

- `graph_key`: which graph in `adata.obsp` to cluster.
- `cluster_range`: requested targets for cluster-range mode.
- `resolution`: manual gamma values. When set, shared search is skipped.
- `n_workers`: top-level worker budget. scICEpy resolves actual outer and inner
  worker layout from this budget.
- `outer_workers`, `inner_workers`: optional explicit caps for outer
  multiprocessing and inner thread work.
- `remove_threshold`: cluster-range pre-filter threshold. It is ignored in
  manual resolution mode.
- `min_cluster_size`: when greater than 1, scICEpy counts effective clusters
  during search and optimization and merges small clusters in final labels.
- `resolution_tolerance`: search tolerance used in cluster-range mode.
- `copy`: return a modified AnnData copy instead of writing in place.
- `scratch_dir`: optional runtime temp root for spill files and temporary
  working directories.
- `beta`: kept for compatibility and metadata, but not applied by the current
  Python backend.

Public helpers:

- `scICEpy.plot_ic(...)`
- `scICEpy.get_robust_labels(...)`

## Large H5AD Inputs

If your `.h5ad` file is dominated by expression matrices or layers that
scICEpy does not need for clustering, use the large-file wrapper to create a
lighter copy, run scICEpy on that copy, and write the results back into the
original `.h5ad`.

Helper scripts:

- `scripts/run_large_h5ad_scice.py`
- `scripts/make_light_h5ad.py`

The light-file step preserves:

- `obs`
- `obsm`
- `obsp`
- `uns`
- only the first `n_vars` feature columns from `X` and aligned per-variable
  metadata

Primary workflow:

```bash
python scripts/run_large_h5ad_scice.py \
  --input your_data.h5ad \
  --light-output your_data.light.h5ad \
  --n-vars 1 \
  --cluster-range 2 3 4 5 6 7 8 9 10 \
  --n-trials 15 \
  --n-bootstrap 100 \
  --seed 42
```

This workflow:

- creates `your_data.light.h5ad`
- runs `scICEpy.scICE_clustering(...)` on the light file
- writes only `uns["scICE"]` back to `your_data.h5ad`
- keeps the light file on disk for reuse or inspection

For H5AD persistence, variable-length result sequences such as bootstrap IC
vectors and stored label collections are written through an internal
H5AD-safe encoding. `plot_ic()` and `get_robust_labels()` read that encoding
transparently after reload.

Use `--resolution 0.05 0.10 0.20` instead of `--cluster-range ...` when you
want manual resolution mode.

If you only want to generate the lightweight file, `scripts/make_light_h5ad.py`
still supports the original one-step conversion:

```bash
python scripts/make_light_h5ad.py \
  --input your_data.h5ad \
  --output your_data.light.h5ad \
  --n-vars 1
```

After the wrapper finishes, read the original file again and plot from the
results that were written back:

```python
import scanpy as sc
import scICEpy

adata = sc.read_h5ad("your_data.h5ad")

fig, ax = scICEpy.plot_ic(adata, threshold=1.005, show_gamma=True)

adata = scICEpy.get_robust_labels(adata, threshold=1.005, return_adata=True)

scice_columns = [column for column in adata.obs.columns if column.startswith("scICE_k_")]
if "X_umap" in adata.obsm and scice_columns:
    sc.pl.umap(adata, color=scice_columns[: min(3, len(scice_columns))], wspace=0.4)
```

## Parallelism and Performance

scICEpy uses:

- shared search plus per-target optimization in cluster-range mode
- outer multiprocessing on Unix where appropriate
- a shared Phase 1 process pool for large graphs (`n_cells >= 200000`)
- per-gamma trial summaries that reuse final-cluster counts and preferred-hit
  trial bookkeeping across optimization and finalization
- inner thread pools mainly for small/medium jobs and bootstrap/finalize work
- runtime temporary directories and optional spill-to-disk for large
  intermediate matrices

Result assembly keeps the public `adata.uns["scICE"]` contract unchanged while
using one shared final-cluster deduplication path for both cluster-range and
manual resolution outputs.

Practical tips:

- Start with a focused `cluster_range` instead of scanning more targets than
  you need.
- Reduce `n_trials` and `n_bootstrap` for large datasets when you need a
  faster exploratory run.
- Use `n_workers` as the top-level budget and only set `outer_workers` or
  `inner_workers` when you need tighter control.
- Set `scratch_dir` if you want runtime temporary files written somewhere
  specific.
- Keep `verbose=True` when tuning large jobs so you can inspect search,
  optimization, and final summary logs.

## Testing

From the repository root:

```bash
pip install -e ".[dev]"
python -m pytest -q
python test_scICEpy.py
```

If you launch Python from the repository parent directory, `import scICEpy`
now resolves through a repository-root shim and still exposes the packaged
API (`scICE_clustering`, `plot_ic`, `get_robust_labels`, and helper submodules
such as `scICEpy.target_optimizer`).
