# scICEpy

**Single-cell Inconsistency-based Clustering Evaluation for Python**

scICEpy is a Python implementation of the scICE algorithm for evaluating clustering consistency in single-cell RNA-seq data. It integrates seamlessly with the [scanpy](https://scanpy.readthedocs.io/) ecosystem and AnnData objects.

## Overview

scICE (Single-cell Inconsistency-based Clustering Evaluation) provides a systematic framework to:

- Evaluate clustering consistency across multiple resolutions
- Identify stable cluster numbers in your data
- Calculate Element-Centric Similarity (ECS) between clusterings
- Generate robust clustering labels with quantified uncertainty

## Installation

### From source

```bash
cd scICEpy
pip install -e .
```

### Dependencies

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scanpy >= 1.9.0
- anndata >= 0.8.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- python-igraph >= 0.10.0
- leidenalg >= 0.9.0
- tqdm >= 4.62.0

## Quick Start

```python
import scanpy as sc
import scICEpy

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Preprocess (if not already done)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata)

# Run scICE clustering evaluation
scICEpy.scICE_clustering(
    adata,
    cluster_range=list(range(2, 21)),  # Test 2-20 clusters
    n_trials=15,
    n_bootstrap=100,
    seed=42,
    verbose=True
)

# Plot results
fig, ax = scICEpy.plot_ic(adata, threshold=1.005)

# Extract consistent clustering labels
labels_df = scICEpy.get_robust_labels(adata, threshold=1.005)
```

## Key Features

### Element-Centric Similarity (ECS)

scICE uses a fast and efficient similarity metric to compare clustering results, approximately 150x faster than traditional methods like Adjusted Rand Index (ARI).

### Inconsistency Coefficient (IC)

The IC quantifies clustering stability:

- **IC < 1.005**: Highly consistent (<0.5% inconsistent cells)
- **IC 1.005-1.01**: Moderately consistent (0.5-1% inconsistent)
- **IC > 1.01**: Low consistency (>1% inconsistent)

### Three-Phase Algorithm

1. **Resolution Search**: Binary search to find resolution parameter ranges
2. **Optimization**: Iterative refinement using Leiden clustering
3. **Bootstrap Validation**: Stability assessment via bootstrap sampling

## API Reference

### Main Functions

#### `scICE_clustering()`

Main function to perform scICE clustering evaluation.

**Parameters:**
- `adata`: AnnData object with computed neighbor graph
- `cluster_range`: List of cluster numbers to test (default: 2-20)
- `n_trials`: Number of clustering trials per resolution (default: 15)
- `n_bootstrap`: Number of bootstrap iterations (default: 100)
- `seed`: Random seed for reproducibility (default: None)
- `objective_function`: "CPM" or "modularity" (default: "CPM")
- `ic_threshold`: IC threshold for consistency (default: Inf)
- `verbose`: Print progress messages (default: True)

**Returns:** Modifies `adata.uns['scICE']` in place

#### `get_robust_labels()`

Extract consistent clustering labels from scICE results.

**Parameters:**
- `adata`: AnnData object with scICE results
- `threshold`: IC threshold for consistency (default: 1.005)

**Returns:** DataFrame with cluster labels for each cell

#### `plot_ic()`

Plot IC scores across cluster numbers.

**Parameters:**
- `adata`: AnnData object with scICE results
- `threshold`: IC threshold line to plot (default: 1.005)
- `figsize`: Figure size (default: (8, 6))

**Returns:** matplotlib figure and axis

## Performance Tips

For large datasets (>50k cells):

- Reduce `n_trials` to 8-10
- Reduce `n_bootstrap` to 50
- Use a focused `cluster_range` based on biological expectations
- Set `verbose=True` to monitor progress

## Testing

Run the test script to verify installation:

```bash
python test_scICEpy.py
```

## Related Projects

- **scICER**: R package implementation - https://github.com/ATPs/scICER
- **scICE**: Original Julia implementation - https://github.com/Mathbiomed/scICE
- **scanpy**: Single-cell analysis in Python - https://scanpy.readthedocs.io/

## Citation

If you use scICEpy in your research, please cite the original scICE paper:

```
[Citation information to be added]
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions and support, please open an issue on the GitHub repository.
