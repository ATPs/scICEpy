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

### Prerequisites

- Python >= 3.8
- pip or conda package manager

### Option 1: Install from source (recommended for development)

```bash
# Navigate to the scICEpy directory
cd scICEpy

# Install in editable mode
pip install -e .
```

### Option 2: Install dependencies manually

If you encounter issues with the automatic installation, you can install dependencies manually:

```bash
# Core scientific computing
pip install numpy>=1.20.0 pandas>=1.3.0 scipy>=1.7.0

# Single-cell analysis
pip install scanpy>=1.9.0 anndata>=0.8.0

# Clustering
pip install python-igraph>=0.10.0 leidenalg>=0.9.0

# Utilities
pip install matplotlib>=3.4.0 tqdm>=4.62.0

# Then install scICEpy
pip install -e .
```

### Option 3: Using conda (alternative)

```bash
# Create a new conda environment
conda create -n scicepy python=3.10
conda activate scicepy

# Install dependencies via conda
conda install -c conda-forge scanpy python-igraph leidenalg

# Install remaining dependencies
pip install tqdm

# Install scICEpy
cd scICEpy
pip install -e .
```

### Verify Installation

Run the test script to verify everything is working:

```bash
python test_scICEpy.py
```

You should see output indicating successful imports and test completion.

### Common Installation Issues

#### Issue 1: igraph installation fails

**Solution:** Install system dependencies first

On Ubuntu/Debian:
```bash
sudo apt-get install build-essential python-dev libxml2 libxml2-dev zlib1g-dev
pip install python-igraph
```

On macOS:
```bash
brew install igraph
pip install python-igraph
```

#### Issue 2: leidenalg installation fails

**Solution:** Ensure igraph is installed first, then:

```bash
pip install leidenalg --no-cache-dir
```

#### Issue 3: scanpy dependencies

If scanpy installation is slow or fails:

```bash
# Use conda for scientific packages
conda install -c conda-forge scanpy
```

### Testing Your Installation

Quick test in Python:

```python
import scICEpy
print(scICEpy.__version__)  # Should print: 0.1.0
```

Full test with data:

```python
import scanpy as sc
import scICEpy

# Load test data
adata = sc.datasets.pbmc68k_reduced()

# Preprocess
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.neighbors(adata)

# Run scICE (small test)
scICEpy.scICE_clustering(adata, cluster_range=[2, 3, 4], n_trials=5, n_bootstrap=20)

# Check results
print(adata.uns['scICE']['n_cluster'])
print(adata.uns['scICE']['ic'])
```

### Development Installation

For development with additional tools:

```bash
pip install -e ".[dev]"
```

This installs additional packages:
- pytest (for testing)
- pytest-cov (for coverage)

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
- `n_workers`: Number of parallel workers (default: 10)
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

## Parallel Processing

scICEpy supports multi-core parallel processing to significantly speed up analysis:

```python
# Use 10 parallel workers (default)
scICEpy.scICE_clustering(
    adata,
    cluster_range=list(range(2, 21)),
    n_workers=10,  # Parallel processing with 10 cores
    seed=42
)

# Sequential processing (no parallelization)
scICEpy.scICE_clustering(
    adata,
    cluster_range=list(range(2, 21)),
    n_workers=1,  # Single-threaded
    seed=42
)
```

### Setting n_workers

- **Default**: 10 workers provides good balance for most analyses
- **Recommended**: Set to the number of physical CPU cores available
- **Large cluster ranges**: For testing N cluster numbers, use `n_workers=N` for best efficiency
- **Memory constraints**: Reduce `n_workers` if you encounter memory issues
- **Small datasets**: For testing only 2-3 cluster numbers, use `n_workers=1` to avoid overhead

### Performance Improvements

Parallel processing provides speedups in three areas:
1. **Resolution search**: Processes different cluster numbers simultaneously
2. **Clustering optimization**: Optimizes multiple cluster numbers in parallel
3. **ECS calculations**: Parallelizes similarity computations

Expected speedup depends on your `cluster_range` size and available CPU cores.

## Performance Tips

For large datasets (>50k cells):

- Reduce `n_trials` to 8-10
- Reduce `n_bootstrap` to 50
- Use a focused `cluster_range` based on biological expectations
- Adjust `n_workers` based on available CPU cores and memory
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
