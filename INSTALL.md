# Installation Guide for scICEpy

## Prerequisites

- Python >= 3.8
- pip or conda package manager

## Installation Steps

### Option 1: Install from source (recommended for development)

```bash
# Navigate to the scICEpy directory
cd /data/p/xiaolong/scICE_dev/scICEpy

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
cd /data/p/xiaolong/scICE_dev/scICEpy
pip install -e .
```

## Verify Installation

Run the test script to verify everything is working:

```bash
python test_scICEpy.py
```

You should see output like:

```
Testing scICEpy...
✓ scanpy imported successfully
✓ scICEpy imported successfully
✓ igraph imported successfully
✓ leidenalg imported successfully
...
✓ All tests completed successfully!
```

## Common Issues

### Issue 1: igraph installation fails

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

### Issue 2: leidenalg installation fails

**Solution:** Ensure igraph is installed first, then:

```bash
pip install leidenalg --no-cache-dir
```

### Issue 3: scanpy dependencies

If scanpy installation is slow or fails:

```bash
# Use conda for scientific packages
conda install -c conda-forge scanpy
```

## Testing Your Installation

### Quick test in Python:

```python
import scICEpy
print(scICEpy.__version__)  # Should print: 0.1.0
```

### Full test with data:

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

## Development Installation

For development with additional tools:

```bash
pip install -e ".[dev]"
```

This installs additional packages:
- pytest (for testing)
- pytest-cov (for coverage)

## Uninstallation

```bash
pip uninstall scICEpy
```

## Next Steps

After successful installation, see:
- [README.md](README.md) for usage examples
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for technical details
- Run `test_scICEpy.py` for a complete example
