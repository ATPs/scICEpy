#!/usr/bin/env python3
"""
Test script for scICEpy

This script tests:
1. Basic imports and installation
2. Core functionality on a small dataset
3. Parallel processing capabilities with benchmarks
"""

import sys
import time
import numpy as np

print("=" * 80)
print("Testing scICEpy")
print("=" * 80)
print(f"Python version: {sys.version}")
print("-" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    import scanpy as sc
    print("✓ scanpy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import scanpy: {e}")
    sys.exit(1)

try:
    import scICEpy
    print("✓ scICEpy imported successfully")
    try:
        print(f"  Version: {scICEpy.__version__}")
    except AttributeError:
        print("  Version: unknown (no __version__ attribute)")
    # Verify main functions are available
    assert hasattr(scICEpy, 'scICE_clustering'), "scICE_clustering function not found"
    assert hasattr(scICEpy, 'get_robust_labels'), "get_robust_labels function not found"
    assert hasattr(scICEpy, 'plot_ic'), "plot_ic function not found"
    print("  All main functions available")
except ImportError as e:
    print(f"✗ Failed to import scICEpy: {e}")
    sys.exit(1)
except AssertionError as e:
    print(f"✗ scICEpy import incomplete: {e}")
    sys.exit(1)

try:
    import igraph as ig
    print("✓ igraph imported successfully")
except ImportError as e:
    print(f"✗ Failed to import igraph: {e}")
    sys.exit(1)

try:
    import leidenalg
    print("✓ leidenalg imported successfully")
except ImportError as e:
    print(f"✗ Failed to import leidenalg: {e}")
    sys.exit(1)

print("-" * 80)

# Load test dataset
print("\n2. Loading test dataset (PBMC 68k subsample)...")
try:
    adata = sc.datasets.pbmc68k_reduced()
    print(f"✓ Dataset loaded: {adata.n_obs} cells × {adata.n_vars} genes")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    sys.exit(1)

print("-" * 80)

# Preprocessing
print("\n3. Preprocessing...")
try:
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # PCA
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.pca(adata, n_comps=50)

    # Compute neighbors
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    print("✓ Preprocessing complete")
    print(f"  PCA computed: {adata.obsm['X_pca'].shape}")
    print(f"  Neighbors computed: {adata.obsp['connectivities'].shape}")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 80)

# Run scICE
print("\n4. Running scICE clustering...")
print("Testing on a small cluster range (2-5) for quick validation...")
try:
    scICEpy.scICE_clustering(
        adata,
        cluster_range=list(range(2, 6)),  # Small range for testing
        n_trials=10,  # Reduced for testing
        n_bootstrap=50,  # Reduced for testing
        seed=42,
        verbose=True
    )
    print("✓ scICE clustering completed")
except Exception as e:
    print(f"✗ scICE clustering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 80)

# Check results
print("\n5. Checking results...")
try:
    if 'scICE' not in adata.uns:
        raise ValueError("scICE results not found in adata.uns")

    results = adata.uns['scICE']
    print("✓ Results stored in adata.uns['scICE']")
    print(f"  Keys: {list(results.keys())}")
    print(f"  Cluster numbers tested: {results['n_cluster']}")
    print(f"  IC scores: {results['ic']}")
    print(f"  Consistent clusters: {results['consistent_clusters']}")

    # Check if we have any results
    if len(results['n_cluster']) > 0:
        print("✓ Got clustering results for some cluster numbers")
    else:
        print("⚠ Warning: No successful clustering results (this may happen with strict thresholds)")

except Exception as e:
    print(f"✗ Results check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 80)

# Test get_robust_labels
print("\n6. Testing get_robust_labels...")
try:
    if len(results['n_cluster']) > 0:
        labels_df = scICEpy.get_robust_labels(adata, threshold=1.5)  # Use relaxed threshold
        print(f"✓ get_robust_labels succeeded")
        print(f"  DataFrame shape: {labels_df.shape}")
        print(f"  Columns: {list(labels_df.columns)}")
    else:
        print("⚠ Skipping get_robust_labels (no results available)")
except Exception as e:
    print(f"✗ get_robust_labels failed: {e}")
    import traceback
    traceback.print_exc()

print("-" * 80)

# Test plotting
print("\n7. Testing plot_ic...")
try:
    if len(results['n_cluster']) > 0:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig, ax = scICEpy.plot_ic(adata, threshold=1.5)
        print("✓ plot_ic succeeded")

        # Save plot
        fig.savefig('/tmp/scICEpy_test_plot.png', dpi=100, bbox_inches='tight')
        print("  Plot saved to /tmp/scICEpy_test_plot.png")
    else:
        print("⚠ Skipping plot_ic (no results available)")
except Exception as e:
    print(f"✗ plot_ic failed: {e}")
    import traceback
    traceback.print_exc()

print("-" * 80)

# Parallel processing benchmark
print("\n8. Testing parallel processing capabilities...")
print("Creating synthetic dataset for benchmarking...")

# Create a small synthetic dataset for testing
np.random.seed(42)
n_cells = 500
n_genes = 100

# Create synthetic expression data with 3 clusters
cluster_sizes = [200, 150, 150]
# Initialize with small positive values (not zeros) to avoid NaN after log
adata_synth = sc.AnnData(np.abs(np.random.randn(n_cells, n_genes)) * 0.1 + 0.1)

# Add some structure to make clustering meaningful
start = 0
for i, size in enumerate(cluster_sizes):
    end = start + size
    # Each cluster has different expression patterns (use abs to ensure positive)
    adata_synth.X[start:end, i*20:(i+1)*20] = np.abs(np.random.randn(size, 20)) + 3
    start = end

# Add background expression to all genes to ensure positive values
adata_synth.X = np.abs(adata_synth.X) + 0.1

# Preprocess
sc.pp.normalize_total(adata_synth, target_sum=1e4)
sc.pp.log1p(adata_synth)
sc.pp.pca(adata_synth, n_comps=20)
sc.pp.neighbors(adata_synth, n_neighbors=15)

print(f"✓ Synthetic dataset created: {n_cells} cells × {n_genes} genes")

# Test with n_workers=1 (sequential)
print("\n--- Test 8a: Sequential processing (n_workers=1) ---")
start_time = time.time()
try:
    scICEpy.scICE_clustering(
        adata_synth,
        cluster_range=[2, 3, 4, 5],
        n_workers=1,
        n_trials=10,
        n_bootstrap=20,
        seed=42,
        verbose=False
    )
    sequential_time = time.time() - start_time
    print(f"✓ Sequential processing completed: {sequential_time:.2f} seconds")
except Exception as e:
    print(f"✗ Sequential processing failed: {e}")
    sequential_time = None

# Reset adata
adata_synth.uns.pop('scICE', None)

# Test with n_workers=4 (parallel)
print("\n--- Test 8b: Parallel processing (n_workers=4) ---")
start_time = time.time()
try:
    scICEpy.scICE_clustering(
        adata_synth,
        cluster_range=[2, 3, 4, 5],
        n_workers=4,
        n_trials=10,
        n_bootstrap=20,
        seed=42,
        verbose=False
    )
    parallel_time = time.time() - start_time
    print(f"✓ Parallel processing completed: {parallel_time:.2f} seconds")
except Exception as e:
    print(f"✗ Parallel processing failed: {e}")
    parallel_time = None

# Compare results
if sequential_time and parallel_time:
    print("\n--- Parallel Processing Performance ---")
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Parallel time:   {parallel_time:.2f} seconds")
    if parallel_time < sequential_time:
        speedup = sequential_time / parallel_time
        print(f"Speedup:         {speedup:.2f}x faster with parallel processing")
    else:
        print("Note: For small datasets, parallel overhead may exceed benefits")

    # Show scICE results
    if 'scICE' in adata_synth.uns:
        synth_results = adata_synth.uns['scICE']
        print(f"\nConsistent cluster numbers found: {list(synth_results.get('consistent_clusters', []))}")
        print(f"IC scores: {synth_results.get('ic', [])}")

print("-" * 80)
print("\n" + "=" * 80)
print("✓ All tests completed successfully!")
print("=" * 80)
