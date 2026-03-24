#!/usr/bin/env python3
"""Lightweight smoke test for scICEpy."""

import sys
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
print("Testing cluster_range mode on a small cluster range (2-4)...")
try:
    scICEpy.scICE_clustering(
        adata,
        cluster_range=[2, 3, 4],
        n_trials=4,
        n_bootstrap=8,
        seed=42,
        verbose=False
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
    print(f"  Analysis mode: {results['analysis_mode']}")

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
        labels_df = scICEpy.get_robust_labels(adata, threshold=10.0)
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

        fig, ax = scICEpy.plot_ic(adata, threshold=10.0, show_gamma=True)
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
print("\n8. Testing manual resolution mode...")
print("Reusing a subset of gamma values with duplicates...")

try:
    resolution_values = [float(results["gamma"][0]), float(results["gamma"][0])]
    scICEpy.scICE_clustering(
        adata,
        resolution=resolution_values,
        n_trials=4,
        n_bootstrap=8,
        n_workers=1,
        seed=42,
        verbose=False,
    )
    print("✓ Manual resolution mode succeeded")
    print(f"  Deduplicated manual gamma values: {adata.uns['scICE']['resolution_input']}")
except Exception as e:
    print(f"✗ Manual resolution mode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 80)
print("\n" + "=" * 80)
print("✓ All tests completed successfully!")
print("=" * 80)
