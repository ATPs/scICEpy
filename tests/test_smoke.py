"""End-to-end smoke tests for scICEpy."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
import numpy as np
import pytest
import scanpy as sc

import scICEpy

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _load_and_preprocess_smoke_adata():
    adata = sc.datasets.pbmc68k_reduced()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some cells have zero counts")
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in log1p",
            category=RuntimeWarning,
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if "X_pca" not in adata.obsm:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.pca(adata, n_comps=50)

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    return adata


@pytest.mark.slow
def test_scanpy_smoke_covers_clustering_plotting_and_manual_resolution(tmp_path):
    assert getattr(scICEpy, "__version__", None)
    assert hasattr(scICEpy, "scICE_clustering")
    assert hasattr(scICEpy, "get_robust_labels")
    assert hasattr(scICEpy, "plot_ic")

    adata = _load_and_preprocess_smoke_adata()

    scICEpy.scICE_clustering(
        adata,
        cluster_range=[2, 3, 4],
        n_trials=4,
        n_bootstrap=8,
        n_workers=1,
        seed=42,
        verbose=False,
    )

    results = adata.uns["scICE"]
    assert results["analysis_mode"] == "cluster_range"
    assert len(results["n_cluster"]) > 0
    assert len(results["gamma"]) == len(results["n_cluster"])

    labels_df = scICEpy.get_robust_labels(adata, threshold=10.0)
    assert labels_df.shape[0] == adata.n_obs
    assert all(column.startswith("scICE_k_") for column in labels_df.columns)

    fig, ax = scICEpy.plot_ic(adata, threshold=10.0, show_gamma=True)
    try:
        assert ax.get_title()
        plot_path = Path(tmp_path) / "scICEpy_test_plot.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        assert plot_path.exists()
    finally:
        plt.close(fig)

    resolution_value = float(results["gamma"][0])
    scICEpy.scICE_clustering(
        adata,
        resolution=[resolution_value, resolution_value],
        n_trials=4,
        n_bootstrap=8,
        n_workers=1,
        seed=42,
        verbose=False,
    )

    manual_results = adata.uns["scICE"]
    assert manual_results["analysis_mode"] == "resolution"
    assert np.allclose(
        manual_results["resolution_input"],
        np.asarray([resolution_value], dtype=float),
    )
