import logging
import matplotlib
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

import scICEpy
from scICEpy.runtime import create_runtime_context, get_scicepy_log_formatter
from scICEpy.visualization import plot_ic

matplotlib.use("Agg")


def _make_toy_adata(seed: int = 1) -> AnnData:
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.4, size=(20, 12)),
            rng.normal(loc=0.0, scale=0.4, size=(20, 12)),
            rng.normal(loc=2.0, scale=0.4, size=(20, 12)),
        ]
    )
    adata = AnnData(x)
    sc.pp.neighbors(adata, n_neighbors=8)
    return adata


def test_manual_resolution_mode_deduplicates_inputs():
    adata = _make_toy_adata()
    scICEpy.scICE_clustering(
        adata,
        resolution=[0.05, 0.05, 0.10],
        n_trials=3,
        n_bootstrap=4,
        n_workers=1,
        seed=123,
        verbose=False,
    )
    results = adata.uns["scICE"]
    assert results["analysis_mode"] == "resolution"
    assert np.allclose(results["resolution_input"], np.asarray([0.05, 0.10], dtype=float))
    assert results["resolution_search_diagnostics"] is None
    assert results["target_diagnostics"] is None
    assert len(results["resolution_diagnostics"]) == 2
    assert np.all(results["cluster_range_tested"] == results["n_cluster"])


def test_plot_ic_show_gamma_renders_gamma_labels():
    results = {
        "n_cluster": np.asarray([2, 3], dtype=int),
        "gamma": np.asarray([0.05, 0.10], dtype=float),
        "ic": np.asarray([1.0, 1.1], dtype=float),
        "ic_vec": [np.asarray([1.0, 1.0], dtype=float), np.asarray([1.05, 1.15], dtype=float)],
        "target_diagnostics": None,
    }
    fig, ax = plot_ic(results, threshold=1.2, show_gamma=True)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert any("\n" in label for label in labels)
    fig.clf()


def test_get_robust_labels_returns_scanpy_style_columns():
    adata = _make_toy_adata(seed=2)
    scICEpy.scICE_clustering(
        adata,
        cluster_range=[2, 3],
        n_trials=3,
        n_bootstrap=4,
        n_workers=1,
        seed=321,
        verbose=False,
    )
    labels_df = scICEpy.get_robust_labels(adata, threshold=10.0)
    assert all(column.startswith("scICE_k_") for column in labels_df.columns)
    assert labels_df.shape[0] == adata.n_obs


def test_beta_warning_and_graph_name_alias():
    adata = _make_toy_adata(seed=3)
    with pytest.warns(RuntimeWarning, match="beta="):
        scICEpy.scICE_clustering(
            adata,
            cluster_range=[2],
            n_trials=2,
            n_bootstrap=2,
            n_workers=1,
            seed=123,
            beta=0.2,
            verbose=False,
        )
    results = adata.uns["scICE"]
    assert results["graph_name"] == "connectivities"
    assert results["beta_supported"] is False
    assert results["beta_applied"] is False


def test_parallel_layout_fields_respect_explicit_outer_inner_workers():
    adata = _make_toy_adata(seed=4)
    scICEpy.scICE_clustering(
        adata,
        resolution=[0.05, 0.10],
        n_trials=2,
        n_bootstrap=2,
        n_workers=2,
        outer_workers=1,
        inner_workers=2,
        seed=123,
        verbose=False,
    )
    layout = adata.uns["scICE"]["parallel_layout"]
    assert int(layout["outer_workers"]) == 1
    assert int(layout["inner_workers"]) == 2


def test_runtime_context_defaults_to_current_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime_context = create_runtime_context()
    try:
        assert str(tmp_path) == runtime_context.scratch_root
        assert str(tmp_path / ".scicepy_tmp") in runtime_context.runtime_dir
    finally:
        from scICEpy.runtime import cleanup_runtime_spill

        cleanup_runtime_spill(runtime_context)


def test_scicepy_log_formatter_matches_scicer_style():
    formatter = get_scicepy_log_formatter()
    record = logging.LogRecord(
        name="scICEpy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert rendered.startswith("[")
    assert "] test message" in rendered
