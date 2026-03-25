import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from scICEpy.large_h5ad import create_light_h5ad, write_scice_results_back


def _make_large_h5ad_fixture(seed: int = 11) -> AnnData:
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.35, size=(15, 8)),
            rng.normal(loc=0.0, scale=0.35, size=(15, 8)),
            rng.normal(loc=2.0, scale=0.35, size=(15, 8)),
        ]
    )
    obs = pd.DataFrame({"sample": ["a"] * 15 + ["b"] * 15 + ["c"] * 15})
    obs.index = [f"cell_{idx}" for idx in range(x.shape[0])]
    var = pd.DataFrame({"feature_type": ["gene"] * x.shape[1]})
    var.index = [f"gene_{idx}" for idx in range(x.shape[1])]
    adata = AnnData(x, obs=obs, var=var)
    adata.layers["counts"] = np.rint(np.abs(x) * 10).astype(np.int32)
    adata.obsm["X_custom"] = rng.normal(size=(adata.n_obs, 3))
    adata.uns["user_note"] = {"name": "toy"}
    sc.pp.neighbors(adata, n_neighbors=6)
    return adata


def test_create_light_h5ad_preserves_scicepy_inputs(tmp_path):
    source_path = tmp_path / "source.h5ad"
    light_path = tmp_path / "source.light.h5ad"
    adata = _make_large_h5ad_fixture()
    adata.write_h5ad(source_path)

    create_light_h5ad(source_path, light_path, n_vars=2)

    light = sc.read_h5ad(light_path)
    assert light.n_obs == adata.n_obs
    assert light.n_vars == 2
    assert list(light.obs_names) == list(adata.obs_names)
    assert list(light.var_names) == list(adata.var_names[:2])
    assert "X_custom" in light.obsm
    assert set(light.obsp.keys()) == set(adata.obsp.keys())
    assert "user_note" in light.uns
    assert "counts" in light.layers
    meta = light.uns["scICEpy_light_h5ad"]
    assert meta["source_path"] == str(source_path)
    assert meta["n_obs"] == adata.n_obs
    assert meta["n_vars_original"] == adata.n_vars
    assert meta["n_vars_kept"] == 2


def test_write_scice_results_back_persists_nested_results(tmp_path):
    source_path = tmp_path / "source.h5ad"
    adata = _make_large_h5ad_fixture(seed=12)
    adata.write_h5ad(source_path)

    results = {
        "analysis_mode": "resolution",
        "n_cluster": np.asarray([2, 3], dtype=int),
        "gamma": np.asarray([0.05, 0.10], dtype=float),
        "ic": np.asarray([1.0, 1.1], dtype=float),
        "ic_vec": [np.asarray([1.0, 1.0], dtype=float), np.asarray([1.05, 1.15], dtype=float)],
        "target_diagnostics": pd.DataFrame(
            {"searched_target_cluster": [2, 3], "excluded": [False, True]}
        ),
        "cell_names": np.asarray(adata.obs_names, dtype=object),
    }

    write_scice_results_back(source_path, results=results, expected_obs_names=adata.obs_names)

    restored = sc.read_h5ad(source_path)
    assert restored.uns["scICE"]["analysis_mode"] == "resolution"
    assert np.allclose(restored.uns["scICE"]["gamma"], np.asarray([0.05, 0.10], dtype=float))
    assert isinstance(restored.uns["scICE"]["target_diagnostics"], pd.DataFrame)
    assert list(restored.uns["scICE"]["target_diagnostics"]["excluded"]) == [False, True]


def test_run_large_h5ad_scice_script_writes_results_back_to_original(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_large_h5ad_scice.py"
    source_path = tmp_path / "source.h5ad"
    light_path = tmp_path / "source.light.h5ad"
    adata = _make_large_h5ad_fixture(seed=13)
    adata.write_h5ad(source_path)

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(source_path),
            "--light-output",
            str(light_path),
            "--n-vars",
            "2",
            "--resolution",
            "0.05",
            "0.10",
            "--n-trials",
            "2",
            "--n-bootstrap",
            "2",
            "--n-workers",
            "1",
            "--seed",
            "123",
            "--quiet",
        ],
        check=True,
        cwd=repo_root,
    )

    assert light_path.exists()
    restored = sc.read_h5ad(source_path)
    light = sc.read_h5ad(light_path)
    assert "scICE" in restored.uns
    assert restored.uns["scICE"]["analysis_mode"] == "resolution"
    assert restored.uns["scICE"]["graph_key"] == "connectivities"
    assert len(restored.uns["scICE"]["cell_names"]) == restored.n_obs
    assert "scICE" in light.uns
