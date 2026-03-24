# Repository Guidelines

## Project Structure & Module Organization
`scICEpy/` contains the library code. Keep public AnnData-facing entry points in `scICEpy/api.py`, and re-export stable APIs from `scICEpy/__init__.py`. Core algorithm logic is split across focused modules such as `optimization.py`, `resolution_search.py`, `runtime.py`, `metrics.py`, `results.py`, `visualization.py`, and `leiden_wrapper.py`.

Tests live in `tests/` and cover unit-level behavior. `test_scICEpy.py` is a root-level smoke test that exercises installation, Scanpy integration, plotting, and manual resolution mode. Supporting documentation is in `README.md` and `design.md`. Utility scripts belong in `scripts/` (currently `scripts/qs_to_h5ad.R`).

## Build, Test, and Development Commands
Use the repository’s Python environment when possible:

```bash
export PATH=/data/p/bin:$PATH
/data/p/anaconda3/bin/python -m pip install -e ".[dev]"
/data/p/anaconda3/bin/python -m pytest -q
/data/p/anaconda3/bin/python -m pytest --cov=scICEpy tests
/data/p/anaconda3/bin/python test_scICEpy.py
```

`pip install -e ".[dev]"` installs the package plus pytest tooling. `pytest -q` runs the main automated suite. The coverage variant is useful before merging larger algorithm changes. `test_scICEpy.py` is slower, but it verifies end-to-end behavior with Scanpy and matplotlib.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, PEP 8 naming, `snake_case` for functions and variables, and short module docstrings. Prefer explicit imports and small helper functions over large monolithic blocks. Keep new public API behavior AnnData-centric and preserve the `adata.uns["scICE"]` result contract. Match the existing use of type hints where they improve readability.

There is no repository-enforced formatter or linter config yet, so keep changes consistent with neighboring code and avoid unrelated style churn.

## Testing Guidelines
Add or update pytest coverage for every behavior change. Place new tests in `tests/test_*.py`, and name functions `test_<behavior>()`. Prefer deterministic toy AnnData fixtures and fixed random seeds for clustering-related assertions. If you touch plotting or packaging, also run `test_scICEpy.py`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Enhance get_robust_labels...` and `Refactor installation instructions...`. Keep commit messages concise, capitalized, and focused on one logical change.

Pull requests should explain the user-visible impact, note any API or result-structure changes, and list the validation commands you ran. Include plots or screenshots only when changing visualization output.
