"""Compatibility shim for importing scICEpy from the repository parent."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_IMPL_PACKAGE = import_module(".scICEpy", __name__)

__all__ = list(getattr(_IMPL_PACKAGE, "__all__", []))
__version__ = getattr(_IMPL_PACKAGE, "__version__", None)
__author__ = getattr(_IMPL_PACKAGE, "__author__", None)

for _name in __all__:
    globals()[_name] = getattr(_IMPL_PACKAGE, _name)

for _submodule in (
    "api",
    "large_h5ad",
    "leiden_wrapper",
    "metrics",
    "optimization",
    "resolution_search",
    "results",
    "runtime",
    "visualization",
):
    _module = import_module(f".scICEpy.{_submodule}", __name__)
    globals()[_submodule] = _module
    _sys.modules[f"{__name__}.{_submodule}"] = _module
