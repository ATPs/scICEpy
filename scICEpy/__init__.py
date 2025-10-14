"""
scICEpy - Single-cell Inconsistency-based Clustering Evaluation

A Python implementation of scICE for evaluating clustering consistency
in single-cell RNA-seq data.
"""

__version__ = "0.1.0"
__author__ = "Xiaolong Cao"

from .scICEpy import (
    scICE_clustering,
    get_robust_labels,
    plot_ic,
)

__all__ = [
    "scICE_clustering",
    "get_robust_labels",
    "plot_ic",
]
