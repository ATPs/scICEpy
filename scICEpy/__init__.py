"""scICEpy public package exports."""

__version__ = "0.1.0"
__author__ = "Xiaolong Cao"

from .scICEpy import scICE_clustering
from .visualization import get_robust_labels, plot_ic

__all__ = ["scICE_clustering", "get_robust_labels", "plot_ic"]
