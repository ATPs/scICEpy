"""Compatibility facade for the public scICEpy API."""

from .api import get_robust_labels, plot_ic, scICE_clustering

__all__ = ["scICE_clustering", "get_robust_labels", "plot_ic"]
