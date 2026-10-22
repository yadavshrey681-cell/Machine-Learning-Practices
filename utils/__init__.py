"""
Shared utilities for machine learning practices: evaluation metrics, plotting, and data inspection.
"""

from .metrics import print_classification_summary, print_regression_summary
from .visualization import plot_confusion_matrix, plot_learning_curve

__all__ = [
    "print_classification_summary",
    "print_regression_summary",
    "plot_confusion_matrix",
    "plot_learning_curve",
]
