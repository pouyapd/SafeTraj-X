"""
SafeTraj-X package.

Small demo package for:
- simple trajectory prediction
- input-space OOD detection
- risk scoring and simple feature importance
"""

from .config import SafeTrajConfig
from .evaluator import SafeTrajEvaluator
from .plotting import plot_trajectory, plot_feature_importance

__all__ = [
    "SafeTrajConfig",
    "SafeTrajEvaluator",
    "plot_trajectory",
    "plot_feature_importance",
]
