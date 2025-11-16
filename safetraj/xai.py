import numpy as np
from typing import Dict


def simple_feature_importance(x: np.ndarray) -> Dict[str, float]:
    """
    Very simple, heuristic "importance":
    importance_i = |x_i| / sum_j |x_j|

    This is NOT model-based explainability,
    just an easy way to show relative magnitudes.
    """
    names = ["orientation", "v_lin", "v_rot"]
    abs_x = np.abs(x)
    total = float(np.sum(abs_x) + 1e-6)
    importance = abs_x / total
    return {name: float(val) for name, val in zip(names, importance)}
