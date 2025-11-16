from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(traj: np.ndarray):
    """
    Returns a matplotlib figure showing x-y trajectory.
    """
    fig, ax = plt.subplots()
    ax.plot(traj[:, 0], traj[:, 1], marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted trajectory")
    ax.grid(True)
    ax.axis("equal")
    return fig


def plot_feature_importance(importance_dict: Dict[str, float]):
    """
    Returns a matplotlib figure showing a bar chart of feature importance.
    """
    names = list(importance_dict.keys())
    values = [importance_dict[k] for k in names]

    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_ylabel("relative importance")
    ax.set_title("Input feature importance")
    ax.set_ylim(0.0, 1.0)
    return fig
