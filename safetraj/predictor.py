from dataclasses import dataclass

import numpy as np


@dataclass
class DummyTrajectoryPredictor:
    """
    Simple kinematic-style trajectory predictor.

    Input:  x = [orientation, v_lin, v_rot]
    Output: array of shape (T, 3) -> [x(t), y(t), orientation(t)]
    """

    horizon: float = 1.5
    num_steps: int = 20

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Compute a simple straight/curved trajectory."""
        angle, v_lin, v_rot = x
        t = np.linspace(0.0, self.horizon, self.num_steps)

        x_traj = v_lin * np.cos(angle) * t
        y_traj = v_lin * np.sin(angle) * t
        theta_traj = angle + v_rot * t

        traj = np.vstack([x_traj, y_traj, theta_traj]).T
        return traj
