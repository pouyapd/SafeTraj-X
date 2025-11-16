from typing import Any, Dict, Sequence

import numpy as np

from .config import SafeTrajConfig
from .data import generate_training_data
from .ood import MahalanobisOOD, ISOForestOOD
from .predictor import DummyTrajectoryPredictor
from .xai import simple_feature_importance


class SafeTrajEvaluator:
    """
    High-level interface that combines:

    - simple kinematic trajectory predictor
    - two input-space OOD detectors
    - a combined risk score + discrete label

    This is meant as a small, self-contained demo
    for safety-aware trajectory evaluation.
    """

    def __init__(self, config: SafeTrajConfig | None = None):
        self.config = config or SafeTrajConfig()

        # training data for OOD models
        self.training_data = generate_training_data(
            n_samples=self.config.n_train_samples,
            seed=self.config.seed,
        )

        # components
        self.predictor = DummyTrajectoryPredictor(
            horizon=self.config.horizon,
            num_steps=self.config.num_steps,
        )
        self.maha = MahalanobisOOD(self.training_data)
        self.iso = ISOForestOOD(
            self.training_data,
            contamination=self.config.contamination,
        )

    # ---------- public API ----------

    def evaluate(
        self,
        x_list: Sequence[float],
        return_traj: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a single input command.

        x_list: [orientation, v_lin, v_rot]
        """
        x = np.asarray(x_list, dtype=float)

        traj = self.predictor.predict(x)
        maha_score = self.maha.score(x)
        iso_score = self.iso.score(x)
        importance = simple_feature_importance(x)

        risk = 0.6 * maha_score + 0.4 * iso_score

        label = self._label_from_scores(maha_score, iso_score)

        result: Dict[str, Any] = {
            "input": [float(v) for v in x],
            "mahalanobis_score": float(maha_score),
            "isolation_forest_score": float(iso_score),
            "risk_score": float(risk),
            "risk_label": label,
            "feature_importance": importance,
        }

        if return_traj:
            result["trajectory"] = traj

        return result

    def get_trajectory(self, x_list: Sequence[float]) -> np.ndarray:
        """Convenience method: only return the trajectory."""
        x = np.asarray(x_list, dtype=float)
        return self.predictor.predict(x)

    # ---------- internal helpers ----------

    def _label_from_scores(self, maha: float, iso: float) -> str:
        """
        Map OOD scores to a simple risk label.
        This is a hand-crafted rule, not learned.
        """
        if maha < self.config.maha_thr and iso < self.config.iso_thr:
            return "in-distribution & low-risk"

        if (maha > self.config.maha_thr * 1.7) or (
            iso > self.config.iso_thr * 2.0
        ):
            return "high OOD / high-risk"

        return "borderline / uncertain"
