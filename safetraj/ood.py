from typing import Any

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest


class MahalanobisOOD:
    """
    OOD score based on Mahalanobis distance in input space.
    """

    def __init__(self, training_data: np.ndarray):
        self.cov = EmpiricalCovariance().fit(training_data)
        self.mean = np.mean(training_data, axis=0)

    def score(self, x: np.ndarray) -> float:
        """
        Return sqrt of Mahalanobis distance.
        Larger score => more abnormal.
        """
        delta = x - self.mean
        d2 = self.cov.mahalanobis(delta.reshape(1, -1))
        return float(d2[0] ** 0.5)


class ISOForestOOD:
    """
    OOD score based on Isolation Forest in input space.
    """

    def __init__(self, training_data: np.ndarray, contamination: float = 0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
        )
        self.model.fit(training_data)

    def score(self, x: np.ndarray) -> float:
        """
        Return a positive OOD score.
        Larger score => more abnormal.
        """
        s = -self.model.decision_function([x])[0]
        return float(s)
