from dataclasses import dataclass


@dataclass
class SafeTrajConfig:
    """
    Configuration for SafeTrajEvaluator.

    You can change these values when you create the evaluator.
    """

    # training data
    n_train_samples: int = 800
    seed: int = 42

    # OOD parameters
    maha_thr: float = 3.0
    iso_thr: float = 0.3
    contamination: float = 0.05

    # trajectory prediction
    horizon: float = 1.5
    num_steps: int = 20
