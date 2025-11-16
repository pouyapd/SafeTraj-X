import numpy as np


def generate_training_data(
    n_samples: int = 800,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic training data in the same range
    as the wheelchair project.

    Each sample: [orientation, v_lin, v_rot]
    """
    rng = np.random.default_rng(seed)

    low = np.array([-np.pi, -1.05, -1.99])
    high = np.array([np.pi, 2.88, 1.99])

    data = rng.uniform(low=low, high=high, size=(n_samples, 3))
    return data
