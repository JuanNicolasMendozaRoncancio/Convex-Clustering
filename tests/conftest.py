from sklearn.datasets import make_regression
from convex_clustering.utils import knn_w
import numpy as np
import pytest


@pytest.fixture
def six_points():
    X = np.random.default_rng(7).normal(size=(6, 2))
    W = knn_w(X, k=2, phi=0.5)
    return X, W

@pytest.fixture
def expected_centers():
    return {
        0: np.array([0.1837386,  0.6652826]),
        1: np.array([-0.40700505, -0.83423776]),
    }

@pytest.fixture
def linear_problem():
    X, y, b_true = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        noise=0.01,
        coef=True,
        random_state=42
    )
    return X, y, b_true
