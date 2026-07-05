from convex_clustering import ConvexClusterer
import numpy as np
import pytest


def test_admm_centers(six_points, expected_centers):
    X, W = six_points

    model = ConvexClusterer(
        algorithm="ADMM",
        gamma=10,
        step_size=0.5,
        max_iter=100,
        tol = 1e-4,
    )

    model.fit(X, W)


    center_comp0 = model.cluster_centers_[[0, 3, 5]].mean(axis=0)
    center_comp1 = model.cluster_centers_[[1, 2, 4]].mean(axis=0)

    assert np.allclose(center_comp0, expected_centers[0], atol=1e-2)
    assert np.allclose(center_comp1, expected_centers[1], atol=1e-2)

def test_ama_centers(six_points, expected_centers):
    X, W = six_points

    model = ConvexClusterer(
        algorithm="AMA",
        gamma=10,
        step_size=0.5,
        max_iter=100,
        tol = 1e-4,
    )

    model.fit(X, W)

    center_comp0 = model.cluster_centers_[[0, 3, 5]].mean(axis=0)
    center_comp1 = model.cluster_centers_[[1, 2, 4]].mean(axis=0)

    assert np.allclose(center_comp0, expected_centers[0], atol=1e-2)
    assert np.allclose(center_comp1, expected_centers[1], atol=1e-2)

def test_DR_centers(six_points, expected_centers):
    X, W = six_points

    model = ConvexClusterer(
        algorithm="DR",
        gamma=10,
        step_size=0.5,
        max_iter=100,
        tol = 1e-4,
    )

    model.fit(X, W)

    center_comp0 = model.cluster_centers_[[0, 3, 5]].mean(axis=0)
    center_comp1 = model.cluster_centers_[[1, 2, 4]].mean(axis=0)

    assert np.allclose(center_comp0, expected_centers[0], atol=1e-2)
    assert np.allclose(center_comp1, expected_centers[1], atol=1e-2)

@pytest.mark.parametrize("algorithm",["ADMM", "AMA", "DR"])
def test_history_not_empty(six_points, algorithm):
    X, W = six_points

    model = ConvexClusterer(
        algorithm=algorithm,
        gamma=10,
        step_size=0.5,
        max_iter=100,
        tol = 1e-4,
    )

    model.fit(X, W)

    assert len(model.history_) > 0, (
        f"{algorithm} history is empty after fitting the model"
    )
    assert all(isinstance(v, float) for v in model.history_.values()), (
        f"{algorithm} history values are not all floats"
    )