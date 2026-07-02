from convex_clustering import Boosting
import numpy as np
import pytest

def test_fastrfs_sparse(linear_problem):
    X, y, _ = linear_problem

    model = Boosting(algorithm="FastRFS", delta=300, step_size=0.01, max_iter=5000)
    model.fit(X, y)
    y_pred = model.predict(X)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    assert r2 > 0.8, f"R^2 is too low: {r2}"

def test_rfs_sparse_solves_regression(linear_problem):
    X, y, _ = linear_problem
    model = Boosting(algorithm="RFS", delta=1000, step_size=0.01, max_iter=5000)
    model.fit(X, y)
    y_pred = model.predict(X)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    assert r2 > 0.8, f"R² demasiado bajo: {r2:.4f}"

def test_rfs_and_fastrfs_coefficients_are_equivalent(linear_problem):
    X, y, _ = linear_problem
    model_rfs = Boosting(algorithm="RFS", delta=1.0, step_size=0.01, max_iter=5000)
    model_rfs.fit(X, y)

    model_fast = Boosting(algorithm="FastRFS", delta=1.0, step_size=0.01, max_iter=5000)
    model_fast.fit(X, y)

    assert np.allclose(model_rfs.coef_, model_fast.coef_, atol=1e-2), \
        f"Coeficientes divergen:\n  rfs:     {model_rfs.coef_}\n  fastrfs: {model_fast.coef_}"

def test_boosting_invalid_algo(linear_problem):
    X, y, _ = linear_problem

    model = Boosting(algorithm="InvalidAlgo", delta=1.0, step_size=0.01, max_iter=100)
    with pytest.raises(ValueError, match="Algorithm InvalidAlgo not supported. Choose from"):
        model.fit(X, y)

def test_boosting_predict_before_fit(linear_problem):
    X, y, _ = linear_problem

    model = Boosting(algorithm="FastRFS", delta=1.0, step_size=0.01, max_iter=100)
    with pytest.raises(AttributeError, match="object has no attribute 'coef_'"):
        model.predict(X) 
