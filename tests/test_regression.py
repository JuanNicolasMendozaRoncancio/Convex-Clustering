from convex_clustering.regression import fastrfs_sparse, rfs_sparse
import numpy as np

def test_fastrfs_sparse(linear_problem):
    X, y, _ = linear_problem

    b = fastrfs_sparse(X, y, delta=300, epsilon=0.01, numiter=5000)

    y_pred = X @ b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    assert r2 > 0.8, f"R^2 is too low: {r2}"

def test_rfs_sparse_solves_regression(linear_problem):
    X, y, _ = linear_problem
    b = rfs_sparse(X, y, delta=1000, epsilon=0.01, numiter=5000)

    y_pred = X @ b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    assert r2 > 0.8, f"R² demasiado bajo: {r2:.4f}"

def test_rfs_and_fastrfs_coefficients_are_equivalent(linear_problem):
    X, y, _ = linear_problem
    b_rfs = rfs_sparse(X, y, delta=1.0, epsilon=0.01, numiter=5000)
    b_fast = fastrfs_sparse(X, y, delta=1.0, epsilon=0.01, numiter=5000)

    assert np.allclose(b_rfs, b_fast, atol=1e-2), \
        f"Coeficientes divergen:\n  rfs:     {b_rfs}\n  fastrfs: {b_fast}"
