from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse
from scipy.sparse.linalg import norm as sp_norm
from sklearn.base import BaseEstimator, RegressorMixin

def _normalize_sparse(X: npt.NDArray[np.float64],
                      y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Normalize the data matrix X and target vector y.
 
    Centers y and normalizes columns of X to unit norm.
 
    Parameters
    ----------
    X : array-like of shape (n_features, n_samples)
        Data matrix (note: transposed convention used internally).
    y : array-like of shape (n_samples,)
        Target vector.
 
    Returns
    -------
    X_normalized : ndarray of shape (n_features, n_samples)
        Column-normalized data matrix.
    y_normalized : ndarray of shape (n_samples,)
        Mean-centered target vector.
    """
    y = y - y.mean()
    mean = np.mean(X, axis=0)
    X = X - mean
    if issparse(X):
        norms = sp_norm(X, axis=0)
    else:
        X = np.asarray(X)
        norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    if issparse(X):
        X = X.multiply(1 / norms) # type: ignore[attr-defined]
    else:
        X = X / norms
    return X, y

def rfs_sparse(X: npt.NDArray[np.float64],
               y: npt.NDArray[np.float64],
               delta: float,
               epsilon: float,
               numiter: int) -> npt.NDArray[np.float64]:
    """
    RF-S algorithm for linear regression (dense and sparse data).
 
    Implements the incremental forward stagewise path. As epsilon → 0
    the path converges to the Lasso solution path (Efron et al., 2004).
 
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix in sklearn convention. Transposed internally.
    y : array-like of shape (n_samples,)
        Target vector.
    delta : float
        L1 regularization parameter. Must satisfy 0 < epsilon < delta.
    epsilon : float
        Step size for the coefficient update. Must satisfy 0 < epsilon < delta.
    numiter : int
        Number of iterations (no early stopping).
 
    Returns
    -------
    b : ndarray of shape (n_features,)
        Regression coefficients.
 
    Raises
    ------
    ValueError
        If epsilon <= 0, delta <= 0, or epsilon >= delta.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if epsilon >= delta:
        raise ValueError("epsilon must be less than delta.")
    X = X.T
    X, y = _normalize_sparse(X, y)
    b = np.zeros(X.shape[0], dtype=np.float64)  # (p,)
    r = y.copy()
    for _ in range(numiter):
        if issparse(X):
            corr = np.abs(X @ r)
            if issparse(corr):
                corr = corr.A.ravel() # type: ignore[attr-defined]
        else:
            corr = np.abs(X @ r)
        j_k = np.argmax(corr)
        x_j = X[j_k, :]
        if issparse(x_j):
            x_j = x_j.A.ravel() # type: ignore[attr-defined]
        else:
            x_j = np.asarray(x_j).ravel()
        s = np.sign(np.dot(x_j, r))
        r -= epsilon * (s * x_j + (1.0 / delta) * (r - y))
        b *= (1.0 - epsilon / delta)
        b[j_k] += epsilon * s
    return b

def fastrfs_sparse(X: npt.NDArray[np.float64],
                   y: npt.NDArray[np.float64],
                   delta: float,
                   epsilon: float,
                   numiter: int) -> npt.NDArray[np.float64]:
    """
    Fast RF-S algorithm for linear regression (dense and sparse data).
 
    Accelerated version of rfs_sparse that maintains a running correlation
    vector updated in closed form, avoiding a full matrix-vector product
    per iteration.
 
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix in sklearn convention.
    y : array-like of shape (n_samples,)
        Target vector.
    delta : float
        L1 regularization parameter.
    epsilon : float
        Step size for the coefficient update.
    numiter : int
        Number of iterations (no early stopping).
 
    Returns
    -------
    b : ndarray of shape (n_features,)
        Regression coefficients.
    """
    X, y = _normalize_sparse(X, y)
    r = y.copy()
    b = np.zeros(X.shape[1])
    alpha = (epsilon / delta) * (y.T @ X)
    gamma = r.T @ X
    for _ in range(numiter):
        j_k = np.argmax(np.abs(gamma))
        s_k = np.sign(gamma[j_k])
        x_j = X[:, j_k]
        temp = np.dot(x_j.T, X)
        if issparse(temp):
            sigma_k = temp.A.ravel()
        else:
            sigma_k = temp.ravel()
        gamma += alpha - epsilon * s_k * sigma_k - (epsilon / delta) * gamma
        b *= (1 - epsilon / delta)
        b[j_k] += epsilon * s_k
    return b

class Boosting(BaseEstimator, RegressorMixin): # type: ignore[misc]
    """
    Forward stagewise boosting regressor via RF-S / Fast RF-S.
 
    Wraps rfs_sparse and fastrfs_sparse behind a scikit-learn-compatible
    fit()/predict() interface. Both underlying algorithms solve the same
    L1-regularized regression problem — the incremental forward stagewise
    path, which converges to the Lasso solution path as step_size -> 0
    (Efron et al., 2004) — so 'algorithm' selects a computational strategy,
    not a different statistical model.
 
    Parameters
    ----------
    algorithm : str, optional
        One of 'RFS', 'FastRFS', by default 'FastRFS'.
    delta : float, optional
        L1 regularization parameter, by default 1.0.
        Must satisfy 0 < step_size < delta.
    step_size : float, optional
        Step size for the iterative coefficient update, by default 0.01.
    max_iter : int, optional
        Number of iterations to run, by default 1000. No early stopping;
        both algorithms run for exactly max_iter steps.
 
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted regression coefficients.
    n_iter_ : int
        Number of iterations actually run (always equals max_iter).
    """

    _ALGORITHMS = frozenset({'RFS', 'FastRFS'})

    def __init__(self,
                 algorithm: str = "FastRFS",
                 delta: float = 1.0,
                 step_size: float = 0.01,
                 max_iter: int = 1000,
    ) -> None:
        self.algorithm = algorithm
        self.delta = delta
        self.step_size = step_size
        self.max_iter = max_iter

    def fit(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64]
    ) -> Boosting:
        """
        Fit coefficients via forward stagewise boosting.
 
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix.
        y : ndarray of shape (n_samples,)
            Target vector.
 
        Returns
        -------
        self : Boosting
            Fitted estimator.
 
        Raises
        ------
        ValueError
            If algorithm is not one of the supported options.
        """
        if self.algorithm not in self._ALGORITHMS:
            raise ValueError(f"Algorithm {self.algorithm} not supported. Choose from {self._ALGORITHMS}.")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.algorithm == "RFS":
            coef = rfs_sparse(X, y, self.delta, self.step_size, self.max_iter)
        else:  
            coef = fastrfs_sparse(X, y, self.delta, self.step_size, self.max_iter)

        self.coef_: npt.NDArray[np.float64] = coef
        self.n_iter_: int = self.max_iter

        return self
    
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predict target values as X @ coef_.
 
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix.
 
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_
