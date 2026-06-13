import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import norm as sp_norm

def _normalize_sparse(X, y):
    """
    Normalizes the data matrix X and the target vector y for sparse data.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        y : array-like, shape (n,), target vector.

    Returns
    -------
        X_normalized : array-like, shape (p, n), normalized data matrix.
        y_normalized : array-like, shape (n,), normalized target vector.
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
        X = X.multiply(1 / norms)
    else:
        X = X / norms

    return X, y

def fastrfs_sparse(X, y, delta, epsilon, numiter):
    """
    Implements the Fast RF-S algorithm with sparce data

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        y : array-like, shape (n,), target vector.
        delta : float, regularization parameter.
        epsilon : float, step size for the primal variable update.
        numiter : int, maximum number of iterations.

    Returns
    -------
        b : array-like, shape (p,), the rsulting coefficients.
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