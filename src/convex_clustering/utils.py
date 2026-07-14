from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, identity, kron
from scipy.spatial.distance import cdist


def knn_w(X: npt.NDArray[np.float64], k:int=3, phi: float=0.5) -> npt.NDArray[np.float64]:
    """
    Compute the k-nearest neighbor weight matrix for the given data.
 
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    k : int, optional
        Number of nearest neighbors to consider, by default 3.
    phi : float, optional
        Scaling factor for the Gaussian kernel weights, by default 0.5.
        Higher phi down-weights distant neighbors more aggressively.
 
    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Weight matrix where W[i, j] = exp(-phi * d(i, j)) if j is among
        the k nearest neighbors of i, and 0 otherwise.
    """
    D = cdist(X,X, 'euclidean')
    np.fill_diagonal(D, np.inf)
    n = D.shape[0]
    W = np.zeros((n,n))
    for i in range(n):
        idx = np.argsort(D[i,:])
        for j in idx[0:k]:
            W[i,j] = np.exp(-phi * D[i,j])

    return W

def construct_weighted_laplacian(W: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Build the weighted Laplacian matrix from the weight matrix W.
 
    Parameters
    ----------
    W : array-like of shape (n_samples, n_samples)
        Symmetric weight matrix.
 
    Returns
    -------
    L : ndarray of shape (n_samples, n_samples)
        Weighted Laplacian matrix L = D - W, where D is the degree matrix.
    """
    D = np.diag(np.sum(W, axis=1))
    return D - W

def built_edges(W: npt.NDArray[np.float64]
                ) -> tuple[list[tuple[int,int]], npt.NDArray[np.float64]]:
    """
    Build the edge list and corresponding weights from the weight matrix W.
 
    Only upper-triangular entries are considered (W is assumed symmetric).
 
    Parameters
    ----------
    W : array-like of shape (n_samples, n_samples)
        Symmetric weight matrix. W[i, j] > 0 means there is an edge
        between points i and j.
 
    Returns
    -------
    edges : list of tuple of (int, int)
        List of (i, j) pairs with i < j for all edges with positive weight.
    weights : ndarray of shape (n_edges,)
        Weight W[i, j] for each edge in edges.
    """
    n = W.shape[0] # number of nodes
    edges: list[tuple[int,int]] = []
    weights: list[float] = []
    for i in range(n):
        for j in range(i+1, n): # W is symmetric
            if W[i,j] > 0:
                edges.append((i,j))
                weights.append(W[i,j])
    return edges, np.array(weights, dtype=np.float64)

def compute_b_penal(W: npt.NDArray[np.float64],
                    X: npt.NDArray[np.float64],
                    gamma: float) -> tuple[Any, float]:
    """
    Compute the incidence matrix B and the penalty term for the RF-S variants.
 
    Parameters
    ----------
    W : array-like of shape (n_samples, n_samples)
        Weight matrix representing the graph structure.
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    gamma : float
        Regularization parameter.
 
    Returns
    -------
    B : scipy.sparse.csr_matrix of shape (n_samples * n_features, n_edges * n_features)
        Kronecker-expanded incidence matrix used in the RF-S regularization term.
    penalty : float
        Penalty term gamma * sqrt(n_features) * sum(weights).
    """
    edges, weigths = built_edges(W)
    n, p = X.shape
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for k, (i, j) in enumerate(edges):
        rows.extend([i, j])
        cols.extend([k, k])
        data.extend([1, -1])
    b_B = coo_matrix((data, (rows, cols)), shape=(n, len(edges))).tocsr()
    I = identity(p, format='csr')
    B = kron(b_B, I, format='csr')
    penalty: float = gamma * np.sqrt(p)*np.sum(weigths)
    return B, penalty
