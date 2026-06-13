import numpy as np
from scipy.sparse import kron
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, identity, csr_matrix

def knn_w(X, k=3, phi=0.5):
    """
    Compute the k-nearest neighbor weight matrix for the given data.

    Parameters:
    -----------
        X: Data matrix of shape (n_samples, p_features).
        k: Number of nearest neighbors to consider.
        phi: Scaling factor for the weights.

    Returns:
    -----------
        W: Weight matrix of shape (n_samples, n_samples).
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

def construct_Weighted_Laplacian(W):
    """
    Build the weighted Laplacian matrix from the weight matrix W.

    Parameters:
    -----------
        W: Weight matrix.
    Returns:
    -----------
        L: Weighted Laplacian matrix.
    """
    D = np.diag(np.sum(W, axis=1))
    return D - W

def built_edges(W):
    """
    Build edges and weights from the weight matrix W.

    Parameters:
    -----------
        W: Weight matrix.

    Returns:  
    -----------
        edges: List of edges.
        weights: Corresponding weights.
    """
    n = W.shape[0] # number of nodes
    edges = []
    weights = []
    for i in range(n):
        for j in range(i+1, n): # W is symmetric
            if W[i,j] > 0:
                edges.append((i,j))
                weights.append(W[i,j])

    return edges, np.array(weights, dtype=np.float64)

def compute_B_penal(W,X,gamma):
    """
    Computes the matrix B and the penalty term for the convex clustering problem on the
    RFS vertions.

    Parameters
    ----------
        W : array-like of shape (n_samples, n_samples)
            The weight matrix representing the graph structure of the data.
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        gamma : float
            The regularization parameter.

    Returns
    -------
        B : scipy.sparse.csr_matrix
            The matrix used in the regularization term of the convex clustering problem.
        penalty : float
            The penalty term associated with the regularization.
    """
    edges, weigths = built_edges(W)

    n, p = X.shape

    rows = []
    cols = []
    data = []
    for k, (i, j) in enumerate(edges):
        rows.extend([i, j])
        cols.extend([k, k])
        data.extend([1, -1])    

    b_B = coo_matrix((data, (rows, cols)), shape=(n, len(edges))).tocsr()

    I = identity(p, format='csr')
    B = kron(b_B, I, format='csr')

    penalty = gamma * np.sqrt(p)*np.sum(weigths)

    return B, penalty