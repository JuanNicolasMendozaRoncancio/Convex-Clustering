# Here we can find the algortihms we have developed for convex clustering
# As well as the already existing algorimths.
# All in all we may find: ADMM, AMA, SSNAL, DR algorithm, 
# An RF-S algorithm and a Fast RF-S algorithm with L2 norm and L1 norm.


#Imports
import numpy as np
from scipy.sparse import kron
from scipy.sparse.linalg import factorized
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix, identity, csr_matrix


#Inner imports
from .regression import fastrfs_sparse
from .utils import construct_Weighted_Laplacian, built_edges, compute_B_penal



def ADMM(X, W, gamma, nu=1,max_iter=1000, tol=1e-5, verbose=False):
    """
    ADMM algorithm for convex clustering.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gamma : float, regularization parameter.
        nu : float, step size for the dual variable update.
        max_iter : int, maximum number of iterations.
        tol : float, tolerance for convergence.
        verbose : bool, if True, print convergence information.

    Returns
    -------
        U : array-like, shape (p, n), the final cluster centers.
        history : dict, containing the history of the difference on centers.
        U_history : dict, containing the history of the centers at each iteration. 
    """

    X = X.T
    edges, weights = built_edges(W)

    p, n = X.shape
    m = len(edges)

    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    V = np.zeros((p, m), dtype=np.float64)
    lambda_ = np.zeros((p, m), dtype=np.float64)

    E = np.zeros((n, m), dtype=np.float64)
    for l, (i, j) in enumerate(edges):
        E[i, l] = 1.0
        E[j, l] = -1.0

    lhs = np.eye(n, dtype=np.float64) + E @ E.T  

    history = {0: 0.0}
    U_hist = {}
    U = X.copy()
    U_hist[0] = U.copy()
    prev_U = U.copy()                    

    for it in range(1, max_iter + 1):
        V_bar = V + lambda_ / nu
        rhs = X + V_bar @ E.T                     
        U = np.linalg.solve(lhs, rhs.T).T        

        U_hist[it] = U.copy()

        
        diff_U = U @ E                            
        diff = diff_U - lambda_ / nu

       
        norms = np.linalg.norm(diff, axis=0)     
        tau_vec = gamma * weights / nu            
        coef = np.maximum(0., 1. - tau_vec / norms)  
        V = coef[None, :] * diff                  

        
        lambda_ += nu * (V - diff_U)

        diff_iterations = np.max(np.linalg.norm(U - prev_U, axis=0))
        history[it] = diff_iterations

        if verbose and it % 50 == 0:
            print(f"it {it:4d} | diff_centers {diff_iterations:.3e}")

        if diff_iterations < tol:
            if verbose:
                print(f"Converged at iter {it}: diff_centers={diff_iterations:.2e}")
            break

        prev_U = U.copy()

    return U, dict(list(history.items())[2:]), U_hist


def AMA(X, W, gamma, nu, max_iter=1000, tol=1e-5,verbose=False):
    """
    AMA algorithm for convex clustering.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gamma : float, regularization parameter.
        nu : float, step size for the dual variable update.
        max_iter : int, maximum number of iterations.
        tol : float, tolerance for convergence.
        verbose : bool, if True, print convergence information.

    Returns
    -------
        U_curr : array-like, shape (p, n), the final cluster centers.
        history : dict, containing the history of the difference on centers.
        Centers_history : dict, containing the history of the centers at each iteration.
    """

    X = X.T
    p, _ = X.shape

    edges, weights = built_edges(W)
    num_edges = len(edges)
    
    if num_edges == 0:
        return X.copy(), {0: 0.0}, {0: X.copy()}

    edges_array = np.array(edges, dtype=np.int32)

    r = gamma * weights                                 
    lambda_ = np.zeros((p, num_edges), dtype=np.float64)

    row_indices = np.tile(np.arange(p), num_edges)      
    col_i = np.repeat(edges_array[:, 0], p)            
    col_j = np.repeat(edges_array[:, 1], p)             

    history = {0: 0.0}
    U_prev = X.copy()
    U_curr = X.copy()


    Centers_history = {0: X.copy()}
    for it in range(1, max_iter + 1):

        Delta = np.zeros_like(X, dtype=np.float64)
        lambda_ravel = lambda_.ravel(order='F')         

        np.add.at(Delta, (row_indices, col_i), lambda_ravel)
        np.add.at(Delta, (row_indices, col_j), -lambda_ravel)

        diff_X = X[:, edges_array[:, 0]] - X[:, edges_array[:, 1]]
        diff_Delta = Delta[:, edges_array[:, 0]] - Delta[:, edges_array[:, 1]]
        g = diff_X + diff_Delta

        lambda_step = lambda_ - nu * g

        norms = np.linalg.norm(lambda_step, axis=0)
        scale_factor = np.minimum(1.0, r / np.maximum(norms, 1e-12))
        lambda_ = lambda_step * scale_factor[None, :]

        U_curr = X + Delta

        Centers_history[it] = U_curr.copy()
        if it > 10:
            diff_iterations = np.max(np.linalg.norm(U_curr - U_prev, axis=0))
            history[it] = diff_iterations

            if verbose and it % 50 == 0:
                print(f"it {it:4d} | diff_centers {diff_iterations:.3e}")
            
            if diff_iterations < tol:
                if verbose:
                    print(f"Converged at iter {it}: diff_centers={diff_iterations:.2e}")
                break
        
        U_prev = U_curr

    return U_curr, dict(list(history.items())[2:]), Centers_history


def dr_primal(X, W, gamma, rho, max_iter=1000, tol=1e-5):
    """
    DR algorithm for convex clustering.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gamma : float, regularization parameter.
        rho : float, step size for the dual variable update.
        max_iter : int, maximum number of iterations.
        tol : float, tolerance for convergence.

    Returns
    -------
        U_final : array-like, shape (p, n), the final cluster centers.
        history : dict, containing the history of the difference on centers.
        U_hist : dict, containing the history of the centers at each iteration.
    """

    n, p = X.shape

    L = csr_matrix(construct_Weighted_Laplacian(W)).tocsc()
    M = gamma * L + rho * identity(X.shape[0])
    solve = factorized(M)

    alpha = 1.0 / (1 + rho)
    beta = rho / (1 + rho)
    alpha_X = alpha * X

    U_hist_arr = np.empty((max_iter + 1, n, p), dtype=X.dtype)
    U_hist_arr[0] = X.copy()          
    rhs_buffer = np.empty_like(X)

    U_k = X.copy()
    history = {0: 0.0}

    for it in range(1, max_iter + 1):
        U_hist_arr[it][:] = alpha_X
        U_hist_arr[it] += beta * U_k
        A_new = U_hist_arr[it]                    

        np.multiply(A_new, 2.0, out=rhs_buffer)
        rhs_buffer -= U_k
        hat_A_k = solve(rhs_buffer)
        hat_A_k *= rho

        U_k -= A_new
        U_k += hat_A_k

        error = np.linalg.norm(U_hist_arr[it] - U_hist_arr[it - 1])
        history[it] = error

        if it > 10:
            if error < tol:
                print(f"Converged in {it} iterations. Error: {error:.2e}")
                break

        if it == max_iter:
            print(f"Maximum iterations reached. Final error: {error:.2e}")


    num_stored = it + 1
    U_hist = {kk: U_hist_arr[kk] for kk in range(num_stored)}

    U_final = U_hist_arr[it]

    return U_final,  dict(list(history.items())[2:]), U_hist


def centers_rfs_l2(X, W, gamma=10, epsilon=0.01, numiter=5000):
    """
    Applies the FRFS algorithm for convex clustering.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gamma : float, regularization parameter.
        epsilon : float, step size for the primal variable update.
        numiter : int, maximum number of iterations.

    Returns
    -------
        U_final: array-like, shape (p, n), the final cluster centers.
        history : dict, containing the history of the difference on centers.
        Centers_hist : dict, containing the history of the centers at each iteration.
    """
    
    
    n, p = X.shape

    _, penalty = compute_B_penal(W, X, gamma)
    edges, _ = built_edges(W)
    num_edges = len(edges)

    if num_edges == 0 or penalty == 0:
        return X.copy(), history, centers_hist

    mean_all = np.mean(X)
    X_c = X - mean_all
    factor = 1.0 - epsilon / penalty

    edges_arr = np.array(edges, dtype=np.int32)
    gamma_mat = (X_c[edges_arr[:, 0]] - X_c[edges_arr[:, 1]]) / np.sqrt(2.0)
    alpha_mat = (epsilon / penalty) * gamma_mat

    incident = [[] for _ in range(n)]
    for k in range(num_edges):
        i, j = edges[k]
        incident[i].append((k, 1.0))
        incident[j].append((k, -1.0))

    incident_edges = []
    incident_signs = []
    for inc in incident:
        if inc:
            ks, sgns = zip(*inc)
            incident_edges.append(np.array(ks, dtype=np.int32))
            incident_signs.append(np.array(sgns, dtype=np.float64))
        else:
            incident_edges.append(np.empty(0, dtype=np.int32))
            incident_signs.append(np.empty(0, dtype=np.float64))

    shift = (1.0 - factor) * X
    abs_gamma = np.empty_like(gamma_mat)

    centers = np.copy(X)
    centers_hist_arr = np.empty((numiter, n, p), dtype=np.float64)

    history = {0: 0.0}
    for k in range(numiter):
        np.abs(gamma_mat, out=abs_gamma)
        idx_flat = np.argmax(abs_gamma)
        l = idx_flat // p
        g = idx_flat % p
        s_k = np.sign(gamma_mat[l, g])

        i, j = edges_arr[l]

        gamma_mat *= factor
        gamma_mat += alpha_mat

        eps_s_half = epsilon * s_k / 2.0
        gamma_mat[incident_edges[i], g] -= eps_s_half * incident_signs[i]
        gamma_mat[incident_edges[j], g] += eps_s_half * incident_signs[j]

        centers *= factor
        centers += shift
        delta_c = epsilon * s_k / np.sqrt(2.0)
        centers[i, g] -= delta_c
        centers[j, g] += delta_c

        error = np.linalg.norm(centers - centers_hist_arr[k - 1]) if k > 0 else 0
        history[k] = error

        centers_hist_arr[k] = centers



    num_stored = k + 1
    centers_hist = {kk: centers_hist_arr[kk] for kk in range(num_stored)}

    U_final = centers_hist_arr[k]

    return U_final, dict(list(history.items())[2:]), centers_hist


def centers_fast_rfs_l2(X,W,gammas=None, epsilon = 0.01, numiter = 10000):
    """
    Applies the Fast RF-S algorithm for convex clustering.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gammas : list, regularization parameters.
        epsilon : float, step size for the primal variable update.
        numiter : int, maximum number of iterations.

    Returns
    -------
        U_final: array-like, shape (p, n), the final cluster centers.
        History: dict, containing the history of the objective function values at each iteration.
        Center_hist: dict, containing the history of the centers at each iteration.
    """

    if gammas is None:
        gammas = [10]
    
    center_hist = {0: X.copy()}
    history = {}
    for i, gamma in enumerate(gammas):
            
        B, penalty = compute_B_penal(W, X, gamma)
        x = X.reshape(-1)

        b = fastrfs_sparse(B, x, penalty, epsilon=epsilon, numiter=numiter)

        Bnorm = normalize(B, norm='l2', axis=0)

        centers = X - (Bnorm @ b).reshape(X.shape[0], X.shape[1])

        center_hist[i] = centers
        history[i] = np.linalg.norm(center_hist[i] - center_hist[i-1]) if i > 0 else 0.0

    U_final = list(center_hist.values())[-1]

    return U_final, dict(list(history.items())[2:]), center_hist


def centers_rfs_l1(X, W, gamma, epsilon=0.01, cauchy=1e-5, M=1000):
    """
    Applies the RF-S algorithm for convex clustering with L1 norm.

    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gamma : float, regularization parameter.
        epsilon : float, step size for the primal variable update.
        cauchy : float, convergence threshold for the centers.
        M : int, maximum number of iterations.

    Returns
    -------
        U_final: array-like, shape (p, n), the final cluster centers.
        history : dict, containing the history of the difference on centers.
        U_hist : dict, containing the history of the centers at each iteration.
    """
    
    n, p = X.shape

    edges, weights = built_edges(W)
    q = len(edges)

    rows = []
    cols = []
    data = []
    for k, (i, j) in enumerate(edges):
        rows.extend([i, j])
        cols.extend([k, k])
        data.extend([1.0, -1.0])
    b_B = coo_matrix((data, (rows, cols)), shape=(n, q)).tocsr()

    b_B_T = b_B.T

    X = np.asarray(X, dtype=float)
    R = X.copy()                    
    Beta_mat = np.zeros((q, p))   

    U_hist = {0: X.copy()}
    history = {0: 0.0}

    for m in range(M):
        edge_grad = b_B_T @ R                          
        G_mat = weights[:, None] * np.sign(edge_grad) 

        term1 = b_B @ G_mat
        term2 = (1.0 / gamma) * (R - X)
        R -= epsilon * (term1 + term2)

        Beta_mat += (-epsilon / gamma) * Beta_mat + epsilon * G_mat

        centers = X - b_B @ Beta_mat

        U_hist[m + 1] = centers.copy()

        if m > 0:
            U_diff1 = np.mean(np.linalg.norm(centers - U_hist[m], axis=0))
            U_diff2 = np.linalg.norm(centers - U_hist[m], ord='fro')
            history[m + 1] = U_diff1
            if (U_diff1 < cauchy) or (U_diff2 < cauchy):
                print(f"Convergencia alcanzada en la iteración {m+1}")
                break

    final_centers = list(U_hist.values())[-1]
    return final_centers, dict(list(history.items())[2:]), U_hist


def centers_fast_rfs_l1(X, W, gamma, epsilon=0.01, cauchy=1e-5, M=1000):
    """
    Applies the Fast RF-S algorithm for convex clustering with L1 norm.
    
    Parameters
    ----------
        X : array-like, shape (p, n), data matrix with p features and n samples.
        W : array-like, shape (n, n), matrix with the weights of our garph.
        gamma : float, regularization parameter.
        epsilon : float, step size for the primal variable update.
        cauchy : float, convergence threshold for the centers.
        M : int, maximum number of iterations.

    Returns
    -------
        U_final: array-like, shape (p, n), the final cluster centers.
        History: dict, containing the history of the difference on centers.
        U_hist : dict, containing the history of the centers at each iteration.    
    """

    n, p = X.shape
  
    edges, weights = built_edges(W)

    num_edges = len(edges)
    rows = np.empty(2 * num_edges, dtype=np.int32)
    cols = np.empty(2 * num_edges, dtype=np.int32)
    data = np.empty(2 * num_edges, dtype=np.float64)
    for k in range(num_edges):
        i, j = edges[k]
        idx = 2 * k
        rows[idx] = i
        rows[idx + 1] = j
        cols[idx] = k
        cols[idx + 1] = k
        data[idx] = 1.0
        data[idx + 1] = -1.0

    b_B = coo_matrix((data, (rows, cols)), shape=(n, num_edges)).tocsr()

    I = identity(p, format='csr')
    B = kron(b_B, I, format='csr')

 
    D = np.diag(weights)
    W_kron = kron(D, I, format='csr')

    x = X.flatten()

    beta = np.zeros(B.shape[1])
    N = W_kron @ B.T @ B @ W_kron
    w = W_kron @ B.T @ x
    y_mono = W_kron @ B.T @ x

    U_hist = {0: X.copy()}
    history = {0: 0.0}

    eps_gamma = epsilon / gamma

    for m in range(M):
        w += -epsilon * (N @ np.sign(w)) - eps_gamma * (w - y_mono)
        beta += -eps_gamma * beta + epsilon * (W_kron @ np.sign(w))

        beta_mat = beta.reshape(num_edges, p)
        centers = X - (b_B @ beta_mat)

        U_hist[m + 1] = centers   # sin copia extra (ya es un array nuevo)

        if m > 0:
            diff = U_hist[m + 1] - U_hist[m]   # calculamos la diferencia solo una vez
            U_diff1 = np.max(np.linalg.norm(diff, axis=1))
            U_diff2 = np.linalg.norm(diff, ord='fro')
            history[m + 1] = U_diff1
            if (U_diff1 < cauchy) or (U_diff2 < cauchy):
                print(f"Convergencia alcanzada en la iteración {m + 1}")
                break

    U_final = list(U_hist.values())[-1]

    return U_final, dict(list(history.items())[2:]), U_hist


class ConvexClusterer(BaseEstimator, ClusterMixin):
    """
    Convex clustering via ADMM, AMA, Douglas-Rachford, or RF-S variants.
 
    Unified sklearn-compatible facade over seven convex clustering algorithms.
    Implements fit() and fit_predict() following the scikit-learn estimator API,
    enabling use in GridSearchCV and Pipeline (with precomputed W).
 
    Parameters
    ----------
    algorithm : str, default='ADMM'
        Algorithm to use. One of:
        'ADMM', 'AMA', 'DR', 'RFS_L2', 'FastRFS_L2', 'RFS_L1', 'FastRFS_L1'.
    gamma : float, default=1.0
        Regularization parameter controlling the strength of the fusion penalty.
        Higher gamma → fewer, larger clusters.
    step_size : float, default=1.0
        Step size for the iterative update. Maps to:
        nu (ADMM, AMA), rho (DR), epsilon (RFS variants).
    max_iter : int, default=1000
        Maximum number of iterations. Maps to:
        max_iter (ADMM, AMA, DR), numiter (RFS_L2, FastRFS_L2), M (RFS_L1, FastRFS_L1).
    tol : float, default=1e-5
        Convergence tolerance. Maps to:
        tol (ADMM, AMA, DR), cauchy (RFS_L1, FastRFS_L1).
        Not used by RFS_L2 and FastRFS_L2, which run for exactly max_iter steps.
    verbose : bool, default=False
        If True, print convergence information at every 50 iterations.
        Only supported by ADMM and AMA; ignored by other algorithms.
    merge_tol : float, default=1e-3
        Distance threshold below which two final centers are considered
        the same cluster when extracting labels.
 
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label for each point, extracted via connected components
        on the fused center distances.
    cluster_centers_ : ndarray of shape (n_samples, n_features)
        Final cluster centers. Points in the same cluster share the same center.
    history_ : dict of {int: float}
        Convergence history mapping iteration index to the center difference value.
    centers_history_ : dict of {int: ndarray}
        Full trajectory of centers at each iteration.
    n_iter_ : int
        Number of iterations actually run.
 
    Examples
    --------
    >>> model = ConvexClusterer(algorithm='ADMM', gamma=0.5)
    >>> model.fit(X, W)
    >>> print(model.labels_)
 
    >>> # sklearn Pipeline with precomputed W
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('clusterer', ConvexClusterer())])
    >>> pipe.fit(X, W=W)
    """
    _ALGORITHMS = frozenset({
        "ADMM","AMA","DR","RFS_L2","Fast_RFS_L2","RFS_L1","Fast_RFS_L1"
    })

    def __init__(
            self,
            algorithm= "ADMM",
            gamma = 1,
            step_size = 0.01,
            max_iter = 1000,
            tol = 1e-5,
            verbose = False,
            merge_tol = 1e-5
    ):
        self.algorithm = algorithm
        self.gamma = gamma
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.merge_tol = merge_tol


    def fit(self, X, W, y =None):
        """
        Fit the convex clustering model.
 
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix. Convention follows the algorithm implementations:
            rows are samples, columns are features.
        W : ndarray of shape (n_samples, n_samples)
            Symmetric weight matrix encoding the graph structure.
            W[i, j] > 0 means there is an edge between points i and j.
        y : ignored
            Not used. Present for sklearn API compatibility.
 
        Returns
        -------
        self : ConvexClusterer
            Fitted estimator.
 
        Raises
        ------
        ValueError
            If algorithm is not one of the supported options.
        """
        if self.algorithm not in self._ALGORITHMS:
            raise ValueError(f"Algorithm {self.algorithm} not recognized. Available algorithms: {self._ALGORITHMS}")
        
        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)

        U_final, history, centers_hist = self._run_experiment(X, W)

        self.cluster_centers_ = U_final
        self.history_ = history
        self.centers_hist_ = centers_hist
        self.n_iter_ = len(history)
        self.labels_ = self._extract_labels(U_final)

        return self
    
    def _run_experiment(self, X, W):
        """
        Dispatch to the selected algorithm with unified parameter mapping.
 
        All algorithms return (U_final, history, centers_history).
        """
        algo = self.algorithm

        if algo == "ADMM":
            return ADMM(X, W, 
                        gamma = self.gamma, 
                        nu = self.step_size, 
                        max_iter = self.max_iter, 
                        tol = self.tol, 
                        verbose = self.verbose)
        elif algo == "AMA":
            return AMA(X, W, 
                        gamma = self.gamma, 
                        nu = self.step_size, 
                        max_iter = self.max_iter, 
                        tol = self.tol, 
                        verbose = self.verbose)
        elif algo == "DR":
            return dr_primal(X, W, 
                            gamma = self.gamma, 
                            rho = self.step_size, 
                            max_iter = self.max_iter, 
                            tol = self.tol)
        elif algo == "RFS_L2":
            return centers_rfs_l2(X, W, 
                                gamma = self.gamma, 
                                epsilon = self.step_size, 
                                numiter = self.max_iter)
        elif algo == "Fast_RFS_L2":
            gammas = self.gamma if isinstance(self.gamma, list) else [self.gamma]
            return centers_fast_rfs_l2(X, W, 
                                        gammas = gammas, 
                                        epsilon = self.step_size, 
                                        numiter = self.max_iter)
        elif algo == "RFS_L1":
            return centers_rfs_l1(X, W, 
                                gamma = self.gamma, 
                                epsilon = self.step_size, 
                                cauchy = self.tol, 
                                M = self.max_iter)
        elif algo == "Fast_RFS_L1":
            return centers_fast_rfs_l1(X, W, 
                                        gamma = self.gamma, 
                                        epsilon = self.step_size, 
                                        cauchy = self.tol, 
                                        M = self.max_iter)
        
    def fit_predict(self, X, W, y=None):
        """
        Fit and return cluster labels.
 
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        W : ndarray of shape (n_samples, n_samples)
        y : ignored
 
        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(X,W).labels_
    
        
    def _extract_labels(self, U_final):
        """
        Extract cluster labels from final centers.
 
        Two points are assigned the same label when the Euclidean distance
        between their final centers is below merge_tol. Labels are computed
        via connected components on the resulting adjacency graph.
 
        Why connected components and not argmin: in convex clustering, fused
        centers are exactly equal (up to numerical precision) — there is no
        notion of "closest centroid" as in k-means. Connected components
        correctly handles chains of fused points without a fixed k.
 
        Parameters
        ----------
        U_final : ndarray of shape (n_samples, n_features)
 
        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
        """

        dist = squareform(pdist(U_final))
        adjacency = (dist < self.merge_tol).astype(np.float64)
        np.fill_diagonal(adjacency, 0.0)
        
        _, labels = connected_components(
                    csr_matrix(adjacency), directed=False
                )
        return labels

