# Regression
 
The `regression` module implements **forward stagewise boosting** via two
algorithms: RF-S and Fast RF-S. Both solve the same L1-regularized regression
problem — the incremental forward stagewise path, which converges to the Lasso
solution path as `step_size → 0` (Efron et al., 2004). The `algorithm`
parameter selects a computational strategy, not a different statistical model.
 
---
 
## Boosting
 
The `Boosting` class wraps `rfs_sparse` and `fastrfs_sparse` behind a
scikit-learn-compatible `fit()` / `predict()` interface.
 
```python
from convex_clustering import Boosting
import numpy as np
 
X = np.random.default_rng(0).normal(size=(100, 10))
y = X[:, 0] * 3 + X[:, 2] * -1.5 + 0.1 * np.random.default_rng(1).normal(size=100)
 
model = Boosting(algorithm="FastRFS", delta=1.0, step_size=0.01, max_iter=5000)
model.fit(X, y)
 
print(model.coef_)      # sparse coefficient vector
y_pred = model.predict(X)
```
 
**Parameters:**
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm` | `"FastRFS"` | `"RFS"` or `"FastRFS"`. Same model, different computation. |
| `delta` | `1.0` | L1 regularization strength. Must satisfy `step_size < delta`. |
| `step_size` | `0.01` | Step size for the coefficient update. Smaller → finer path. |
| `max_iter` | `1000` | Number of iterations (no early stopping). |
 
**Attributes after `fit()`:**
 
- `coef_` — sparse coefficient vector of shape `(n_features,)`.
- `n_iter_` — number of iterations run (always equals `max_iter`).
---
 
## Algorithm details
 
### RFS (Regularized Forward Stagewise)
 
At each iteration, selects the feature with the largest absolute correlation
with the current residual and takes a step of size `step_size` in that
direction. Updates the residual and the coefficient simultaneously.
 
### FastRFS (Fast Regularized Forward Stagewise)
 
Avoids recomputing the full correlation vector at each step by maintaining
a running update `gamma` that tracks correlations incrementally. Same
asymptotic path as RFS, lower per-iteration cost when `n_features` is large.
 
---
 
## Relationship to the Lasso
 
Both algorithms trace the **incremental forward stagewise path**, which
converges to the Lasso solution path as `step_size → 0`. For small
`step_size` (e.g. 0.001) and large `max_iter` (e.g. 10000), the resulting
`coef_` approximates the Lasso solution at regularization strength `delta`.
 
For an exact Lasso solution, use `sklearn.linear_model.Lasso`. `Boosting`
is most useful when you want explicit control over the stepwise path or
when the forward stagewise formulation maps naturally to your problem
(as in the RF-S convex clustering variants).
 