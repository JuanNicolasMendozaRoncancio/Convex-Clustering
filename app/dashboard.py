"""
Dashboard — Convex Clustering
  1. Interactive clustering  → calls ConvexClusterer directly (centers_hist_ access)
  2. Lasso vs Boosting       → calls the library directly
  3. Weight graph            → visualizes W as a graph (networkx + PyVis)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons, make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
API_URL: str = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Convex Clustering",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — dark palette, indigo accent
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom-color: #818cf8 !important;
    }

    .project-header {
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 1.5rem;
    }
    .project-header h1 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 0;
    }
    .project-header p {
        font-size: 0.8rem;
        color: #6b7280;
        margin: 0.2rem 0 0 0;
    }

    .api-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .api-ok   { background: #14532d; color: #86efac; }
    .api-fail { background: #450a0a; color: #fca5a5; }

    .metric-row { display: flex; gap: 1.5rem; margin: 0.8rem 0; }
    .metric-box {
        background: #1a1d27;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 0.7rem 1.1rem;
        min-width: 110px;
    }
    .metric-box .label { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-box .value { font-size: 1.4rem; font-weight: 700; color: #818cf8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="project-header">
  <h1>⬡ Convex Clustering</h1>
  <p>Juan Nicolás Mendoza Roncancio</p>
</div>
""", unsafe_allow_html=True)

# API status badge
# def _api_status() -> bool:
#     try:
#         r = requests.get(f"{API_URL}/", timeout=2)
#         return r.status_code == 200
#     except Exception:
#         return False

# api_ok = _api_status()
# badge_cls = "api-ok" if api_ok else "api-fail"
# badge_txt = f"API {API_URL} · {'online' if api_ok else 'offline'}"
# st.markdown(f'<span class="api-badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
# st.markdown("")

from convex_clustering import __version__

st.markdown(
    f'<span class="api-badge api-ok">standalone · convex-clustering v{__version__}</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_cluster, tab_boost, tab_graph = st.tabs([
    "⬡  Interactive Clustering",
    "📈  Lasso vs Boosting",
    "🕸️  Weight Graph",
])

# ==========================================================================
# TAB 1 — Interactive Clustering
# ==========================================================================
with tab_cluster:
    col_ctrl, col_plot = st.columns([1, 2], gap="large")
    st.markdown(
        "This tab allows you to run convex clustering interactively. " \
        "You can set the dataset, algorithm, and hyperparameters, then visualize the results." \
        " Nevertheless, you should be carefull selecting the hyperparameters. For instance for DR, the step size should be large " \
        "while for all the others it should be small. Also, for ADMM and AMA, you should use step sizes smaller than 1.0. " \
        "Use a great value of gamma to get a small number of clusters, and a small value to get a large number of clusters.\n"
        )

    with col_ctrl:
        st.subheader("Data")
        dataset_name = st.selectbox(
            "Dataset",
            ["blobs", "moons", "circles"],
            help="Three standard sklearn datasets with different geometric properties.",
        )
        n_samples = st.slider("n_samples", 10, 250, 200, step=10)

        st.subheader("Algorithm")
        algorithm = st.selectbox(
            "Algorithm",
            ["ADMM", "AMA", "DR", "RFS_L2", "RFS_L1", "Fast_RFS_L1"],
            index=2,  # DR default — converges faster on well-conditioned graphs
            help="All algorithms solve the same convex problem via different computational strategies.",
        )
        gamma = st.select_slider(
            "gamma (regularization)",
            options=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
            value=100,
            help="Higher gamma → fewer clusters. Controls the strength of the fusion penalty.",
        )
        k_neighbors = st.slider(
            "k neighbors (graph W)", 1, 10, 5,
            help="Number of nearest neighbors used to build the weight matrix W.",
        )
        phi = st.slider(
            "phi (weight scale)", 0.1, 2.0, 0.5, step=0.1,
            help="Scale factor in exp(-phi·d(i,j)). Higher phi → weights decay faster with distance.",
        )
        step_size = st.select_slider(
            "step_size",
            options=[0.001, 0.005, 0.01, 0.05, 0.1, 1, 1.5, 2, 5,10, 20, 50, 100,150,200,250,300,400,500,1000],
            value=0.01  ,
            help="Step size for the optimization algorithm.",
        )
        max_iter = st.select_slider(
            "max_iter",
            options=[100, 250, 500, 1000,1500,2000,2500,3000,4000,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000],
            value=500,
            help="Maximum number of iterations for the optimization algorithm.",
        )
        merge_tol = st.select_slider(
            "merge_tol",
            options=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            value=0.2,
            help="Distance threshold below which two final centers are merged into the same cluster.",
        )
        run_btn = st.button("▶  Run Clustering", use_container_width=True, type="primary")

    _ds = dataset_name.lower()
    if _ds == "blobs":
        X, y_true = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8, random_state=42)
    elif _ds == "moons":
        X, y_true = make_moons(n_samples=n_samples, noise=0.08, random_state=42)
    else:
        X, y_true = make_circles(n_samples=n_samples, noise=0.08, factor=0.5, random_state=42)

    # Build W locally to avoid an API round-trip for preprocessing
    from scipy.spatial.distance import cdist

    def _knn_w(X: np.ndarray, k: int, phi: float) -> np.ndarray:
        """KNN weight matrix — local replica of utils.knn_w for the dashboard."""
        D = cdist(X, X, "euclidean")
        np.fill_diagonal(D, np.inf)
        n = D.shape[0]
        W = np.zeros((n, n))
        for i in range(n):
            idx = np.argsort(D[i, :])
            for j in idx[:k]:
                W[i, j] = np.exp(-phi * D[i, j])
        return W

    W = _knn_w(X, k=k_neighbors, phi=phi)


    if run_btn:
        from convex_clustering import ConvexClusterer

        with st.spinner("Running algorithm…"):
            try:
                model = ConvexClusterer(
                    algorithm=algorithm,
                    gamma=gamma,
                    step_size=step_size,
                    max_iter=max_iter,
                    tol=1e-4,
                    merge_tol=merge_tol,
                )
                model.fit(X, W)

                hist = model.centers_hist_
                hist_keys_all = sorted(hist.keys())
                n_keys = len(hist_keys_all)
                percentile_indices = [
                    int(round(p * (n_keys - 1)))
                    for p in np.linspace(0, 1, 21)
                ]

                seen: set[int] = set()
                hist_keys: list[int] = []
                for idx in percentile_indices:
                    key = hist_keys_all[idx]
                    if key not in seen:
                        seen.add(key)
                        hist_keys.append(key)

                st.session_state["cluster_result"] = {
                    "X": X,
                    "y_true": y_true,
                    "labels": model.labels_,
                    "n_clusters": int(len(set(model.labels_.tolist()))),
                    "n_iter": model.n_iter_,
                    "conv": model.history_,
                    "hist": {k: hist[k] for k in hist_keys},
                    "hist_keys": hist_keys,
                    "dataset_name": dataset_name,
                    "algorithm": algorithm,
                    "gamma": gamma,
                }
                st.session_state["cluster_error"] = None

            except Exception as exc:
                st.session_state["cluster_error"] = str(exc)
                st.session_state["cluster_result"] = None

    with col_plot:
        st.subheader("Result")

        error = st.session_state.get("cluster_error")
        result = st.session_state.get("cluster_result")

        if error:
            st.error(f"Algorithm failed: {error}")

        elif result is not None:
            labels = result["labels"]
            n_clusters = result["n_clusters"]
            n_iter = result["n_iter"]
            conv = result["conv"]
            hist = result["hist"]
            hist_keys = result["hist_keys"]
            ds_label = result["dataset_name"]
            algo_label = result["algorithm"]
            gamma_label = result["gamma"]
            X_stored = result["X"]

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box"><div class="label">Clusters</div><div class="value">{n_clusters}</div></div>
              <div class="metric-box"><div class="label">Iterations</div><div class="value">{n_iter}</div></div>
              <div class="metric-box"><div class="label">Algorithm</div><div class="value" style="font-size:1rem">{algo_label}</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            if len(hist_keys) > 1:
                st.markdown("**Center trajectory** — drag the slider to watch centers fuse together")
                frame_idx = st.slider(
                    "Iteration",
                    min_value=0,
                    max_value=len(hist_keys) - 1,
                    value=len(hist_keys) - 1,
                    format="",
                    key="path_slider",
                )
                current_iter = hist_keys[frame_idx]
            else:
                # Single-frame algorithm (e.g. Fast_RFS_L2 with one gamma value):
                # no trajectory to animate, just show the final result.
                frame_idx = 0
                current_iter = hist_keys[0]
                st.caption("This algorithm does not produce an iteration-by-iteration trajectory — showing final centers.")
 
            centers_at_frame = hist[current_iter]
            st.caption(f"Iteration {current_iter} / {hist_keys[-1]}")
 
            palette = [
                "#818cf8", "#34d399", "#f472b6", "#fbbf24",
                "#60a5fa", "#a78bfa", "#fb923c", "#2dd4bf",
            ]
 
            fig_path = go.Figure()
 
            for lbl in sorted(set(labels.tolist())):
                mask = labels == lbl
                fig_path.add_trace(go.Scatter(
                    x=X_stored[mask, 0], y=X_stored[mask, 1],
                    mode="markers",
                    marker=dict(
                        color=palette[lbl % len(palette)],
                        size=7, opacity=0.4,
                        line=dict(width=0, color="#0f1117"),
                    ),
                    name=f"Cluster {lbl}",
                ))
 
            fig_path.add_trace(go.Scatter(
                x=centers_at_frame[:, 0],
                y=centers_at_frame[:, 1],
                mode="markers",
                marker=dict(
                    symbol="circle", size=9,
                    color="#ffffff", opacity=0.9,
                    line=dict(width=1, color="#818cf8"),
                ),
                name="Centers (current iter)",
            ))
 
            for i in range(len(X_stored)):
                fig_path.add_trace(go.Scatter(
                    x=[X_stored[i, 0], centers_at_frame[i, 0]],
                    y=[X_stored[i, 1], centers_at_frame[i, 1]],
                    mode="lines",
                    line=dict(color="#818cf8", width=0.6, dash="dot"),
                    showlegend=False,
                    hoverinfo="none",
                ))
 
            fig_path.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                margin=dict(l=20, r=20, t=35, b=20),
                legend=dict(bgcolor="#0f1117", borderwidth=0),
                height=400,
                title=dict(
                    text=f"{ds_label} · {algo_label} · γ={gamma_label} · iter {current_iter}",
                    font_size=13,
                ),
            )
            st.plotly_chart(fig_path, use_container_width=True, key="path_chart")
 
            if conv:
                iters_c = list(conv.keys())
                diffs_c = list(conv.values())
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=iters_c, y=diffs_c,
                    mode="lines",
                    line=dict(color="#818cf8", width=2),
                    name="‖ΔU‖",
                ))
                if current_iter in conv:
                    fig_conv.add_trace(go.Scatter(
                        x=[current_iter], y=[conv[current_iter]],
                        mode="markers",
                        marker=dict(color="#f472b6", size=10, symbol="circle"),
                        name="Selected iteration",
                    ))
                fig_conv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0f1117",
                    plot_bgcolor="#0f1117",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=200,
                    title=dict(text="Convergence — ‖U_k − U_{k-1}‖", font_size=12),
                    xaxis_title="Iteration",
                    yaxis_title="‖ΔU‖",
                    yaxis_type="log",
                )
                st.plotly_chart(fig_conv, use_container_width=True, key="conv_chart")
 
        else:
            # Dataset preview before running
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scatter(
                x=X[:, 0], y=X[:, 1],
                mode="markers",
                marker=dict(color=y_true, colorscale="Viridis", size=7,
                            opacity=0.8, showscale=False),
                name="Data points",
            ))
            fig_prev.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                margin=dict(l=20, r=20, t=30, b=20),
                height=420,
                title=dict(text=f"Preview — {dataset_name} ({n_samples} points)", font_size=13),
            )
            st.plotly_chart(fig_prev, use_container_width=True)
            st.caption("Set the parameters and press **▶ Run Clustering**.")

# ==========================================================================
# TAB 2 — Lasso vs Boosting
# ==========================================================================
with tab_boost:
    st.subheader("Lasso vs Boosting (RF-S)")
    st.markdown(
        "Both models solve the same problem Lasso regression problem"
        "via different strategies. Lasso solves it with quadratic programming. "
        "Boosting (RF-S) traces it incrementally via forward stagewise, "
        "which converges to the Lasso path as step_size tends to 0 (Efron et al., 2004)."
    )

    col_b1, col_b2 = st.columns([1, 2], gap="large")

    with col_b1:
        st.subheader("Parameters")
        n_reg = st.slider("n_samples", 50, 300, 200, step=25, key="n_reg")
        n_feat = st.slider("n_features", 10, 50, 5, key="n_feat")
        n_info = st.slider("n_informative", 1, 10, 2, key="n_info")
        noise_reg = st.slider("noise", 0.0, 10.0, 0.1, step=0.1, key="noise_reg")

        st.subheader("Boosting (RF-S)")
        boost_algo = st.radio("Variant", ["FastRFS", "RFS"], horizontal=True)
        delta = st.select_slider("delta", [100, 150, 200, 250, 300, 350, 400, 450, 500], value=100,
                                  help="L1 regularization parameter.")
        step_size = st.select_slider("step_size (epsilon)", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
        max_iter_b = st.slider("max_iter", 1000, 10000, 3000, step=200)

        st.subheader("Lasso")
        alpha_lasso = st.select_slider("alpha (sklearn)", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)

        run_boost = st.button("▶  Compare Models", use_container_width=True, type="primary", key="run_boost")

    with col_b2:
        if run_boost:
            from convex_clustering import Boosting

            X_r, y_r, coef_true = make_regression(
                n_samples=n_reg,
                n_features=n_feat,
                n_informative=n_info,
                noise=noise_reg,
                coef=True,
                random_state=42,
            )

            with st.spinner("Fitting models…"):
                import time

                # --- Boosting ---
                t0 = time.perf_counter()
                boost = Boosting(algorithm=boost_algo, delta=delta,
                                 step_size=step_size, max_iter=max_iter_b)
                boost.fit(X_r, y_r)
                t_boost = time.perf_counter() - t0

                # --- Lasso ---
                t0 = time.perf_counter()
                lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
                lasso.fit(X_r, y_r)
                t_lasso = time.perf_counter() - t0

            y_boost = boost.predict(X_r)
            y_lasso = lasso.predict(X_r)

            r2_b = r2_score(y_r, y_boost)
            r2_l = r2_score(y_r, y_lasso)
            mse_b = mean_squared_error(y_r, y_boost)
            mse_l = mean_squared_error(y_r, y_lasso)

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box">
                <div class="label">R² Boosting</div>
                <div class="value">{r2_b:.3f}</div>
              </div>
              <div class="metric-box">
                <div class="label">R² Lasso</div>
                <div class="value">{r2_l:.3f}</div>
              </div>
              <div class="metric-box">
                <div class="label">Boosting time</div>
                <div class="value" style="font-size:1rem">{t_boost:.2f}s</div>
              </div>
              <div class="metric-box">
                <div class="label">Lasso time</div>
                <div class="value" style="font-size:1rem">{t_lasso:.3f}s</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Predictions scatter
            fig_pred = go.Figure()
            lim = max(abs(y_r).max(), abs(y_boost).max(), abs(y_lasso).max()) * 1.05
            fig_pred.add_trace(go.Scatter(
                x=y_r, y=y_boost, mode="markers",
                marker=dict(color="#818cf8", size=5, opacity=0.7),
                name=f"Boosting (R²={r2_b:.3f})",
            ))
            fig_pred.add_trace(go.Scatter(
                x=y_r, y=y_lasso, mode="markers",
                marker=dict(color="#34d399", size=5, opacity=0.7),
                name=f"Lasso (R²={r2_l:.3f})",
            ))
            fig_pred.add_trace(go.Scatter(
                x=[-lim, lim], y=[-lim, lim],
                mode="lines", line=dict(color="#6b7280", width=1, dash="dot"),
                name="y = ŷ",
            ))
            fig_pred.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                margin=dict(l=20, r=20, t=35, b=20),
                height=300,
                title=dict(text="Predictions vs actual values (train)", font_size=13),
                xaxis_title="y actual",
                yaxis_title="ŷ predicted",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Coefficients
            features = [f"x{i}" for i in range(n_feat)]
            fig_coef = go.Figure()
            fig_coef.add_trace(go.Bar(
                x=features, y=boost.coef_,
                name="Boosting", marker_color="#818cf8", opacity=0.8,
            ))
            fig_coef.add_trace(go.Bar(
                x=features, y=lasso.coef_,
                name="Lasso", marker_color="#34d399", opacity=0.8,
            ))
            if coef_true is not None:
                fig_coef.add_trace(go.Scatter(
                    x=features, y=coef_true,
                    mode="markers",
                    marker=dict(symbol="diamond", color="white", size=8),
                    name="True coefficients",
                ))
            fig_coef.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                margin=dict(l=20, r=20, t=35, b=20),
                height=280,
                title=dict(text="Learned vs true coefficients", font_size=13),
                barmode="group",
                xaxis_title="Feature",
                yaxis_title="Coefficient",
            )
            st.plotly_chart(fig_coef, use_container_width=True)

        else:
            st.info("Set the parameters and press **▶ Compare Models**.", icon="📊")
            st.markdown("""
**Why this tab matters:**

- `Boosting` (RF-S) and Lasso solve the **same statistical problem** (L1-regularized regression) via different computational strategies.
- RF-S traces the **regularization path incrementally** as each iteration updates the coefficient most correlated with the current residual.
- Lasso solves it with QP. As step_size tends to 0, both converge to the same path (Efron et al., 2004).
- This tab makes that connection concrete: coefficients, R², and compute time side by side.
            """)

# ==========================================================================
# TAB 3 — Weight Graph
# ==========================================================================
with tab_graph:
    st.subheader("Weight graph W")
    st.markdown(
        "The graph structure determines which pairs of points can fuse. "
        "An edge (i, j) with high weight implies strong pressure for centers u_i and u_j to converge. "
        "If there is no edge, there is no fusion pressure between those two points."
    )

    col_g1, col_g2 = st.columns([1, 2], gap="large")

    with col_g1:
        ds_g = st.selectbox("Dataset", ["blobs", "moons", "circles"], key="g_ds")
        n_g = st.slider("n_samples", 10, 250, 200, step=5, key="g_n")
        k_g = st.slider("k neighbors", 1, 10, 5, key="g_k")
        phi_g = st.slider("phi", 0.1, 2.0, 0.5, step=0.1, key="g_phi")
        show_btn = st.button("▶  Show Graph", use_container_width=True, type="primary", key="g_btn")

    with col_g2:
        if show_btn:
            if ds_g == "blobs":
                Xg, yg = make_blobs(n_samples=n_g, centers=3, cluster_std=0.8, random_state=42)
            elif ds_g == "moons":
                Xg, yg = make_moons(n_samples=n_g, noise=0.08, random_state=42)
            else:
                Xg, yg = make_circles(n_samples=n_g, noise=0.08, factor=0.5, random_state=42)

            Wg = _knn_w(Xg, k=k_g, phi=phi_g)

            # Build networkx graph
            G = nx.Graph()
            for i in range(n_g):
                G.add_node(i, x=float(Xg[i, 0]), y=float(Xg[i, 1]), label=int(yg[i]))

            w_vals: list[float] = []
            for i in range(n_g):
                for j in range(i + 1, n_g):
                    w = max(Wg[i, j], Wg[j, i])
                    if w > 0:
                        G.add_edge(i, j, weight=w)
                        w_vals.append(w)

            w_min = min(w_vals) if w_vals else 0.0
            w_max = max(w_vals) if w_vals else 1.0

            # Try PyVis if available, fall back to static Plotly
            try:
                from pyvis.network import Network
                import streamlit.components.v1 as components

                net = Network(height="480px", width="100%", bgcolor="#0f1117",
                              font_color="white", notebook=False)
                net.toggle_physics(True)

                palette_g = ["#818cf8", "#34d399", "#f472b6", "#fbbf24",
                             "#60a5fa", "#a78bfa", "#fb923c", "#2dd4bf"]
                for node, data in G.nodes(data=True):
                    color = palette_g[data["label"] % len(palette_g)]
                    net.add_node(node, label=str(node), color=color,
                                 x=data["x"] * 80, y=-data["y"] * 80, size=12)

                for u, v, edata in G.edges(data=True):
                    w_norm = (edata["weight"] - w_min) / (w_max - w_min + 1e-9)
                    width = 0.5 + 3.5 * w_norm
                    alpha = int(60 + 160 * w_norm)
                    color_e = f"rgba(129,140,248,{alpha/255:.2f})"
                    net.add_edge(u, v, width=width, color=color_e,
                                 title=f"w = {edata['weight']:.4f}")

                net.set_options("""
                {
                  "physics": {
                    "forceAtlas2Based": {
                      "springLength": 80,
                      "springConstant": 0.04,
                      "damping": 0.4
                    },
                    "solver": "forceAtlas2Based",
                    "stabilization": {"iterations": 100}
                  }
                }
                """)

                with tempfile.TemporaryDirectory() as tmp:
                    html_path = Path(tmp) / "graph.html"
                    net.save_graph(str(html_path))
                    html_content = html_path.read_text(encoding="utf-8")

                components.html(html_content, height=500, scrolling=False)

            except ImportError:
                # Plotly static fallback
                st.caption("PyVis not available — showing static graph with Plotly.")

                pos = {i: (float(Xg[i, 0]), float(Xg[i, 1])) for i in range(n_g)}

                edge_traces = []
                for u, v, edata in G.edges(data=True):
                    w_norm = (edata["weight"] - w_min) / (w_max - w_min + 1e-9)
                    alpha = 0.15 + 0.75 * w_norm
                    edge_traces.append(go.Scatter(
                        x=[pos[u][0], pos[v][0], None],
                        y=[pos[u][1], pos[v][1], None],
                        mode="lines",
                        line=dict(
                            width=0.5 + 3.0 * w_norm,
                            color=f"rgba(129,140,248,{alpha:.2f})",
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    ))

                palette_g = ["#818cf8", "#34d399", "#f472b6", "#fbbf24"]
                node_traces = []
                for lbl in sorted(set(yg.tolist())):
                    mask_g = yg == lbl
                    node_traces.append(go.Scatter(
                        x=Xg[mask_g, 0], y=Xg[mask_g, 1],
                        mode="markers+text",
                        marker=dict(color=palette_g[lbl % len(palette_g)], size=10,
                                    line=dict(width=1, color="#0f1117")),
                        text=[str(i) for i in np.where(mask_g)[0]],
                        textposition="top center",
                        textfont=dict(size=9, color="#9ca3af"),
                        name=f"Class {lbl}",
                        hovertemplate="Node %{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
                    ))

                fig_g = go.Figure(data=edge_traces + node_traces)
                fig_g.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0f1117",
                    plot_bgcolor="#0f1117",
                    margin=dict(l=20, r=20, t=35, b=20),
                    height=480,
                    title=dict(text=f"KNN graph — k={k_g}, phi={phi_g}", font_size=13),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    showlegend=True,
                )
                st.plotly_chart(fig_g, use_container_width=True)

            # Graph statistics
            n_edges = G.number_of_edges()
            avg_degree = (2 * n_edges / n_g) if n_g > 0 else 0
            avg_w = float(np.mean(w_vals)) if w_vals else 0.0
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box"><div class="label">Nodes</div><div class="value">{n_g}</div></div>
              <div class="metric-box"><div class="label">Edges</div><div class="value">{n_edges}</div></div>
              <div class="metric-box"><div class="label">Avg degree</div><div class="value">{avg_degree:.1f}</div></div>
              <div class="metric-box"><div class="label">Avg weight</div><div class="value">{avg_w:.3f}</div></div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("Select a dataset and press **▶ Show Graph**.", icon="🕸️")
            st.markdown("""
**Why the graph matters:**

- The underlying graph is a **key factor** in the clustering behavior. If we know prior structure of our data, we can code it into the graph and leverage it for better clustering results.
- `knn_w(X, k, phi)` creates edges only to the `k` nearest neighbors, with weight `exp(-phi·d(i,j))`.
- Higher `k`, implies denser graph which also implies more pairs will fuse.
- Higher `phi` implies only immediate neighbors have real influence.
            """)