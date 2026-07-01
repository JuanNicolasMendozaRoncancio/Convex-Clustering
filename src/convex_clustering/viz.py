from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize


def plot_graph_weights(X: npt.NDArray[np.float64],
                       W: npt.NDArray[np.float64],
                       title: str) -> None:
    """
    Plots the graph structure of the data points with edges colored by their weights.

    Parameters:
    -----------
        X: Data points (n_samples, p_features).
        W: Weight matrix (n_samples, n_samples).
        title: Title of the plot.

    Returns:
    -----------
        None (It displays a plot with the graph structure).
    """
    _, ax = plt.subplots(figsize=(6,6))
    nonzero_W = W[W > 0]
    vmin, vmax = np.min(nonzero_W), np.max(nonzero_W)
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if W[i,j] > 0:
                ax.plot(
                    [X[i,0], X[j,0]],
                    [X[i,1], X[j,1]],
                    color=plt.cm.plasma((W[i,j]-vmin)/(vmax-vmin)),
                    linewidth=0.25+ 2.5 * (W[i,j]/vmax),
                    alpha=0.6,
                    zorder=1
                )
    ax.scatter(X[:,0], X[:,1], c='steelblue', s=80, zorder=3, edgecolors='white', linewidths=1)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Edge Weights', rotation=270, labelpad=15)
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    plt.show()

def animation_save(X: npt.NDArray[np.float64],
                   hist_: dict[Any, npt.NDArray[np.float64]],
                   f: int,
                   title: str = "Cluster Evolution",
                   save_path: str | None = None,
                   fps: int = 5,
                   Algo: str = "") -> HTML:
    """
    Creates an animation of the clustering paths and MAY save it as a GIF.

    Parameters:
    -----------
        X: Original data points (n_samples, p_features).
        hist_: Dictionary with iteration keys (it or lambda) and corresponding cluster centers.
        f: Step size for selecting frames to animate.
        save_path: Optional path to save the animation as a GIF.
        fps: Frames per second for the animation.
        Algo: Name of the algorithm (for title purposes).

    Returns:
    --------
        HTML object containing the animation.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], c='red', marker='x', label='Centers')
    ax.scatter(X[:, 0], X[:, 1],c='blue', edgecolors="gray", alpha=0.6,label='Data')
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.legend()
    ax.set_title(title)
    keys = list(hist_.keys())[::f]

    def init() -> tuple[Any, ...]:
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def update(frame: int) -> tuple[Any, ...]:
        arr = hist_[keys[frame]]
        scat.set_offsets(arr[:, :2])
        return (scat,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(keys),
        init_func=init,
        interval=200,
        blit=False,
        repeat=False
    )
    if save_path is not None:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    plt.close(fig)
    return HTML(anim.to_jshtml()) # type: ignore[no-untyped-call]
