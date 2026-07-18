"""
Generate static figures for docs/applications/customer_segmentation.md.
 
Outputs written to docs/applications/img/:
  clusters_final.png      — final cluster assignment in PCA 2D space
  fusion_trajectory.gif   — animation of center fusion across iterations
 
Run AFTER run_rfm_analysis.py (which generates results/rfm/v2_fusion_path.npy).
 
Usage:
    python scripts/generate_rfm_figures.py
"""
from __future__ import annotations
 
import numpy as np
import numpy.typing as npt
from pathlib import Path
 
_DATA_DIR    = Path(__file__).parent.parent / "data"
_IMG_DIR     = Path(__file__).parent.parent / "docs" / "applications" / "img"
 
# Segment labels in PC1 order (low PC1 = At-Risk, high PC1 = Champions)
_SEG_NAMES  = {0: "Regulars", 1: "Champions", 2: "At-Risk"}
_PALETTE     = ["#818cf8", "#34d399", "#f472b6"]
_BG_COLOR    = "#0f1117"
_TEXT_COLOR  = "#e0e0e0"
_MUTED_COLOR = "#9ca3af"
_SPINE_COLOR = "#1f2937"
 
 
def _load_artifacts() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.intp],
    npt.NDArray[np.float64],
    npt.NDArray[np.intp],
]:
    """Load clustering results produced by run_rfm_analysis.py."""
    required = [
        _DATA_DIR / "X_sub_2d.npy",
        _DATA_DIR / "U_final_2d.npy",
        _DATA_DIR / "labels_2d.npy",
        _DATA_DIR / "fusion_path.npy",
        _DATA_DIR / "hist_keys.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {missing}\n"
            "Run scripts/run_rfm_analysis.py first."
        )
    return (
        np.load(_DATA_DIR / "X_sub_2d.npy"),
        np.load(_DATA_DIR / "U_final_2d.npy"),
        np.load(_DATA_DIR / "labels_2d.npy"),
        np.load(_DATA_DIR / "fusion_path.npy"),   # (n_frames, n_points, 2)
        np.load(_DATA_DIR / "hist_keys.npy"),
    )
 
 
def _apply_dark_style(ax, fig) -> None:  # type: ignore[no-untyped-def]
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)
    ax.tick_params(colors="#6b7280", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
 
 
def generate_static_png(
    X: npt.NDArray[np.float64],
    U: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
    out_path: Path,
) -> None:
    """
    Scatter plot of the final cluster assignment in PCA 2D space.
 
    Each point is a customer coloured by cluster label. White stars mark
    the mean center of each cluster in the fused space.
    """
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _apply_dark_style(ax, fig)
 
    for lbl in sorted(set(labels.tolist())):
        mask = labels == lbl
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=_PALETTE[lbl % len(_PALETTE)],
            s=22, alpha=0.55, linewidths=0,
            label=_SEG_NAMES.get(lbl, f"Cluster {lbl}"),
        )
 
    # Mean center per cluster
    unique_centers = np.array([
        U[labels == lbl].mean(axis=0)
        for lbl in sorted(set(labels.tolist()))
    ])
    ax.scatter(
        unique_centers[:, 0], unique_centers[:, 1],
        c="white", s=130, marker="*", zorder=5,
        edgecolors="#818cf8", linewidths=0.8, label="Cluster center",
    )
 
    ax.set_xlabel(
        "PC1  (Recency ↔ Frequency + Monetary)",
        color=_MUTED_COLOR, fontsize=9,
    )
    ax.set_ylabel("PC2  (Recency dominance)", color=_MUTED_COLOR, fontsize=9)
    ax.set_title(
        "Customer Segmentation — Convex Clustering (PCA 2D)",
        color=_TEXT_COLOR, fontsize=11, pad=10,
    )
    ax.legend(
        framealpha=0.15, facecolor="#1a1d27",
        edgecolor=_SPINE_COLOR, labelcolor=_TEXT_COLOR, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close()
    print(f"PNG saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
 
 
def generate_fusion_gif(
    X: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
    path: npt.NDArray[np.float64],
    keys: npt.NDArray[np.intp],
    out_path: Path,
    fps: int = 1,
    skip: int = 1,
) -> None:
    """
    Animated GIF of the center fusion trajectory.
 
    White dots represent the center assigned to each customer at iteration k.
    As gamma increases (iterations progress), nearby centers are pulled together
    until the three final clusters emerge.
 
    Parameters
    ----------
    skip : int
        Use every `skip`-th frame to reduce GIF file size.
    fps : int
        Frames per second for the output GIF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
 
    frame_idx      = list(range(0, len(path), skip)) + [len(path) - 1]
    frames_centers = [path[i] for i in frame_idx]
    frames_iters   = [keys[i] for i in frame_idx]
    n_frames       = len(frames_centers)
    print(f"GIF: {n_frames} frames at {fps} fps ...")
 
    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_style(ax, fig)
 
    # Static background: data points coloured by final label
    for lbl in sorted(set(labels.tolist())):
        mask = labels == lbl
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=_PALETTE[lbl % len(_PALETTE)],
            s=14, alpha=0.25, linewidths=0,
        )
 
    scat      = ax.scatter([], [], c="white", s=8, alpha=0.85, zorder=4, linewidths=0)
    iter_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        color=_MUTED_COLOR, fontsize=8, va="top",
    )
 
    margin = 0.3
    ax.set_xlim(X[:, 0].min() - margin, X[:, 0].max() + margin)
    ax.set_ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)
    ax.set_xlabel("PC1", color=_MUTED_COLOR, fontsize=8)
    ax.set_ylabel("PC2", color=_MUTED_COLOR, fontsize=8)
    ax.set_title(
        "Fusion trajectory — centers converging",
        color=_TEXT_COLOR, fontsize=10,
    )
    fig.tight_layout()
 
    def init():  # type: ignore[return]
        scat.set_offsets(np.empty((0, 2)))
        iter_text.set_text("")
        return scat, iter_text
 
    def update(frame: int):  # type: ignore[return]
        scat.set_offsets(frames_centers[frame])
        iter_text.set_text(f"iter {frames_iters[frame]}")
        return scat, iter_text
 
    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=1000 // fps, blit=True,
    )
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"GIF saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
 
 
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
 
    _IMG_DIR.mkdir(parents=True, exist_ok=True)
 
    X, U, labels, fusion_path, hist_keys = _load_artifacts()
 
    generate_static_png(
        X, U, labels,
        out_path=_IMG_DIR / "clusters_final.png",
    )
    generate_fusion_gif(
        X, labels, fusion_path, hist_keys,
        out_path=_IMG_DIR / "fusion_trajectory.gif",
        fps=2, skip=100,
    )
    print("\nAll figures generated. Commit docs/applications/img/ with the rest.")
 
 
if __name__ == "__main__":
    main()