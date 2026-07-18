"""
Generate figures for docs/applications/customer_segmentation.md.

Produces 5 outputs in docs/applications/img/:
  clusters_pca.png        — V2: final clusters in PCA 2D space
  clusters_v3_umap.png    — V3: 3D clustering labels projected with UMAP
  clusters_v4_umap.png    — V4: UMAP → clustering result
  fusion_pca.gif          — V2: center fusion animation in PCA space
  fusion_umap.gif         — V4: center fusion animation in UMAP space

Run AFTER run_rfm_analysis.py (which populates data/).

Usage:
    python scripts/generate_rfm_figures.py
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent.parent / "data"
_IMG_DIR  = Path(__file__).parent.parent / "docs" / "applications" / "img"

# Cluster 0 = Regulars, 1 = Champions, 2 = At-Risk
_SEG_NAMES  = {0: "Regulars", 1: "Champions", 2: "At-Risk"}
_PALETTE    = ["#818cf8", "#f472b6", "#34d399"]   # indigo, pink, teal
_BG         = "#0f1117"
_TEXT       = "#e0e0e0"
_MUTED      = "#9ca3af"
_SPINE      = "#1f2937"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_arrays() -> dict[str, npt.NDArray[Any]]:
    """Load all arrays produced by run_rfm_analysis.py."""
    keys = [
        "X_sub_2d", "U_final_2d", "labels_2d", "fusion_path", "hist_keys",
        "X_sub_umap", "U_final_umap", "labels_umap",
        "fusion_path_umap", "hist_keys_umap",
        "labels_v1",
    ]
    missing = [k for k in keys if not (_DATA_DIR / f"{k}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing arrays in data/: {missing}\n"
            "Run scripts/run_rfm_analysis.py first."
        )
    return {k: np.load(_DATA_DIR / f"{k}.npy") for k in keys}


def _dark_ax(ax: Any, fig: Any) -> None:
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors="#6b7280", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)


def _scatter_clusters(
    ax: Any,
    X: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
    centers: npt.NDArray[np.float64] | None = None,
    alpha: float = 0.55,
    s: int = 22,
) -> None:
    """Plot coloured scatter with optional cluster-center stars."""
    for lbl in sorted(set(labels.tolist())):
        mask = labels == lbl
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=_PALETTE[lbl % len(_PALETTE)], s=s, alpha=alpha,
            linewidths=0, label=_SEG_NAMES.get(lbl, f"Cluster {lbl}"),
        )
    if centers is not None:
        ax.scatter(
            centers[:, 0], centers[:, 1],
            c="white", s=130, marker="*", zorder=5,
            edgecolors="#818cf8", linewidths=0.8, label="Cluster center",
        )


def _mean_centers(
    X: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    return np.array([
        X[labels == lbl].mean(axis=0)
        for lbl in sorted(set(labels.tolist()))
    ])


def _save_legend(ax) -> None:  # type: ignore[no-untyped-def]
    ax.legend(
        framealpha=0.15, facecolor="#1a1d27",
        edgecolor=_SPINE, labelcolor=_TEXT, fontsize=8,
    )


# ---------------------------------------------------------------------------
# Figure 1 — V2: PCA 2D clustering result
# ---------------------------------------------------------------------------

def fig_clusters_pca(
    X_pca: npt.NDArray[np.float64],
    U_pca: npt.NDArray[np.float64],
    labels_pca: npt.NDArray[np.intp],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _dark_ax(ax, fig)

    centers = _mean_centers(U_pca, labels_pca)
    _scatter_clusters(ax, X_pca, labels_pca, centers=centers)

    ax.set_xlabel("PC1  (Recency ↔ Frequency + Monetary)", color=_MUTED, fontsize=9)
    ax.set_ylabel("PC2  (Recency dominance)", color=_MUTED, fontsize=9)
    ax.set_title("V2 — Customer Segmentation (PCA 2D, DR clustering)",
                 color=_TEXT, fontsize=11, pad=10)
    _save_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Figure 2 — V3: 3D labels projected into UMAP
# ---------------------------------------------------------------------------

def fig_clusters_v3_umap(
    X_umap: npt.NDArray[np.float64],
    labels_v1: npt.NDArray[np.intp],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _dark_ax(ax, fig)

    centers = _mean_centers(X_umap, labels_v1)
    _scatter_clusters(ax, X_umap, labels_v1, centers=centers)

    ax.set_xlabel("UMAP 1", color=_MUTED, fontsize=9)
    ax.set_ylabel("UMAP 2", color=_MUTED, fontsize=9)
    ax.set_title("V3 — 3D Clustering Labels Projected with UMAP\n"
                 "(clustering in 3D RFM space, UMAP for visualization only)",
                 color=_TEXT, fontsize=10, pad=10)
    _save_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Figure 3 — V4: UMAP → clustering result
# ---------------------------------------------------------------------------

def fig_clusters_v4_umap(
    X_umap: npt.NDArray[np.float64],
    U_umap: npt.NDArray[np.float64],
    labels_umap: npt.NDArray[np.intp],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _dark_ax(ax, fig)

    centers = _mean_centers(U_umap, labels_umap)
    _scatter_clusters(ax, X_umap, labels_umap, centers=centers)

    ax.set_xlabel("UMAP 1", color=_MUTED, fontsize=9)
    ax.set_ylabel("UMAP 2", color=_MUTED, fontsize=9)
    ax.set_title("V4 — Customer Segmentation (UMAP 2D, DR clustering)",
                 color=_TEXT, fontsize=11, pad=10)
    _save_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# GIF helper — shared by V2 and V4
# ---------------------------------------------------------------------------

def _make_fusion_gif(
    X: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
    path: npt.NDArray[np.float64],      # (n_frames, n_points, 2)
    keys: npt.NDArray[np.intp],         # iteration index per frame
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    fps: int = 8,
    skip: int = 2,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frame_idx = list(range(0, len(path), skip)) + [len(path) - 1]
    # deduplicate while preserving order
    seen: set[int] = set()
    frame_idx = [i for i in frame_idx if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]
    frames_centers = [path[i] for i in frame_idx]
    frames_iters   = [int(keys[i]) for i in frame_idx]
    print(f"GIF ({out_path.name}): {len(frames_centers)} frames at {fps} fps ...")

    fig, ax = plt.subplots(figsize=(6, 5))
    _dark_ax(ax, fig)

    # Static background: final cluster assignment
    for lbl in sorted(set(labels.tolist())):
        mask = labels == lbl
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=_PALETTE[lbl % len(_PALETTE)], s=14, alpha=0.25, linewidths=0)

    scat = ax.scatter([], [], c="white", s=8, alpha=0.85, zorder=4, linewidths=0)
    txt  = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                   color=_MUTED, fontsize=8, va="top")

    margin = 0.4
    ax.set_xlim(X[:, 0].min() - margin, X[:, 0].max() + margin)
    ax.set_ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=8)
    ax.set_title(title, color=_TEXT, fontsize=10)
    fig.tight_layout()

    def init() -> tuple[Any, Any]:
        scat.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return scat, txt

    def update(frame: int) -> tuple[Any, Any]:
        scat.set_offsets(frames_centers[frame])
        txt.set_text(f"iter {frames_iters[frame]}")
        return scat, txt

    anim = FuncAnimation(fig, update, frames=len(frames_centers),
                         init_func=init, interval=1000 // fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    d = _load_arrays()

    # ── 3 static PNGs ──────────────────────────────────────────────────────
    fig_clusters_pca(
        X_pca      = d["X_sub_2d"],
        U_pca      = d["U_final_2d"],
        labels_pca = d["labels_2d"],
        out_path   = _IMG_DIR / "clusters_pca.png",
    )

    fig_clusters_v3_umap(
        X_umap    = d["X_sub_umap"],
        labels_v1 = d["labels_v1"],
        out_path  = _IMG_DIR / "clusters_v3_umap.png",
    )

    fig_clusters_v4_umap(
        X_umap      = d["X_sub_umap"],
        U_umap      = d["U_final_umap"],
        labels_umap = d["labels_umap"],
        out_path    = _IMG_DIR / "clusters_v4_umap.png",
    )

    # ── 2 GIFs ─────────────────────────────────────────────────────────────
    _make_fusion_gif(
        X       = d["X_sub_2d"],
        labels  = d["labels_2d"],
        path    = d["fusion_path"],
        keys    = d["hist_keys"],
        out_path= _IMG_DIR / "fusion_pca.gif",
        xlabel  = "PC1", ylabel = "PC2",
        title   = "Fusion trajectory — PCA 2D (DR)",
        fps=8, skip=1,
    )

    _make_fusion_gif(
        X       = d["X_sub_umap"],
        labels  = d["labels_umap"],
        path    = d["fusion_path_umap"],
        keys    = d["hist_keys_umap"],
        out_path= _IMG_DIR / "fusion_umap.gif",
        xlabel  = "UMAP 1", ylabel = "UMAP 2",
        title   = "Fusion trajectory — UMAP 2D (DR)",
        fps=8, skip=1,
    )

    print("\nAll 5 figures generated.")
    print("Commit docs/applications/img/ together with the rest of step 20.")


if __name__ == "__main__":
    main()