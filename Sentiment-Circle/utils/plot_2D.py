from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric_key_prefix: str,
    head_name: Optional[str],
    save_dir: str,
    tsne_label_mappings: Optional[Dict[str, Dict[int, str]]] = None,
    point_head_names: Optional[List[Optional[str]]] = None,
    seed: int = 42,
    global_step: int = 0,
) -> None:
    if not save_dir:
        return
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return

    perplexity = max(1, min(30, embeddings.shape[0] - 1))
    if embeddings.shape[0] <= 2:
        perplexity = 1

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed or 42,
        init="random",
    )
    points_2d = tsne.fit_transform(embeddings)

    os.makedirs(save_dir, exist_ok=True)

    if point_head_names is not None:
        effective_heads: List[Optional[str]] = list(point_head_names)
    else:
        effective_heads = [head_name] * embeddings.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    display_labels = np.array(
        [
            format_tsne_label_name(
                label,
                effective_heads[idx] if idx < len(effective_heads) else head_name,
                tsne_label_mappings,
            )
            for idx, label in enumerate(labels)
        ],
        dtype=object,
    )
    unique_display = np.unique(display_labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_display) or 1)

    for idx, label_name in enumerate(unique_display):
        mask = display_labels == label_name
        ax.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            s=15,
            color=cmap(idx),
            label=str(label_name),
            alpha=0.8,
            edgecolors="none",
        )

    title_prefix = head_name if head_name is not None else "original"
    ax.set_title(f"{title_prefix} Embeddings t-SNE")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if head_name:
        legend_title = head_name
    elif any(h is not None for h in effective_heads):
        legend_title = "Emotion"
    else:
        legend_title = "Label"
    ax.legend(loc="best", fontsize="small", title=legend_title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    suffix = head_name if head_name is not None else "original"
    filename = f"{metric_key_prefix}_{suffix}_tsne_step-{global_step:06d}.png"
    save_path = os.path.join(save_dir, filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def format_tsne_label_name(
    label_value: Any,
    head_name: Optional[str],
    tsne_label_mappings: Optional[Dict[str, Dict[int, str]]] = None,
) -> str:
    if label_value is None:
        return "Unlabeled"
    if isinstance(label_value, (float, np.floating)) and not np.isfinite(label_value):
        return "Unlabeled"

    label_int: Optional[int] = None
    if isinstance(label_value, (int, np.integer)):
        label_int = int(label_value)
    elif isinstance(label_value, (float, np.floating)):
        if np.isfinite(label_value) and float(label_value).is_integer():
            label_int = int(round(float(label_value)))

    if head_name and label_int is not None:
        mapping = (tsne_label_mappings or {}).get(head_name)
        if mapping:
            label_name = mapping.get(label_int)
            if label_name is not None:
                return label_name
            return f"{head_name}:{label_int}"

    if head_name:
        return f"{head_name}:{label_value}"
    return str(label_value)


__all__ = ["plot_tsne_embedding_space", "format_tsne_label_name"]
