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
    reference_embeddings: Optional[np.ndarray] = None,
    reference_labels: Optional[np.ndarray] = None,
    reference_head_names: Optional[List[Optional[str]]] = None,
    reference_label_suffix: str = " (centroid)",
    reference_marker: str = "X",
    reference_size: int = 80,
) -> None:
    if not save_dir:
        return
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return

    base_embeddings = np.asarray(embeddings, dtype=np.float32)
    ref_embeddings: Optional[np.ndarray] = None
    ref_labels: Optional[np.ndarray] = None
    ref_heads: Optional[List[Optional[str]]] = None

    if reference_embeddings is not None and reference_labels is not None:
        ref_embeddings = np.asarray(reference_embeddings, dtype=np.float32)
        ref_labels = np.asarray(reference_labels)
        if (
            ref_embeddings.ndim != 2
            or ref_embeddings.shape[0] == 0
            or ref_embeddings.shape[1] != base_embeddings.shape[1]
        ):
            ref_embeddings = None
            ref_labels = None
        else:
            if reference_head_names is not None:
                ref_heads = list(reference_head_names)
            else:
                ref_heads = []
            if len(ref_heads) < ref_embeddings.shape[0]:
                ref_heads = ref_heads + [head_name] * (ref_embeddings.shape[0] - len(ref_heads))
            elif len(ref_heads) > ref_embeddings.shape[0]:
                ref_heads = ref_heads[: ref_embeddings.shape[0]]

    if ref_embeddings is not None:
        tsne_inputs = np.concatenate([base_embeddings, ref_embeddings], axis=0)
    else:
        tsne_inputs = base_embeddings

    total_points = tsne_inputs.shape[0]
    perplexity = max(1, min(30, total_points - 1))
    if total_points <= 2:
        perplexity = 1

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed or 42,
        init="random",
    )
    points_2d = tsne.fit_transform(tsne_inputs)

    primary_count = base_embeddings.shape[0]
    ref_count = ref_embeddings.shape[0] if ref_embeddings is not None else 0
    primary_points = points_2d[:primary_count]
    reference_points = points_2d[primary_count : primary_count + ref_count]

    os.makedirs(save_dir, exist_ok=True)

    if point_head_names is not None:
        effective_heads: List[Optional[str]] = list(point_head_names)
    else:
        effective_heads = [head_name] * embeddings.shape[0]

    if ref_embeddings is not None:
        if ref_heads is None:
            ref_heads = [head_name] * ref_embeddings.shape[0]
    else:
        ref_heads = []

    fig, ax = plt.subplots(figsize=(8, 6))
    primary_labels = np.array(
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

    if ref_embeddings is not None and ref_labels is not None:
        reference_label_names = np.array(
            [
                format_tsne_label_name(
                    ref_labels[idx],
                    ref_heads[idx] if idx < len(ref_heads) else head_name,
                    tsne_label_mappings,
                )
                for idx in range(ref_labels.shape[0])
            ],
            dtype=object,
        )
    else:
        reference_label_names = np.array([], dtype=object)

    color_keys = np.unique(
        np.concatenate([primary_labels, reference_label_names])
        if reference_label_names.size > 0
        else primary_labels
    )
    cmap = plt.cm.get_cmap("tab20", len(color_keys) or 1)
    color_lookup = {label: idx for idx, label in enumerate(color_keys)}

    for label_name in color_keys:
        idx = color_lookup[label_name]
        mask = primary_labels == label_name
        if mask.any():
            ax.scatter(
                primary_points[mask, 0],
                primary_points[mask, 1],
                s=15,
                color=cmap(idx),
                label=str(label_name),
                alpha=0.8,
                edgecolors="none",
            )

        if reference_points.size > 0:
            ref_mask = reference_label_names == label_name
            if ref_mask.any():
                display_name = (
                    f"{label_name}{reference_label_suffix}"
                    if reference_label_suffix
                    else str(label_name)
                )
                ax.scatter(
                    reference_points[ref_mask, 0],
                    reference_points[ref_mask, 1],
                    s=reference_size,
                    color=cmap(idx),
                    label=display_name,
                    alpha=0.95,
                    marker=reference_marker,
                    edgecolors="black",
                    linewidths=0.5,
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
