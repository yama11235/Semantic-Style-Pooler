from dataclasses import dataclass, field
from scipy.stats import spearmanr, pearsonr
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    adjusted_mutual_info_score,
    v_measure_score,
)
from itertools import combinations
import math
from typing import Union, List, Dict, Optional, Callable


@dataclass
class HeadContext:
    head: str
    objective: str
    scores: Optional[np.ndarray] = None
    pos_scores: Optional[np.ndarray] = None
    neg_scores: Optional[np.ndarray] = None
    anchor_probs: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    active_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    centroids: Optional[Dict[int, np.ndarray]] = None

def flatten_floats(
    data: Union[List[float], List[List[float]]]
) -> List[float]:
    """
    - data が List[float] の場合はそのまま返す
    - data が List[List[float]] の場合は一段だけ展開して List[float] を返す
    - None や非 float の要素は無視します
    - NaN (math.isnan) は除外します
    """
    if not data:
        return []
    # 全要素が float かつ NaN でないならそのまま返す
    if all(isinstance(elem, float) and not math.isnan(elem) for elem in data):
        return data  # type: ignore

    flattened: List[float] = []
    for elem in data:
        if isinstance(elem, list):
            # リストなら中身をチェックして追加
            for x in elem:
                if isinstance(x, float) and not math.isnan(x):
                    flattened.append(x)
        else:
            # float なら追加（その他は無視）
            if isinstance(elem, float) and not math.isnan(elem):
                flattened.append(elem)

    return flattened


def _to_float_array(value: Optional[Union[np.ndarray, List[float]]]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr


def _build_head_vector(*arrays: Optional[np.ndarray]) -> np.ndarray:
    cleaned: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=float).flatten()
        flat = flat[~np.isnan(flat)]
        if flat.size > 0:
            cleaned.append(flat)
    if cleaned:
        return np.concatenate(cleaned, axis=0)
    return np.array([], dtype=float)


def _compute_regression_metrics(ctx: HeadContext) -> Dict[str, float]:
    if ctx.scores is None or ctx.labels.size == 0:
        return {}
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    scores = ctx.scores[mask].astype(float)
    truths = ctx.labels[mask].astype(float).flatten()
    if scores.size == 0:
        return {}

    mse = mean_squared_error(truths, scores)
    pearson = pearsonr(scores, truths)[0] if scores.size > 1 else 0.0
    if scores.size > 1:
        spearman = spearmanr(scores, truths).correlation
    else:
        spearman = 0.0
    return {
        f"{ctx.head}_mse": float(mse),
        f"{ctx.head}_single-pearson": float(pearson),
        f"{ctx.head}_single-spearman": float(spearman),
    }


def _compute_binary_metrics(ctx: HeadContext) -> Dict[str, float]:
    if ctx.scores is None or ctx.labels.size == 0:
        return {}
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    scores = ctx.scores[mask].astype(float)
    truths = ctx.labels[mask].astype(float).flatten()
    if scores.size == 0:
        return {}

    y_true = truths.astype(int)
    best_thr, best_f1, best_acc = find_best_threshold(y_true, scores)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    mse = mean_squared_error(y_true, scores)
    return {
        f"{ctx.head}_best-threshold": float(best_thr),
        f"{ctx.head}_best-accuracy": float(best_acc),
        f"{ctx.head}_best-f1": float(best_f1),
        f"{ctx.head}_auc": float(auc),
        f"{ctx.head}_mse": float(mse),
    }


def _compute_infonce_metrics(ctx: HeadContext) -> Dict[str, float]:
    if ctx.embeddings is None or ctx.centroids is None or not ctx.centroids:
        return {}
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    embeddings = ctx.embeddings
    if embeddings.ndim != 2:
        return {}
    if mask.shape[0] != embeddings.shape[0]:
        return {}
    emb_subset = embeddings[mask]
    if emb_subset.shape[0] == 0:
        return {}
    labels = ctx.labels[mask] if ctx.labels.size else np.array([], dtype=int)
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = labels.flatten()
    labels = labels.astype(int, copy=False)

    centroid_labels = np.array(list(ctx.centroids.keys()), dtype=int)
    centroid_vectors = np.asarray([ctx.centroids[label] for label in centroid_labels], dtype=float)
    if centroid_vectors.ndim != 2 or centroid_vectors.shape[0] == 0:
        return {}
    if centroid_vectors.shape[1] != emb_subset.shape[1]:
        return {}

    diff = emb_subset[:, None, :] - centroid_vectors[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    y_pred = centroid_labels[nearest_idx]
    if labels.size == 0:
        return {}

    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
    ami = adjusted_mutual_info_score(labels, y_pred)
    v_measure = v_measure_score(labels, y_pred)
    return {
        f"{ctx.head}_knn_accuracy": float(acc),
        f"{ctx.head}_knn_macro_f1": float(f1),
        f"{ctx.head}_cluster_ami": float(ami),
        f"{ctx.head}_cluster_v_measure": float(v_measure),
    }


def _compute_contrastive_metrics(ctx: HeadContext) -> Dict[str, float]:
    if ctx.pos_scores is None or ctx.neg_scores is None:
        return {}
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    pos_scores = ctx.pos_scores[mask].astype(float)
    neg_scores = ctx.neg_scores[mask].astype(float)
    metrics: Dict[str, float] = {}
    if pos_scores.size > 0 and neg_scores.size > 0:
        trip_acc = float(np.mean(pos_scores > neg_scores))
        avg_pos = float(np.mean(pos_scores))
        avg_neg = float(np.mean(neg_scores))
        metrics[f"{ctx.head}_triplet_accuracy"] = trip_acc
        metrics[f"{ctx.head}_avg_positive_similarity"] = avg_pos
        metrics[f"{ctx.head}_avg_negative_similarity"] = avg_neg

    if ctx.anchor_probs is not None and ctx.labels.size > 0:
        probs = ctx.anchor_probs[mask].astype(float)
        y_true = ctx.labels[mask].astype(int)
        if probs.ndim == 2 and y_true.size > 0:
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            metrics[f"{ctx.head}_anchor_accuracy"] = float(acc)
            metrics[f"{ctx.head}_anchor_macro_f1"] = float(f1)
    return metrics

def find_best_threshold(y_true, scores, n_thresholds=100):
    """
    NaN/Inf を除外してから、閾値探索して best_f1, best_acc を返す。
    """
    y = np.array(y_true, dtype=float)
    s = np.array(scores, dtype=float)

    # 有効なインデックス = ラベルもスコアも finite
    mask_valid = (~np.isnan(y)) & np.isfinite(y) & np.isfinite(s)
    if not mask_valid.any():
        return 0.5, 0.0, 0.0

    y = y[mask_valid]
    s = s[mask_valid]

    # -1/1 ラベルを 0/1 に変換
    uniq = set(np.unique(y))
    if uniq == {-1.0, 1.0}:
        y = (y == 1.0).astype(int)
    else:
        y = y.astype(int)

    # 探索用閾値: finite な最小・最大を使う
    thr_min, thr_max = np.min(s), np.max(s)
    if thr_min == thr_max:
        # 全て同じなら閾値は thr_min
        return float(thr_min), 0.0, accuracy_score(y, (s >= thr_min).astype(int))

    # thresholds = np.linspace(thr_min, thr_max, n_thresholds)
    thresholds = np.linspace(0, 1, n_thresholds) 

    best_f1, best_acc, best_thr = 0.0, 0.0, thresholds[0]
    for thr in thresholds:
        y_pred = (s >= thr).astype(int)
        try:
            f1 = f1_score(y, y_pred)
        except ValueError:
            f1 = 0.0
        acc = accuracy_score(y, y_pred)
        if f1 > best_f1:
            best_f1, best_acc, best_thr = f1, acc, thr

    return float(best_thr), float(best_f1), float(best_acc)

def _select_centroids(
    entry: Optional[Dict],
    mode: str,
) -> Optional[Dict[int, np.ndarray]]:
    if not entry:
        return None
    if isinstance(entry, dict) and any(
        isinstance(k, str) and k in ("classifier", "original") for k in entry.keys()
    ):
        primary = "original" if mode == "original" else "classifier"
        if entry.get(primary):
            return entry[primary]
        fallback = "classifier" if primary == "original" else "original"
        return entry.get(fallback)
    return entry


OBJECTIVE_HANDLERS: Dict[str, Callable[[HeadContext], Dict[str, float]]] = {
    "regression": _compute_regression_metrics,
    "binary_classification": _compute_binary_metrics,
    "infoNCE": _compute_infonce_metrics,
    "contrastive_logit": _compute_contrastive_metrics,
}


def compute_metrics(
    eval_pred: EvalPrediction,
    classifier_configs: dict,
    id2_head: dict,
    train_centroid_getter: Optional[Callable[[], Dict[str, Dict[str, Dict[int, np.ndarray]]]]] = None,
    embedding_eval_mode: str = "classifier",
) -> dict:
    """
    - eval_pred.predictions は dict で、
        * 各 head の 2文タスク出力: predictions[head] → shape (N,)
        * 3文タスク出力: predictions[f"{head}_pos_similarity"], 
                     predictions[f"{head}_neg_similarity"],
                     predictions[f"{head}_anchor_prob"] → shape (N,) or (N, num_classes)
        * predictions["active_heads"] → 長さ N の list/array of str
    - eval_pred.label_ids は shape (N,) の numpy array または tensor
    - classifier_configs に従い、サンプルごとに active_head を見てメトリックを計算
    - 最後に全 head ペア間の Pearson 相関を計算
    - embedding_eval_mode: "classifier" の場合は各 head の埋め込みを使用、
      "original" の場合はベース埋め込みで InfoNCE 評価を行う
    """
    preds = eval_pred.predictions
    label_dict = eval_pred.label_ids

    labels = label_dict["labels"] if isinstance(label_dict, dict) else label_dict
    labels = np.asarray(labels)
    active_head_ids = label_dict.get("active_heads", []) if isinstance(label_dict, dict) else []
    active_heads = [id2_head[i] for i in active_head_ids]

    metrics: Dict[str, float] = {}
    head_vectors: Dict[str, np.ndarray] = {}
    train_centroids = train_centroid_getter() if train_centroid_getter is not None else {}

    origin_vector = _build_head_vector(
        _to_float_array(preds.get("original")),
        _to_float_array(preds.get("original_pos_similarity")),
        _to_float_array(preds.get("original_neg_similarity")),
    )
    head_vectors["original"] = origin_vector

    for head, cfg in classifier_configs.items():
        objective = cfg.get("objective", "")
        active_mask = np.asarray([h == head for h in active_heads], dtype=bool)

        scores = _to_float_array(preds.get(head))
        pos_scores = _to_float_array(preds.get(f"{head}_pos_similarity"))
        neg_scores = _to_float_array(preds.get(f"{head}_neg_similarity"))
        anchor_probs = _to_float_array(preds.get(f"{head}_anchor_prob"))

        embeddings = None
        centroids = None
        if objective == "infoNCE":
            embedding_source = preds.get("embeddings") if embedding_eval_mode == "original" else preds.get(head)
            embeddings = _to_float_array(embedding_source)
            centroids = _select_centroids((train_centroids or {}).get(head), embedding_eval_mode)

        context = HeadContext(
            head=head,
            objective=objective,
            scores=scores,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            anchor_probs=anchor_probs,
            embeddings=embeddings,
            labels=labels,
            active_mask=active_mask,
            centroids=centroids,
        )

        handler = OBJECTIVE_HANDLERS.get(objective)
        if handler:
            metrics.update(handler(context))

        head_vectors[head] = _build_head_vector(scores, pos_scores, neg_scores)

    for h1, h2 in combinations(head_vectors.keys(), 2):
        v1 = np.asarray(head_vectors[h1]).flatten()
        v2 = np.asarray(head_vectors[h2]).flatten()

        if v1.size <= 1 or v2.size <= 1:
            continue
        if len(v1) != len(v2):
            continue

        pcc = pearsonr(v1, v2)[0]
        scc = spearmanr(v1, v2).correlation

        metrics[f"{h1}_vs_{h2}_relation-pearson"] = float(pcc)
        metrics[f"{h1}_vs_{h2}_relation-spearman"] = float(scc)

    if embedding_eval_mode == "original":
        metrics = {f"original_{key}": value for key, value in metrics.items()}

    return metrics
