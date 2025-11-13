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
from typing import Union, List, Dict, Optional, Callable, Iterator, Any, TypedDict

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


def _clean_and_concat(*arrays: Optional[Any]) -> np.ndarray:
    """Remove NaNs from the given arrays and concatenate the remaining values."""
    cleaned_parts: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=float).ravel()
        if arr_np.size == 0:
            continue
        arr_np = arr_np[~np.isnan(arr_np)]
        if arr_np.size:
            cleaned_parts.append(arr_np)
    if cleaned_parts:
        return np.concatenate(cleaned_parts, axis=0)
    return np.array([], dtype=float)


class HeadBatch(TypedDict):
    head: str
    config: Dict[str, Any]
    mask: np.ndarray
    scores: np.ndarray
    pos: np.ndarray
    neg: np.ndarray


def _iter_active_heads(
    preds: Dict[str, np.ndarray],
    labels: np.ndarray,
    active_heads: List[str],
    classifier_configs: Dict[str, Dict],
) -> Iterator[HeadBatch]:
    _ = labels  # Kept for signature compatibility
    mask_template = np.asarray(active_heads)
    for head, cfg in classifier_configs.items():
        mask = np.asarray(mask_template == head, dtype=bool)
        if not mask.any():
            continue
        score_raw = preds.get(head)
        pos_raw = preds.get(f"{head}_pos_similarity")
        neg_raw = preds.get(f"{head}_neg_similarity")
        scores = np.asarray(score_raw, dtype=float) if score_raw is not None else np.array([], dtype=float)
        pos = np.asarray(pos_raw, dtype=float) if pos_raw is not None else np.array([], dtype=float)
        neg = np.asarray(neg_raw, dtype=float) if neg_raw is not None else np.array([], dtype=float)
        yield {
            "head": head,
            "config": cfg,
            "mask": mask,
            "scores": scores,
            "pos": pos,
            "neg": neg,
        }


def _extract_scores_and_truths(
    batch: HeadBatch,
    labels: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    scores = np.asarray(batch["scores"], dtype=float)
    mask = batch["mask"]
    if scores.shape[0] != mask.shape[0]:
        return None
    truths = np.asarray(labels, dtype=float)[mask].astype(float).flatten()
    scores_subset = scores[mask].astype(float).flatten()
    return scores_subset, truths


def _handle_binary_classification(
    metrics: Dict[str, float],
    batch: HeadBatch,
    labels: np.ndarray,
    preds: Dict[str, np.ndarray],
    train_centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    embedding_eval_mode: str,
) -> None:
    extracted = _extract_scores_and_truths(batch, labels)
    if not extracted:
        return
    scores_subset, truths = extracted
    if scores_subset.size == 0:
        return
    head = batch["head"]
    y_true = truths.astype(int)
    best_thr, best_f1, best_acc = find_best_threshold(y_true, scores_subset)
    try:
        auc = roc_auc_score(y_true, scores_subset)
    except Exception:
        auc = float("nan")
    mse = mean_squared_error(y_true, scores_subset)

    metrics[f"{head}_best-threshold"] = float(best_thr)
    metrics[f"{head}_best-accuracy"] = float(best_acc)
    metrics[f"{head}_best-f1"] = float(best_f1)
    metrics[f"{head}_auc"] = float(auc)
    metrics[f"{head}_mse"] = float(mse)


def _handle_regression(
    metrics: Dict[str, float],
    batch: HeadBatch,
    labels: np.ndarray,
    preds: Dict[str, np.ndarray],
    train_centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    embedding_eval_mode: str,
) -> None:
    extracted = _extract_scores_and_truths(batch, labels)
    if not extracted:
        return
    scores_subset, truths = extracted
    if scores_subset.size == 0:
        return
    head = batch["head"]
    mse = mean_squared_error(truths, scores_subset)
    pear = pearsonr(scores_subset, truths)[0] if scores_subset.size > 1 else 0.0
    spearman = spearmanr(scores_subset, truths).correlation

    metrics[f"{head}_mse"] = float(mse)
    metrics[f"{head}_single-pearson"] = float(pear)
    metrics[f"{head}_single-spearman"] = float(spearman)


def _handle_info_nce(
    metrics: Dict[str, float],
    batch: HeadBatch,
    labels: np.ndarray,
    preds: Dict[str, np.ndarray],
    train_centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    embedding_eval_mode: str,
) -> None:
    head = batch["head"]
    mask = batch["mask"]
    embedding_source = preds.get("embeddings") if embedding_eval_mode == "original" else preds.get(head)
    if embedding_source is None:
        return
    embeddings = np.asarray(embedding_source, dtype=float)
    if embeddings.ndim != 2:
        return
    emb_subset = embeddings[mask]
    label_subset = labels[mask] if len(labels) else np.array([], dtype=int)
    label_subset = np.asarray(label_subset)
    if label_subset.ndim > 1:
        label_subset = label_subset.reshape(label_subset.shape[0], -1)
        if label_subset.shape[1] == 1:
            label_subset = label_subset[:, 0]
        else:
            label_subset = label_subset.flatten()
    label_subset = label_subset.astype(int, copy=False)
    if emb_subset.shape[0] == 0:
        return

    head_centroids = _select_centroids(
        (train_centroids or {}).get(head),
        embedding_eval_mode,
    )
    if not head_centroids:
        return

    centroid_labels = np.array(list(head_centroids.keys()), dtype=int)
    centroid_vectors = np.asarray(
        [head_centroids[label] for label in centroid_labels], dtype=float
    )
    if centroid_vectors.ndim != 2 or centroid_vectors.shape[0] == 0:
        return
    if centroid_vectors.shape[1] != emb_subset.shape[1]:
        return

    diff = emb_subset[:, None, :] - centroid_vectors[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    y_pred = centroid_labels[nearest_idx]

    if label_subset.size > 0:
        acc = accuracy_score(label_subset, y_pred)
        f1 = f1_score(label_subset, y_pred, average="macro", zero_division=0)
        ami = adjusted_mutual_info_score(label_subset, y_pred)
        v_measure = v_measure_score(label_subset, y_pred)
        metrics[f"{head}_knn_accuracy"] = float(acc)
        metrics[f"{head}_knn_macro_f1"] = float(f1)
        metrics[f"{head}_cluster_ami"] = float(ami)
        metrics[f"{head}_cluster_v_measure"] = float(v_measure)


def _handle_contrastive_logit(
    metrics: Dict[str, float],
    batch: HeadBatch,
    labels: np.ndarray,
    preds: Dict[str, np.ndarray],
    train_centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    embedding_eval_mode: str,
) -> None:
    head = batch["head"]
    mask = batch["mask"]
    pos = batch["pos"]
    neg = batch["neg"]
    if pos.shape[0] != mask.shape[0] or neg.shape[0] != mask.shape[0]:
        return

    pos_hot = pos[mask]
    neg_hot = neg[mask]
    trip_acc = float(np.mean(pos_hot > neg_hot))
    avg_pos = float(np.mean(pos_hot))
    avg_neg = float(np.mean(neg_hot))
    metrics[f"{head}_triplet_accuracy"] = trip_acc
    metrics[f"{head}_avg_positive_similarity"] = avg_pos
    metrics[f"{head}_avg_negative_similarity"] = avg_neg

    anchor_raw = preds.get(f"{head}_anchor_prob")
    if anchor_raw is None:
        return
    probs = np.asarray(anchor_raw, dtype=float)
    if probs.shape[0] != mask.shape[0]:
        return
    probs = probs[mask]
    y_true = labels[mask].astype(int)
    if probs.ndim == 2 and y_true.size > 0:
        y_pred = np.argmax(probs, axis=1)
        acc2 = accuracy_score(y_true, y_pred)
        f12 = f1_score(y_true, y_pred, average="macro")
        metrics[f"{head}_anchor_accuracy"] = float(acc2)
        metrics[f"{head}_anchor_macro_f1"] = float(f12)

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
    print(f"predictions keys: {preds.keys()}")
    print(f"label_dict keys: {label_dict.keys() if isinstance(label_dict, dict) else 'not a dict'}")
    print(f"predictions shape: {len(preds)}, label_dict shape: {len(label_dict)}")
    
    labels = label_dict["labels"] if isinstance(label_dict, dict) else label_dict
    labels = np.asarray(labels)
    # print(f"preds: {preds}")
    # print(f"labels: {labels}")
    # print(f"active_heads: {label_dict['active_heads']}")
    active_heads = [id2_head[i] for i in label_dict['active_heads']]
    metrics: dict = {}
    head_vectors: dict = {}
    train_centroids = train_centroid_getter() if train_centroid_getter is not None else {}

    head_vectors["original"] = _clean_and_concat(
        preds.get("original"),
        preds.get("original_pos_similarity"),
        preds.get("original_neg_similarity"),
    )

    handler_map = {
        "binary_classification": _handle_binary_classification,
        "regression": _handle_regression,
        "infoNCE": _handle_info_nce,
        "contrastive_logit": _handle_contrastive_logit,
    }

    for head_info in _iter_active_heads(preds, labels, active_heads, classifier_configs):
        head = head_info["head"]
        scores = head_info["scores"]
        pos = head_info["pos"]
        neg = head_info["neg"]
        obj = head_info["config"]["objective"]

        head_vectors[head] = _clean_and_concat(scores, pos, neg)

        handler = handler_map.get(obj)
        if handler is None:
            continue
        handler(metrics, head_info, labels, preds, train_centroids, embedding_eval_mode)

    # for k, v in head_vectors.items():
    #     print(f"Head: {k}, vector length: {len(v)}")
    
    for h1, h2 in combinations(head_vectors.keys(), 2):
        # 1D の numpy 配列に変換
        v1 = np.asarray(head_vectors[h1]).flatten()
        v2 = np.asarray(head_vectors[h2]).flatten()

        # print(f"v1, v2: {v1}, {v2}")
        # print(f"v1, v2: {len(v1)}, {len(v2)}")

        if v1.size <= 1 or v2.size <= 1:
            # データ点が足りないのでスキップ
            continue
        elif len(v1) != len(v2):
            # 長さが違うのでスキップ
            continue

        a, b = v1, v2
        # Pearson
        pcc = pearsonr(a, b)[0]
        # Spearman
        scc = spearmanr(a, b).correlation

        metrics[f"{h1}_vs_{h2}_relation-pearson"] = float(pcc)
        metrics[f"{h1}_vs_{h2}_relation-spearman"] = float(scc)

    if embedding_eval_mode == "original":
        metrics = {f"original_{key}": value for key, value in metrics.items()}

    return metrics
