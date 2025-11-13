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
    # print(f"preds: {preds}")
    # print(f"labels: {labels}")
    # print(f"active_heads: {label_dict['active_heads']}")
    active_heads = [id2_head[i] for i in label_dict['active_heads']]
    metrics: dict = {}
    head_vectors: dict = {}
    train_centroids = train_centroid_getter() if train_centroid_getter is not None else {}
    
     # 各ベクトルを取得して float 配列に変換
    origin_pair = np.asarray(preds["original"], dtype=float)
    origin_pos  = np.asarray(preds["original_pos_similarity"], dtype=float)
    origin_neg  = np.asarray(preds["original_neg_similarity"], dtype=float)

    # NaN 以外の要素だけ抽出するヘルパー
    def clean(arr: np.ndarray) -> np.ndarray:
        return arr[~np.isnan(arr)]

    # クリーンした配列を追加（空配列はスキップ）
    origin_arrays = []
    for arr in (origin_pair, origin_pos, origin_neg):
        cleaned = clean(arr)
        if cleaned.size > 0:
            origin_arrays.append(cleaned)

    # 連結
    if origin_arrays:
        origin_vec = np.concatenate(origin_arrays, axis=0)
    else:
        origin_vec = np.array([], dtype=float)

    head_vectors["original"] = origin_vec

    for head, cfg in classifier_configs.items():
        arrays = []

        # 各ベクトルを取得して float 配列に変換
        pair = np.asarray(preds[head], dtype=float)
        pos  = np.asarray(preds[f"{head}_pos_similarity"], dtype=float)
        neg  = np.asarray(preds[f"{head}_neg_similarity"], dtype=float)

        # NaN 以外の要素だけ抽出するヘルパー
        def clean(arr: np.ndarray) -> np.ndarray:
            return arr[~np.isnan(arr)]

        # クリーンした配列を追加（空配列はスキップ）
        for arr in (pair, pos, neg):
            cleaned = clean(arr)
            if cleaned.size > 0:
                arrays.append(cleaned)

        # 連結
        if arrays:
            vec = np.concatenate(arrays, axis=0)
        else:
            vec = np.array([], dtype=float)

        head_vectors[head] = vec
        
        obj = cfg["objective"]
        # サンプルごとにこの head が active な index を集める
        # print(f"Processing head: {head}, objective: {obj}, active_heads: {active_heads}")
        idx = [True if h == head else False for h in active_heads]
        if not any(idx):
            continue

        if obj in ("regression", "binary_classification"):
            # 2文タスク: 予測スコアとラベル
            scores = np.asarray(preds[head], dtype=float)[idx]
            truths = labels[idx].astype(float).flatten()
            # print(f"Head: {head}, scores shape: {scores.shape}, truths shape: {truths.shape}")

            if obj == "binary_classification":
                y_true = truths.astype(int)
                best_thr, best_f1, best_acc = find_best_threshold(y_true, scores)
                try:
                    auc = roc_auc_score(y_true, scores)
                except:
                    auc = float("nan")
                mse = mean_squared_error(y_true, scores)

                metrics[f"{head}_best-threshold"] = float(best_thr)
                metrics[f"{head}_best-accuracy"] = float(best_acc)
                metrics[f"{head}_best-f1"]       = float(best_f1)
                metrics[f"{head}_auc"]      = float(auc)
                metrics[f"{head}_mse"]      = float(mse)

            else:  # regression
                mse = mean_squared_error(truths, scores)
                # サンプル数が 2 以上あれば計算
                pear = pearsonr(scores, truths)[0] if scores.size>1 else 0.0

                metrics[f"{head}_mse"]     = float(mse)
                metrics[f"{head}_single-pearson"] = float(pear)
                metrics[f"{head}_single-spearman"] = float(spearmanr(scores, truths).correlation)

        elif obj == "infoNCE":
            embedding_source = preds.get("embeddings") if embedding_eval_mode == "original" else preds.get(head)
            if embedding_source is None:
                continue
            embeddings = np.asarray(embedding_source, dtype=float)
            if embeddings.ndim != 2:
                continue
            mask = np.asarray(idx, dtype=bool)
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
                continue

            head_centroids = _select_centroids(
                (train_centroids or {}).get(head),
                embedding_eval_mode,
            )
            if not head_centroids:
                continue

            centroid_labels = np.array(list(head_centroids.keys()), dtype=int)
            centroid_vectors = np.asarray(
                [head_centroids[label] for label in centroid_labels], dtype=float
            )
            if centroid_vectors.ndim != 2 or centroid_vectors.shape[0] == 0:
                continue
            if centroid_vectors.shape[1] != emb_subset.shape[1]:
                continue

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

        elif obj == "contrastive_logit":
            # 3文タスク: pos/neg を縦連結してベクトル化

            pos_hot = pos[idx]
            neg_hot = neg[idx]
            # Triplet accuracy, avg pos/neg
            trip_acc = float(np.mean(pos_hot > neg_hot))
            avg_pos  = float(np.mean(pos_hot))
            avg_neg  = float(np.mean(neg_hot))
            metrics[f"{head}_triplet_accuracy"]       = trip_acc
            metrics[f"{head}_avg_positive_similarity"] = avg_pos
            metrics[f"{head}_avg_negative_similarity"] = avg_neg

            # Anchor‐prob での多クラス評価
            probs = np.asarray(preds[f"{head}_anchor_prob"], dtype=float)[idx]
            # int ラベルとして解釈
            y_true = labels[idx].astype(int)
            if probs.ndim == 2 and y_true.size > 0:
                y_pred = np.argmax(probs, axis=1)
                acc2 = accuracy_score(y_true, y_pred)
                f12  = f1_score(y_true, y_pred, average="macro")
                metrics[f"{head}_anchor_accuracy"] = float(acc2)
                metrics[f"{head}_anchor_macro_f1"] = float(f12)

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
