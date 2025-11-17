# 埋め込み表現のばらつき具合計算機能の実装

## 概要
evaluation stepsごとに、下流タスクの精度に加えて、埋め込み表現空間のばらつき具合（isotropy）を計算する機能を追加しました。

## 変更したファイル

### 1. `utils/IsoScore_functions/existing_scores.py`
**変更内容**: 構文エラーの修正
- 54行目の余分な閉じ括弧を削除

**修正前**:
```python
pc_embed = pca_model.fit_transform(points.T))
```

**修正後**:
```python
pc_embed = pca_model.fit_transform(points.T)
```

### 2. `utils/metrics/__init__.py`
**変更内容**: isotropy scoreの計算機能を追加

#### 追加された機能:

1. **インポートの追加**:
   - `IsoScore` from `..IsoScore_functions.IsoScore`
   - `cosine_score`, `partition_score`, `varex_score`, `id_score` from `..IsoScore_functions.existing_scores`

2. **グローバル変数**:
   - `_original_embeddings_processed`: originalの埋め込みが処理済みかどうかを追跡

3. **新しいヘルパー関数: `compute_isotropy_scores()`**:
   ```python
   def compute_isotropy_scores(embeddings: np.ndarray, prefix: str = "") -> Dict[str, float]
   ```
   
   計算される5つのスコア:
   - `cos`: 1 - cosine_score (値が高いほどばらつきが大きい)
   - `part`: partition_score
   - `iso`: IsoScore
   - `varex`: variance explained score
   - `id_score`: intrinsic dimensionality score

4. **`compute_metrics()`関数の修正**:
   
   a. **Original埋め込みのisotropyスコア** (初回のみ計算):
   - 訓練データセットの埋め込み表現を使用
   - `original_avg`, `original_cls`, `original_max`の各pooling methodに対して計算
   - `train_centroids`から`{pool_key}_all_embeddings`を取得
   
   b. **分類器埋め込みのisotropyスコア** (各evaluation stepで計算):
   - 訓練データセットの埋め込み表現を使用
   - 各headの`classifier_all_embeddings`を取得して計算

### 3. `utils/training/centroid_calculator.py`
**変更内容**: 全埋め込み表現の保存機能を追加

#### `build_train_centroids()`メソッドの拡張:

1. **新しいアキュムレータの追加**:
   ```python
   head_all_embeddings_classifier: Dict[str, List[np.ndarray]] = defaultdict(list)
   head_all_embeddings_original: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
       lambda: defaultdict(list)
   )
   ```

2. **埋め込みの収集**:
   - 各サンプルの埋め込みを重心計算用とisotropyスコア計算用の両方で保存
   - classifierとoriginal (avg/cls/max/last) の全pooling methodについて保存

3. **戻り値の拡張**:
   重心に加えて、以下のキーが追加されます:
   - `classifier_all_embeddings`: 分類器の全埋め込み
   - `original_avg_all_embeddings`: avgプーリングの全埋め込み
   - `original_cls_all_embeddings`: clsプーリングの全埋め込み
   - `original_max_all_embeddings`: maxプーリングの全埋め込み
   - `original_last_all_embeddings`: lastプーリングの全埋め込み

## 使用方法

変更後、トレーニングを実行すると自動的に以下のメトリクスが計算されます:

### Original埋め込み (初回評価時のみ):
- `original_avg_cos`
- `original_avg_part`
- `original_avg_iso`
- `original_avg_varex`
- `original_avg_id_score`
- (同様に `original_cls_*`, `original_max_*`)

### 分類器埋め込み (各evaluation stepごと):
- `{head_name}_cos`
- `{head_name}_part`
- `{head_name}_iso`
- `{head_name}_varex`
- `{head_name}_id_score`

例: sentimentヘッドの場合:
- `sentiment_cos`
- `sentiment_part`
- `sentiment_iso`
- `sentiment_varex`
- `sentiment_id_score`

## 注意事項

1. **訓練データを使用**: 全てのisotropyスコアは訓練データセットの埋め込み表現を使用して計算されます
2. **初回のみ vs 毎回**:
   - Original埋め込み: 最初の評価時のみ計算（変化しないため）
   - 分類器埋め込み: 各evaluation stepで計算（訓練により変化するため）
3. **エラーハンドリング**: 計算が失敗した場合は静かにスキップされ、トレーニングは継続されます
4. **メモリ使用**: 全埋め込みを保存するため、大きな訓練データセットの場合はメモリ使用量が増加します

## テスト

syntax checkは成功しています:
```bash
python3 -m py_compile utils/training/centroid_calculator.py
python3 -m py_compile utils/metrics/__init__.py
python3 -m py_compile utils/IsoScore_functions/existing_scores.py
```

実際のトレーニングで動作を確認するには:
```bash
cd /remote/csifs1/disk3/users/yama11235/yama11235/Sentiment-Circle/utils
bash train.sh
```
