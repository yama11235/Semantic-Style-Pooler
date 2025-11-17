# Sentiment-Circle Refactoring Summary

## Overview
このプロジェクトは埋め込みモデルを訓練するプログラムで、複数の分類器を同時に学習することを想定しています。
このリファクタリングでは、長大なファイルを適切にモジュール化し、コードの可読性を向上させました。

## Refactored Structure

### 1. Configuration Module (`utils/config/`)
トレーニングに関する設定を独立したモジュールに分離しました。

- **`utils/config/arguments.py`**: 
  - `TrainingArguments`: トレーニング引数
  - `DataTrainingArguments`: データ処理引数
  - `ModelArguments`: モデル設定引数

### 2. Data Module (`utils/data/`)
データセット読み込みとラベル処理を分離しました。

- **`utils/data/data_loader.py`**:
  - `load_raw_datasets()`: データセット読み込み処理
  
- **`utils/data/label_utils.py`**:
  - `prepare_label_mappings()`: ラベルマッピング処理

### 3. Training Module (`utils/training/`)
トレーニングセットアップ処理を独立したモジュールに分離しました。

- **`utils/training/train_setup.py`**:
  - `initialize_wandb()`: Weights & Biases初期化
  - `setup_model_and_config()`: モデルと設定のセットアップ
  - `setup_tokenizer()`: トークナイザーのセットアップ
  - `prepare_datasets()`: データセットの前処理
  - `create_trainer()`: トレーナーの作成

### 4. Metrics Module (`utils/metrics/`)
メトリクス計算を目的関数ごとに分離しました（433行 → 各ファイル約100-200行）。

- **`utils/metrics/base.py`**: 
  - `HeadContext`: メトリクス計算用のコンテキストデータクラス
  - ヘルパー関数: `flatten_floats`, `to_float_array`, `build_head_vector`, `select_centroids`
  
- **`utils/metrics/regression.py`**:
  - `compute_regression_metrics()`: 回帰タスクのメトリクス計算
  
- **`utils/metrics/classification.py`**:
  - `compute_binary_metrics()`: 二値分類メトリクス
  - `compute_infonce_metrics()`: InfoNCE（対照学習）メトリクス
  - `find_best_threshold()`: 最適閾値探索
  
- **`utils/metrics/contrastive.py`**:
  - `compute_contrastive_metrics()`: 対照学習タスクのメトリクス
  
- **`utils/metrics/__init__.py`**:
  - `compute_metrics()`: メインのメトリクス計算関数
  - 各メトリクス計算関数の統合

### 5. Main Training Script (`utils/train.py`)
メインスクリプトを簡潔化しました（890行 → 約200行）。

従来の長大な関数を各モジュールに分割し、`main()` 関数は以下の流れで構成されています：
1. 引数のパース
2. ログ設定
3. データセット読み込み
4. ラベルマッピング準備
5. モデルとトークナイザーのセットアップ
6. データセット前処理
7. トレーナー作成
8. トレーニング実行

## Benefits of Refactoring

### 可読性の向上
- 各ファイルが100-200行程度の適切なサイズになり、理解しやすくなりました
- 機能ごとにモジュールが分離されており、必要な部分を素早く見つけられます

### 保守性の向上
- 責務が明確に分離されているため、修正や拡張が容易です
- 新しいメトリクスや目的関数の追加が簡単になりました

### テスト容易性
- 各モジュールが独立しているため、ユニットテストが書きやすくなりました
- モックを使った単体テストが可能になりました

### 再利用性
- 各モジュールが独立しているため、他のプロジェクトでも再利用可能です
- 例: metrics モジュールは他の評価タスクでも使用できます

## Backward Compatibility

既存のコードとの後方互換性を維持しています：

- `utils/train.py`: 同じインターフェースを維持
- `utils/metrics.py`: ラッパーモジュールとして従来の import パスをサポート
- 既存のトレーニングスクリプトやコンフィグは変更なしで動作します

## Original Files Backup

リファクタリング前のファイルは以下の名前でバックアップされています：
- `utils/metrics_original.py`: 元の metrics.py
- `utils/metrics_old_backup.py`: バックアップコピー

## Next Steps (推奨される次のリファクタリング)

時間の制約により完了していない部分の推奨事項：

1. **`clf_trainer.py` のリファクタリング** (599行):
   - 損失計算ロジックを `utils/training/loss_computation.py` に分離
   - 重心計算を `utils/training/centroid_calculator.py` に分離
   - T-SNEプロット機能を `utils/visualization/tsne_plotter.py` に分離

2. **`loss_function.py` の整理** (354行):
   - 各損失関数を個別のファイルに分離
   - ヘルパー関数を `utils/training/loss_helpers.py` に移動

3. **エラーハンドリングの改善**:
   - より具体的な例外クラスの定義
   - エラーメッセージの統一

4. **ドキュメンテーション**:
   - 各モジュールの詳細なdocstringの追加
   - 使用例の追加

## Usage Example

```python
# 従来通りの使い方
from utils.train import main

if __name__ == "__main__":
    main()

# 個別のモジュールを使用
from utils.config import TrainingArguments, DataTrainingArguments, ModelArguments
from utils.data import load_raw_datasets, prepare_label_mappings
from utils.training import setup_model_and_config
from utils.metrics import compute_metrics

# ... (各関数を個別に呼び出し可能)
```

## File Size Comparison

### Before Refactoring:
- `train.py`: 890 lines
- `clf_trainer.py`: 599 lines
- `metrics.py`: 433 lines

### After Refactoring:
- `train.py`: ~200 lines (77% reduction)
- `config/arguments.py`: ~200 lines
- `data/data_loader.py`: ~120 lines
- `data/label_utils.py`: ~160 lines
- `training/train_setup.py`: ~320 lines
- `metrics/base.py`: ~110 lines
- `metrics/regression.py`: ~45 lines
- `metrics/classification.py`: ~180 lines
- `metrics/contrastive.py`: ~45 lines
- `metrics/__init__.py`: ~200 lines

各ファイルが適切なサイズになり、単一責任の原則に従っています。
