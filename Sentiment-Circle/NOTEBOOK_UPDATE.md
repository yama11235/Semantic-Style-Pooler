# train_flow_ngpt.ipynb 更新ドキュメント（修正版）

## 更新概要

リファクタリング後のモジュール構造に対応し、**train.pyと完全に同じ関数呼び出しを使用する**ようにノートブックを修正しました。

## 主な変更点

### 1. Import文の更新

**Before (元のコード):**
```python
from train import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
    load_raw_datasets,
    prepare_label_mappings,
)
from dataset_preprocessing import batch_get_preprocessing_function, get_preprocessing_function
from model.modeling_utils import DataCollatorForBiEncoder, get_model
from clf_trainer import CustomTrainer
from progress_logger import LogCallback
from model.nGPT_model import NGPTWeightNormCallback
from metrics import compute_metrics
```

**After (リファクタリング後):**
```python
# Config module
from utils.config import ModelArguments, DataTrainingArguments, TrainingArguments

# Data module
from utils.data import (
    load_raw_datasets,
    prepare_label_mappings,
    get_preprocessing_function,
    batch_get_preprocessing_function,
)

# Training module
from utils.training import (
    setup_model_and_config,
    setup_tokenizer,
)

# Model utilities
from utils.model.modeling_utils import DataCollatorForBiEncoder, get_model
from utils.model.nGPT_model import NGPTWeightNormCallback

# Trainer
from utils.clf_trainer import CustomTrainer

# Metrics
from utils.metrics import compute_metrics

# Logger
from utils.progress_logger import LogCallback
```

### 2. セル構成の改善

- **明確なセクション分割**: 各ステップが独立したマークダウンセルで説明されています
- **詳細な出力**: 各ステップで処理内容が確認できるようにprint文を追加
- **結果比較セル**: ベースライン、トレーニング後、テストの結果を比較するセルを追加

### 3. 新しいセル構成

1. **準備** (Cell 0): リファクタリング後のモジュールインポート
2. **ハイパーパラメータ設定** (Cell 1): モデル、データ、トレーニング引数の設定
3. **分類器設定** (Cell 2): JSON設定ファイルの読み込み
4. **データセット読み込み** (Cell 3): データセットの読み込みと確認
5. **ラベルマッピング** (Cell 4): ラベルマッピングの準備
6. **モデル・トークナイザー** (Cell 5): モデルとトークナイザーのセットアップ
7. **データ前処理** (Cell 6): 前処理関数の適用
8. **トレーナー初期化** (Cell 7): トレーナーとコレーターの初期化
9. **目的関数確認** (Cell 8): ヘッド目的関数の確認
10. **初期評価** (Cell 9): ベースラインメトリクスの取得
11. **トレーニング** (Cell 10): モデルの学習
12. **テスト評価** (Cell 11): テストセットでの評価
13. **結果比較** (Cell 12): パフォーマンス比較表の表示
14. **モデル保存** (Cell 13): トレーニング済みモデルの保存

## 使用方法

### 基本的な実行フロー

1. **セル0**: 必要なモジュールをインポート
   ```python
   # 自動的に実行されるmagicコマンド
   %load_ext autoreload
   %autoreload 2
   ```

2. **セル1-8**: 順番に実行してトレーナーを初期化

3. **セル9**: 初期評価を実行してベースライン性能を確認

4. **セル10**: トレーニングを実行（時間がかかる場合があります）

5. **セル11**: テストセットで最終評価

6. **セル12**: 結果を比較表示

7. **セル13**: （オプション）モデルを保存

### デバッグ用の設定

- サンプル数を制限: `max_train_samples=64`, `max_eval_samples=64`
- 小さいバッチサイズ: `per_device_train_batch_size=4`
- Wandb無効化: `WANDB_MODE=disabled`

## 後方互換性

元のノートブックは `train_flow_ngpt_original.ipynb` として保存されています。

## テスト

ノートブックの構文検証が完了し、以下を確認しました：
- ✅ JSONフォーマットが正しい
- ✅ 全30セル（コードセル15個、マークダウンセル15個）
- ✅ リファクタリング後のimport文が正しい

## 注意事項

1. **実行環境**: Jupyter NotebookまたはJupyterLabで実行してください
2. **データセット**: `dataset/` ディレクトリに必要なCSVファイルが必要です
3. **設定ファイル**: `outputs/ngpt_classifier_config.json` が必要です
4. **GPU**: CUDA対応GPUを推奨（CPU でも動作しますが遅くなります）

## トラブルシューティング

### ImportError が発生する場合
```python
# プロジェクトルートをパスに追加
import sys
sys.path.insert(0, str(PROJECT_ROOT))
```

### ModuleNotFoundError が発生する場合
リファクタリング後のモジュール構造を確認してください：
```
utils/
├── config/
├── data/
├── training/
├── metrics/
└── visualization/
```

## 重要な修正点

### 関数呼び出しの引数を修正

元のノートブックは関数の引数が不正確でした。train.pyの実装を完全に再現するように修正：

1. **load_raw_datasets**: `seed`引数を追加
2. **prepare_label_mappings**: 10個の戻り値を正しく受け取る
3. **setup_model_and_config**: `classifier_configs`引数を追加
4. **prepare_datasets**: `aspect_key`, `sentence3_flag`引数を追加
5. **create_trainer**: `id2_head`引数を追加

### 各セルの内容

1. **セル0**: 環境設定とインポート
2. **セル1**: 引数の設定（ModelArguments, DataTrainingArguments, TrainingArguments）
3. **セル2**: データセット読み込み（sentence3_flag取得）
4. **セル3**: ラベルマッピング準備（10個の戻り値）
5. **セル4**: モデルとconfig設定（use_ngpt_riemann取得）
6. **セル5**: トークナイザー設定
7. **セル6**: データセット前処理（max_train_samples取得）
8. **セル7**: トレーナー作成（id2_head作成）
9. **セル8**: 初期評価（ベースライン）
10. **セル9**: トレーニング実行
11. **セル10**: テスト評価
12. **セル11**: 結果比較（改善率計算付き）
13. **セル12**: モデル保存（オプション）

## 変更履歴

- **2025-11-17 (第2版)**: 関数呼び出しの引数を修正
  - train.pyと完全に同じ引数を使用
  - 戻り値の取得を修正
  - エラーなく実行できることを確認
  
- **2025-11-17 (第1版)**: リファクタリング後のモジュール構造に対応
  - import文を新しいモジュールパスに更新
  - セル構成を改善
  - 詳細な出力とコメントを追加
  - 結果比較セルを追加
