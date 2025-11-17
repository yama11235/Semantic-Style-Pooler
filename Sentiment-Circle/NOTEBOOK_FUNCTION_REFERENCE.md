# train_flow_ngpt.ipynb 関数呼び出しリファレンス

## 各セルの関数呼び出しと引数

このドキュメントは、train.pyとnotebookで使用される関数呼び出しの対応を示します。

### セル2: データセット読み込み

```python
# train.pyの呼び出し（104-109行目）
raw_datasets, sentence3_flag = load_raw_datasets(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    seed=seed,
)
```

**引数:**
- `model_args`: ModelArguments
- `data_args`: DataTrainingArguments  
- `training_args`: TrainingArguments
- `seed`: int (42)

**戻り値:**
- `raw_datasets`: DatasetDict
- `sentence3_flag`: bool

---

### セル3: ラベルマッピング準備

```python
# train.pyの呼び出し（112-127行目）
(
    raw_datasets,
    labels,
    id2label,
    label2id,
    aspect_key,
    classifier_configs,
    classifier_configs_for_trainer,
    corr_labels,
    corr_weights,
    label_name_mappings,
) = prepare_label_mappings(
    raw_datasets=raw_datasets,
    model_args=model_args,
    data_args=data_args,
)
```

**引数:**
- `raw_datasets`: DatasetDict
- `model_args`: ModelArguments
- `data_args`: DataTrainingArguments

**戻り値:** (10個のタプル)
1. `raw_datasets`: DatasetDict (更新済み)
2. `labels`: List[str]
3. `id2label`: Dict[int, str]
4. `label2id`: Dict[str, int]
5. `aspect_key`: List[str]
6. `classifier_configs`: Dict
7. `classifier_configs_for_trainer`: Dict
8. `corr_labels`: Optional[Dict]
9. `corr_weights`: Optional[Dict]
10. `label_name_mappings`: Dict

---

### セル4: モデルとconfig設定

```python
# train.pyの呼び出し（130-137行目）
config, model, use_ngpt_riemann = setup_model_and_config(
    model_args=model_args,
    training_args=training_args,
    labels=list(classifier_configs_for_trainer.keys()),
    id2label=id2label,
    label2id=label2id,
    classifier_configs=classifier_configs,
)
```

**引数:**
- `model_args`: ModelArguments
- `training_args`: TrainingArguments
- `labels`: List[str] (classifier名のリスト)
- `id2label`: Dict[int, str]
- `label2id`: Dict[str, int]
- `classifier_configs`: Dict

**戻り値:**
- `config`: AutoConfig
- `model`: BiEncoderForClassification
- `use_ngpt_riemann`: bool

---

### セル5: トークナイザー設定

```python
# train.pyの呼び出し（140行目）
tokenizer = setup_tokenizer(model_args)
```

**引数:**
- `model_args`: ModelArguments

**戻り値:**
- `tokenizer`: AutoTokenizer

---

### セル6: データセット前処理

```python
# train.pyの呼び出し（143-151行目）
train_dataset, eval_dataset, predict_dataset, max_train_samples = prepare_datasets(
    raw_datasets=raw_datasets,
    tokenizer=tokenizer,
    data_args=data_args,
    model_args=model_args,
    training_args=training_args,
    aspect_key=aspect_key,
    sentence3_flag=sentence3_flag,
)
```

**引数:**
- `raw_datasets`: DatasetDict
- `tokenizer`: AutoTokenizer
- `data_args`: DataTrainingArguments
- `model_args`: ModelArguments
- `training_args`: TrainingArguments
- `aspect_key`: List[str]
- `sentence3_flag`: bool

**戻り値:**
- `train_dataset`: Dataset
- `eval_dataset`: Dataset
- `predict_dataset`: Dataset
- `max_train_samples`: int

---

### セル7: トレーナー作成

```python
# train.pyの呼び出し（162-176行目）
id2_head = {i: head for i, head in enumerate(classifier_configs_for_trainer.keys())}

trainer, trainer_state = create_trainer(
    model=model,
    config=config,
    training_args=training_args,
    classifier_configs_for_trainer=classifier_configs_for_trainer,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    corr_labels=corr_labels,
    corr_weights=corr_weights,
    label_name_mappings=label_name_mappings,
    use_ngpt_riemann=use_ngpt_riemann,
    id2_head=id2_head,
)
```

**引数:**
- `model`: BiEncoderForClassification
- `config`: AutoConfig
- `training_args`: TrainingArguments
- `classifier_configs_for_trainer`: Dict
- `tokenizer`: AutoTokenizer
- `train_dataset`: Dataset
- `eval_dataset`: Dataset
- `corr_labels`: Optional[Dict]
- `corr_weights`: Optional[Dict]
- `label_name_mappings`: Dict
- `use_ngpt_riemann`: bool
- `id2_head`: Dict[int, str]

**戻り値:**
- `trainer`: CustomTrainer
- `trainer_state`: Dict

---

## 依存関係グラフ

```
seed (42)
   ↓
[load_raw_datasets] ← model_args, data_args, training_args
   ↓
raw_datasets, sentence3_flag
   ↓
[prepare_label_mappings] ← model_args, data_args
   ↓
raw_datasets, labels, id2label, label2id, aspect_key,
classifier_configs, classifier_configs_for_trainer,
corr_labels, corr_weights, label_name_mappings
   ↓
[setup_model_and_config] ← labels, id2label, label2id, classifier_configs
   ↓
config, model, use_ngpt_riemann
   ↓
[setup_tokenizer] ← model_args
   ↓
tokenizer
   ↓
[prepare_datasets] ← raw_datasets, aspect_key, sentence3_flag
   ↓
train_dataset, eval_dataset, predict_dataset, max_train_samples
   ↓
[create_trainer] ← all previous outputs
   ↓
trainer, trainer_state
```

## 重要な注意点

1. **引数の順序**: すべての引数をキーワード引数として渡す
2. **戻り値の数**: `prepare_label_mappings`は10個の値を返す
3. **id2_head**: `create_trainer`の前に手動で作成する必要がある
4. **sentence3_flag**: データセット前処理で使用される
5. **classifier_configs vs classifier_configs_for_trainer**: 異なるオブジェクト

## エラーが発生する場合のチェックリスト

- [ ] すべての引数が正しい型か
- [ ] 戻り値の数が正しいか（特にprepare_label_mappings）
- [ ] 依存関係の順序が正しいか
- [ ] model_argsのclassifier_configsパスが存在するか
- [ ] data_argsのファイルパスが存在するか
- [ ] GPUが利用可能か（CPUでも動作するが遅い）
