# !pip -q install transformers datasets accelerate evaluate scikit-learn torch

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
import evaluate
from hate_preproc import PreprocessConfig, preprocess_dataframe
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

SEED = 42
set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
TRAIN_CSV = "/content/drive/MyDrive/Hate Speech/Data/hate_speech_train.csv"   
TEXT_COL_RAW = "text"                
LABEL_COL = "label"                

OUTPUT_DIR = "/content/bert_finetuned"  # where model checkpoints will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(TRAIN_CSV)

# safety checks
if TEXT_COL_RAW not in df.columns:
    raise ValueError(f"Expected text column '{TEXT_COL_RAW}' not found. Columns: {list(df.columns)}")
if LABEL_COL not in df.columns:
    raise ValueError(f"Expected label column '{LABEL_COL}' not found. Columns: {list(df.columns)}")

# Preprocessing
# For BERT: keep preprocessing minimal 
cfg = PreprocessConfig(text_col=TEXT_COL_RAW, label_col=LABEL_COL) # default settings
df = preprocess_dataframe(df, cfg, out_col="text_clean")

# Train/validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df[LABEL_COL],   # preserves class proportions
)

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# Tokenizer
MODEL_CHECKPOINT = "bert-base-uncased" # Base BERT model

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

MAX_LEN = 512  # adjust: 128 (faster) / 256 (balanced) / 512 (slower, more context)

def tokenize_batch(batch):
    """
    Convert raw text into model inputs:
      - input_ids
      - attention_mask
    """
    return tokenizer(
        batch["text_clean"],
        truncation=True,
        max_length=MAX_LEN,
    )

train_ds = train_ds.map(tokenize_batch, batched=True)
val_ds = val_ds.map(tokenize_batch, batched=True)

# Trainer expects the label column to be named "labels"
train_ds = train_ds.rename_column(LABEL_COL, "labels")
val_ds = val_ds.rename_column(LABEL_COL, "labels")

# Keep only the columns needed by the model
keep_cols = ["input_ids", "attention_mask", "labels"]
if "token_type_ids" in train_ds.column_names:
    keep_cols.append("token_type_ids")

train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

# Dynamic padding batches to the longest sample in each batch (more efficient than padding to MAX_LEN)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Handle class imbalance
# Compute weights so minority class is not ignored.
y_train = train_df[LABEL_COL].values
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class counts (train):", np.bincount(y_train))
print("Class weights:", class_weights)

# Define a custom Trainer to use weighted loss
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Model initialization
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2
)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

# Training arguments:
# - learning_rate 2e-5 is a standard starting point for BERT
# - num_train_epochs 3 is a common baseline
# - weight_decay helps regularize
# - load_best_model_at_end selects best checkpoint by metric_for_best_model
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    report_to="none", 
    seed=SEED,
)

# Train
trainer = WeightedLossTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate best model on validation set
metrics = trainer.evaluate()
print("Validation metrics:", metrics)



# ------------------------
# Testing
# ------------------------

# Load test data
testdf = pd.read_csv("/content/drive/MyDrive/Hate Speech/Data/hate_speech_test.csv")   # path in your repo

# Apply the same preprocessing used in training
cfg = PreprocessConfig()
test_df = preprocess_dataframe(testdf,cfg)

# Convert to HF Dataset and tokenize
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text_clean"], truncation=True, max_length=512)

test_ds = test_ds.map(tokenize, batched=True)

# remove non-tensor columns before prediction
cols_to_keep = ["input_ids", "attention_mask"]
if "token_type_ids" in test_ds.column_names:
    cols_to_keep.append("token_type_ids")

test_ds_model = test_ds.remove_columns([c for c in test_ds.column_names if c not in cols_to_keep])

# Ensure collator matches tokenizer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4) Predict
pred_out = trainer.predict(test_ds_model)
logits = pred_out.predictions
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
pred_labels = probs.argmax(axis=1)

# 5) Attach predictions back to ids/text
out = test_df[["id"]].copy()
out["label"] = pred_labels

out.head()
out.to_csv("bert_test_predictions.csv", index=False)
