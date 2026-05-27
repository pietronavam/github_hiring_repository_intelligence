"""
BERT Fine-Tuning (Stage 5)

Model choice: DistilBERT (distilbert-base-uncased)
  - 40% smaller than BERT-base, 60% faster, retains 97% of BERT's performance
  - Fits in CPU RAM (< 4GB) enabling training without a GPU
  - Pre-trained on English text; repository descriptions and signal tokens
    are English-dominant, making this a strong baseline

Why fine-tuning beats zero-shot classification:
  The LLM labels define a specific rubric. Fine-tuning BERT on those labels
  teaches the model THIS rubric, not a general notion of code quality.
  At inference time we get fast, cheap predictions without LLM API calls.

Train/Val/Test split: 70/15/15
  - Stratified by label to preserve class ratios in all splits
  - Test set locked and not touched during hyperparameter tuning
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
from src.utils import setup_logging, CATEGORIES

log = setup_logging("train")

LABELED_PATH = Path("data/labeled/repositories_labeled.csv")
SPLITS_DIR = Path("data/splits")
MODEL_DIR = Path("models/trained_models/distilbert_hiring")
LABEL_MAP_PATH = Path("models/trained_models/label_map.json")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
CONFIDENCE_THRESHOLD = 0.6


class RepoDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {**acc, **f1}


def prepare_splits(df: pd.DataFrame) -> tuple:
    df = df[df["confidence"] >= CONFIDENCE_THRESHOLD].dropna(subset=["label", "bert_input"]).copy()
    df = df[df["label"].isin(CATEGORIES)].copy()

    le = LabelEncoder()
    le.fit(CATEGORIES)
    df["label_id"] = le.transform(df["label"])

    label_map = {int(i): str(cat) for i, cat in enumerate(le.classes_)}
    LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)
    log.info(f"Label map: {label_map}")

    texts = df["bert_input"].tolist()
    labels = df["label_id"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(SPLITS_DIR / "train.csv", index=False)
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv(SPLITS_DIR / "val.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(SPLITS_DIR / "test.csv", index=False)

    log.info(f"Split sizes — train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, label_map


def train(input_path: Path = LABELED_PATH) -> None:
    df = pd.read_csv(input_path)
    X_train, X_val, X_test, y_train, y_val, y_test, label_map = prepare_splits(df)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    num_labels = len(label_map)

    train_dataset = RepoDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = RepoDataset(X_val, y_val, tokenizer, MAX_LENGTH)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=label_map,
        label2id={v: k for k, v in label_map.items()},
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=10,
        report_to="none",
        no_cuda=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    log.info(f"Model saved → {MODEL_DIR}")


if __name__ == "__main__":
    train()
