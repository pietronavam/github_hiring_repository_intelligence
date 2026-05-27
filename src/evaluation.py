"""
Evaluation & Error Analysis (Stage 6)

Beyond accuracy: why per-class metrics matter here
  With 6 imbalanced classes, macro-averaged F1 is more informative than
  accuracy. A model that always predicts "junior" (the most common class)
  can achieve high accuracy but fails recruiters who need to identify
  lead-level repos. We report per-class metrics and analyze confusion
  patterns to surface systematic errors.

Baseline comparison:
  We compare DistilBERT against a majority-class baseline (always predicts
  the most frequent class) and a TF-IDF + Logistic Regression baseline.
  This frames the BERT result in context — even weak supervision + simple
  models should beat the majority baseline significantly.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.utils import setup_logging, CATEGORIES

log = setup_logging("evaluation")

SPLITS_DIR = Path("data/splits")
MODEL_DIR = Path("models/trained_models/distilbert_hiring")
LABEL_MAP_PATH = Path("models/trained_models/label_map.json")
METRICS_DIR = Path("output/metrics")
TABLES_DIR = Path("output/tables")


def predict_bert(texts: list[str], label_map: dict) -> list[int]:
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    preds = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).tolist())
    return preds


def evaluate_baseline_majority(y_train: list, y_test: list) -> dict:
    majority = max(set(y_train), key=y_train.count)
    preds = [majority] * len(y_test)
    acc = accuracy_score(y_test, preds)
    _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    return {"model": "majority_baseline", "accuracy": acc, "f1_weighted": f1}


def evaluate_baseline_tfidf(X_train, y_train, X_test, y_test) -> dict:
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_test, preds)
    _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    return {"model": "tfidf_logreg", "accuracy": acc, "f1_weighted": f1}


def run_evaluation() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    with open(LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    label_names = [label_map[i] for i in sorted(label_map)]

    # DistilBERT predictions
    log.info("Running DistilBERT inference on test set...")
    bert_preds = predict_bert(X_test, label_map)
    bert_preds_names = [label_map[p] for p in bert_preds]
    y_test_names = [label_map[y] for y in y_test]

    # Metrics
    acc = accuracy_score(y_test, bert_preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, bert_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, bert_preds, labels=sorted(label_map.keys()))
    report = classification_report(y_test_names, bert_preds_names, output_dict=True, zero_division=0)

    bert_result = {"model": "distilbert", "accuracy": acc, "precision": p, "recall": r, "f1_weighted": f1}

    # Baselines
    majority_result = evaluate_baseline_majority(y_train, y_test)
    tfidf_result = evaluate_baseline_tfidf(X_train, y_train, X_test, y_test)

    # Save all outputs
    metrics_df = pd.DataFrame([majority_result, tfidf_result, bert_result])
    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        TABLES_DIR / "confusion_matrix.csv"
    )

    per_class = pd.DataFrame(report).T
    per_class.to_csv(TABLES_DIR / "per_class_metrics.csv")

    with open(METRICS_DIR / "bert_metrics.json", "w") as f:
        json.dump(bert_result, f, indent=2)

    np.save(METRICS_DIR / "confusion_matrix.npy", cm)
    with open(METRICS_DIR / "label_names.json", "w") as f:
        json.dump(label_names, f)

    log.info(f"\n{metrics_df.to_string(index=False)}")
    log.info(f"\nPer-class report:\n{classification_report(y_test_names, bert_preds_names, zero_division=0)}")


if __name__ == "__main__":
    run_evaluation()
