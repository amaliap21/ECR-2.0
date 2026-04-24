"""
Step 14: fine-tune cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
on the 96K-row silver-label training set built by step 13.
Uses class weights on the loss to counter the negative-heavy imbalance
(~75%/14%/11%).

Outputs:
    Leiden/finetuned_sentiment_cardiff_xlmroberta/
    config.json, model.safetensors, tokenizer.json, ...
    Leiden/evaluation/outputs/cardiff_xlmroberta/
    val_metrics.json, test_metrics.json,
    test_classification_report.txt
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          set_seed)
from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "dataset_cardiff_xlmroberta")
MODEL_OUT = os.path.join(ROOT, "finetuned_sentiment_cardiff_xlmroberta")
LOG_DIR   = os.path.join(HERE, "outputs", "cardiff_xlmroberta")

BASE_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
SEED = 42

LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_split(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


class WeightedTrainer(Trainer):
    """Trainer variant that applies class weights to the cross-entropy loss."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_neg":   f1_score(labels, preds, labels=[0], average="macro", zero_division=0),
        "f1_neu":   f1_score(labels, preds, labels=[1], average="macro", zero_division=0),
        "f1_pos":   f1_score(labels, preds, labels=[2], average="macro", zero_division=0),
    }


def main():
    set_seed(SEED)
    os.makedirs(MODEL_OUT, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")
    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # --- class weights from training distribution ---
    counts = train_df["label_id"].value_counts().sort_index().values.astype(float)
    weights = counts.sum() / (len(counts) * counts)  # inverse-freq, normalized
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"class weights: {dict(zip(LABEL_LIST, weights.round(3)))}")

    # --- tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tok_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=128)
        out["labels"] = batch["label_id"]
        return out

    def to_ds(df):
        return Dataset.from_pandas(df[["text", "label_id"]]).map(
            tok_fn, batched=True, remove_columns=["text", "label_id"])

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    # --- model (re-init classification head to match our 3-class label order) ---
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True)

    # --- training args ---
    args = TrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # --- final val metrics ---
    val_m = trainer.evaluate(val_ds)
    print("VAL:", val_m)
    with open(os.path.join(LOG_DIR, "val_metrics.json"), "w") as f:
        json.dump(val_m, f, indent=2)

    # --- test metrics ---
    test_pred = trainer.predict(test_ds)
    test_preds = np.argmax(test_pred.predictions, axis=-1)
    test_labels = test_df["label_id"].values
    report = classification_report(test_labels, test_preds,
                                    target_names=LABEL_LIST,
                                    digits=4, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2])
    test_m = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "macro_f1": f1_score(test_labels, test_preds, average="macro", zero_division=0),
        "precision_macro": precision_score(test_labels, test_preds,
                                             average="macro", zero_division=0),
        "recall_macro": recall_score(test_labels, test_preds,
                                       average="macro", zero_division=0),
    }
    print("TEST:", test_m)
    print(report)
    print(f"Confusion matrix (rows=true, cols=pred, order={LABEL_LIST}):\n{cm}")
    with open(os.path.join(LOG_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    with open(os.path.join(LOG_DIR, "test_classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nConfusion matrix (rows=true, cols=pred, order={LABEL_LIST}):\n{cm}\n")

    # --- save final model (best epoch, already loaded via load_best_model_at_end) ---
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"\nFinal model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
