"""
Fine-tune multilingual BERT (encoder-only) for Named Entity Recognition.

Task: NER on WikiANN (high-resource / multilingual) — NOT sentiment analysis.
Beginner-friendly: run in Google Colab or locally with GPU if available.

After training, upload the model to YOUR Hugging Face account.
"""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Settings you can change
# -----------------------------
BASE_MODEL = "bert-base-multilingual-cased"  # mBERT
DATA_DIR = Path("data")
OUTPUT_DIR = Path("models/bert-ner-mbert")
MAX_LENGTH = 128
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

# WikiANN label names (standard 7 labels)
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def parse_list_cell(value):
    """Convert CSV list-like strings into real Python lists."""
    if isinstance(value, list):
        return value
    text = str(value).strip()
    try:
        return ast.literal_eval(text)
    except Exception:
        # Fallback for numpy-like strings: [1 2 3]
        text = text.replace("[", "").replace("]", "")
        parts = text.split()
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except ValueError:
                out.append(p.strip("'\""))
        return out


def load_wikiann_csv(split_name: str, max_rows: int | None = None) -> Dataset:
    path = DATA_DIR / f"wikiann_{split_name}.csv"
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)

    tokens_col = []
    tags_col = []
    for _, row in df.iterrows():
        tokens = parse_list_cell(row["tokens"])
        tags = parse_list_cell(row["ner_tags"])
        # Keep only rows where lengths match
        if len(tokens) == len(tags) and len(tokens) > 0:
            tokens_col.append([str(t) for t in tokens])
            tags_col.append([int(t) for t in tags])

    return Dataset.from_dict({"tokens": tokens_col, "ner_tags": tags_col})


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                # For subword tokens, usually ignore or keep I- tag; ignore is safer for beginners
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def main():
    print("1) Loading WikiANN CSV files...")
    # Use a smaller subset first if your computer is slow
    raw = DatasetDict(
        {
            "train": load_wikiann_csv("train", max_rows=3000),
            "validation": load_wikiann_csv("validation", max_rows=800),
            "test": load_wikiann_csv("test", max_rows=800),
        }
    )
    print(raw)

    print("2) Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("3) Preprocessing...")
    tokenized = raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    print("4) Training...")
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("5) Testing...")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(test_metrics)

    print("6) Saving model locally...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved to: {OUTPUT_DIR.resolve()}")

    # Optional upload (set HF_USERNAME env var first)
    hf_username = os.getenv("HF_USERNAME", "").strip()
    if hf_username:
        repo_id = f"{hf_username}/bert-ner-mbert"
        print(f"7) Uploading to Hugging Face: {repo_id}")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print("Upload done.")
    else:
        print(
            "7) Skip upload. Later run:\n"
            "   set HF_USERNAME=your_hf_username\n"
            "   python train_bert_ner.py\n"
            "Or upload manually from the saved folder."
        )


if __name__ == "__main__":
    main()
