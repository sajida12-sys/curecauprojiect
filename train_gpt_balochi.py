"""
Fine-tune GPT-2 (decoder-only) on Balochi ONLY (low-resource, monolingual).

Task: Balochi instruction → response text generation.
NOT sentiment analysis. Do not mix other languages.
"""

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Settings you can change
# -----------------------------
BASE_MODEL = "gpt2"
DATA_PATH = Path("data/balochi_instructions.csv")
OUTPUT_DIR = Path("models/gpt2-balochi")
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 5e-5


def build_text(row) -> str:
    instruction = str(row["instruction"]).strip()
    response = str(row["response"]).strip()
    return f"سوال: {instruction}\nجواب: {response}"


def main():
    print("1) Loading Balochi dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["instruction", "response"])
    texts = [build_text(row) for _, row in df.iterrows()]
    print(f"Samples: {len(texts)}")

    # Simple split: 90% train, 10% validation
    split = int(len(texts) * 0.9)
    train_texts = texts[:split]
    val_texts = texts[split:]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    print("2) Loading GPT-2 tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # GPT-2 has no pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.eos_token_id

    print("3) Tokenizing...")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    # For causal LM, labels = input_ids
    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    train_tok = train_tok.map(add_labels, batched=True)
    val_tok = val_tok.map(add_labels, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("5) Saving model locally...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved to: {OUTPUT_DIR.resolve()}")

    # Quick sample generation
    print("6) Quick test generation...")
    prompt = "سوال: سلام علیکم، تئی نام چے انت؟\nجواب:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    hf_username = os.getenv("HF_USERNAME", "").strip()
    if hf_username:
        repo_id = f"{hf_username}/gpt2-balochi"
        print(f"7) Uploading to Hugging Face: {repo_id}")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print("Upload done.")
    else:
        print(
            "7) Skip upload. Later set HF_USERNAME and re-run upload, "
            "or upload the saved folder from Hugging Face website."
        )


if __name__ == "__main__":
    main()
