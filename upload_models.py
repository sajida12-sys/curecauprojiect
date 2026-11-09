"""
Upload already-trained local models to YOUR Hugging Face account.

Prerequisites:
1) huggingface-cli login   (or paste token when asked)
2) models already saved in models/bert-ner-mbert and models/gpt2-balochi
"""

import os
from pathlib import Path

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)

HF_USERNAME = os.getenv("HF_USERNAME", "").strip()
if not HF_USERNAME or HF_USERNAME == "YOUR_HF_USERNAME":
    raise SystemExit(
        "Set your Hugging Face username first:\n"
        "  PowerShell:  $env:HF_USERNAME='your_hf_username'"
    )

BERT_DIR = Path("models/bert-ner-mbert")
GPT_DIR = Path("models/gpt2-balochi")


def upload_bert():
    if not BERT_DIR.exists():
        print(f"Skip BERT: folder not found: {BERT_DIR}")
        return
    repo = f"{HF_USERNAME}/bert-ner-mbert"
    print(f"Uploading BERT to {repo} ...")
    model = AutoModelForTokenClassification.from_pretrained(BERT_DIR)
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    model.push_to_hub(repo)
    tok.push_to_hub(repo)
    print("BERT upload done.")


def upload_gpt():
    if not GPT_DIR.exists():
        print(f"Skip GPT: folder not found: {GPT_DIR}")
        return
    repo = f"{HF_USERNAME}/gpt2-balochi"
    print(f"Uploading GPT to {repo} ...")
    model = AutoModelForCausalLM.from_pretrained(GPT_DIR)
    tok = AutoTokenizer.from_pretrained(GPT_DIR)
    model.push_to_hub(repo)
    tok.push_to_hub(repo)
    print("GPT upload done.")


if __name__ == "__main__":
    upload_bert()
    upload_gpt()
    print("\nNext:")
    print("1) Open config.py")
    print("2) Set HF_USERNAME to your username")
    print("3) Set USE_FINETUNED = True")
