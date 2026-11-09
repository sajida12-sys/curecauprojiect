"""
Update these names AFTER you upload your fine-tuned models to Hugging Face.
Replace YOUR_HF_USERNAME with your real Hugging Face username.
"""

# Your Hugging Face username (example: sajida12-sys)
HF_USERNAME = "sajida12"

# Fine-tuned model repos on YOUR Hugging Face account
BERT_MODEL_ID = f"{HF_USERNAME}/bert-ner-mbert"
GPT_MODEL_ID = f"{HF_USERNAME}/gpt2-balochi"

# Temporary public models so the app can run BEFORE you finish fine-tuning.
# After upload, set USE_FINETUNED = True
USE_FINETUNED = False

FALLBACK_BERT_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
FALLBACK_GPT_MODEL = "gpt2"


def get_bert_model_id() -> str:
    return BERT_MODEL_ID if USE_FINETUNED else FALLBACK_BERT_MODEL


def get_gpt_model_id() -> str:
    return GPT_MODEL_ID if USE_FINETUNED else FALLBACK_GPT_MODEL
