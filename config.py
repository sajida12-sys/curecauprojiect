"""
Model settings for the Streamlit app.
Hugging Face username: sajida12
"""

HF_USERNAME = "sajida12"

# Your fine-tuned model repos (create these after training + upload)
BERT_MODEL_ID = f"{HF_USERNAME}/bert-ner-mbert"
GPT_MODEL_ID = f"{HF_USERNAME}/gpt2-balochi"

# Working public models used automatically if fine-tuned repos are not ready yet
FALLBACK_BERT_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
FALLBACK_GPT_MODEL = "gpt2"


def resolve_model_id(preferred: str, fallback: str) -> str:
    """Use your HF model if it exists; otherwise use the public fallback."""
    try:
        from huggingface_hub import model_info

        model_info(preferred)
        return preferred
    except Exception:
        return fallback


def get_bert_model_id() -> str:
    return resolve_model_id(BERT_MODEL_ID, FALLBACK_BERT_MODEL)


def get_gpt_model_id() -> str:
    return resolve_model_id(GPT_MODEL_ID, FALLBACK_GPT_MODEL)
