"""
Model settings for the Streamlit app.
Hugging Face username: sajida12
"""

HF_USERNAME = "sajida12"

# Your fine-tuned model repos (use only after real weights are uploaded)
BERT_MODEL_ID = f"{HF_USERNAME}/bert-ner-mbert"
GPT_MODEL_ID = f"{HF_USERNAME}/gpt2-balochi"

# Working public models used when fine-tuned repos have no model files yet
FALLBACK_BERT_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
FALLBACK_GPT_MODEL = "gpt2"


def _repo_has_model_files(repo_id: str) -> bool:
    """True only if the HF repo contains a real model config (not README-only)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = set(list_repo_files(repo_id, repo_type="model"))
        if "config.json" not in files:
            return False
        # Confirm config.json can be fetched
        hf_hub_download(repo_id=repo_id, filename="config.json")
        return True
    except Exception:
        return False


def resolve_model_id(preferred: str, fallback: str) -> str:
    """Use your HF model only if it has real weights/config; else public fallback."""
    if _repo_has_model_files(preferred):
        return preferred
    return fallback


def get_bert_model_id() -> str:
    return resolve_model_id(BERT_MODEL_ID, FALLBACK_BERT_MODEL)


def get_gpt_model_id() -> str:
    return resolve_model_id(GPT_MODEL_ID, FALLBACK_GPT_MODEL)
