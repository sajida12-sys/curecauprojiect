"""
Push project cards/repos to Hugging Face account: sajida12

Notes:
- HF free tier: Static Spaces are free.
- Streamlit/Docker/Gradio Spaces on free CPU now require Hugging Face PRO.
- This script creates/updates:
  1) Static Space: sajida12/balochi-gpt-test
  2) Model repos: sajida12/bert-ner-mbert , sajida12/gpt2-balochi

For the interactive Streamlit app, deploy from GitHub:
https://share.streamlit.io  →  sajida12-sys/curecauprojiect  → app.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login, whoami

HF_USERNAME = os.getenv("HF_USERNAME", "sajida12")
SPACE_ID = f"{HF_USERNAME}/balochi-gpt-test"
BERT_ID = f"{HF_USERNAME}/bert-ner-mbert"
GPT_ID = f"{HF_USERNAME}/gpt2-balochi"


def main():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    info = whoami()
    username = info.get("name") or HF_USERNAME
    api = HfApi()
    root = Path(__file__).resolve().parent

    print(f"Logged in as: {username}")

    # Free static Space (Streamlit SDK is no longer available via API)
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(root / "hf_static" / "index.html"),
        path_in_repo="index.html",
        repo_id=SPACE_ID,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(root / "README.md"),
        path_in_repo="README.md",
        repo_id=SPACE_ID,
        repo_type="space",
    )
    print(f"Space ready: https://huggingface.co/spaces/{SPACE_ID}")

    # Model repos + cards
    for repo_id, readme in [
        (BERT_ID, root / "model_cards" / "bert-ner-mbert-README.md"),
        (GPT_ID, root / "model_cards" / "gpt2-balochi-README.md"),
    ]:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Model ready: https://huggingface.co/{repo_id}")

    print("\nInteractive app: deploy GitHub repo on Streamlit Cloud.")
    print("https://github.com/sajida12-sys/curecauprojiect")


if __name__ == "__main__":
    main()
