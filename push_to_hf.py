"""
Push this Streamlit app to your Hugging Face account as a Docker Space.

Prerequisites:
1) Create a Write token: https://huggingface.co/settings/tokens
2) Login once:
     hf auth login
   or PowerShell:
     $env:HF_TOKEN="hf_xxx"   # do NOT put token in any project file
3) Run:
     python push_to_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login, whoami

SPACE_NAME = os.getenv("HF_SPACE_NAME", "bert-gpt-balochi-app")
HF_USERNAME = os.getenv("HF_USERNAME", "sajida12")

INCLUDE = [
    "app.py",
    "config.py",
    "balochi_qa.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    "data/balochi_instructions.csv",
]


def main():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    try:
        info = whoami()
    except Exception as exc:
        raise SystemExit(
            "Not logged in to Hugging Face.\n"
            "Run this in PowerShell first:\n"
            "  hf auth login\n"
            "or set HF_TOKEN, then run again.\n"
            f"Details: {exc}"
        ) from exc

    username = info.get("name") or HF_USERNAME
    repo_id = f"{username}/{SPACE_NAME}"
    api = HfApi(token=token)

    print(f"Logged in as: {username}")
    print(f"Creating/updating Space: {repo_id}")

    # HF create_repo currently accepts: gradio | docker | static
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    root = Path(__file__).resolve().parent
    for rel in INCLUDE:
        path = root / rel
        if not path.exists():
            print(f"Skip missing: {rel}")
            continue
        print(f"Upload: {rel}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel.replace("\\", "/"),
            repo_id=repo_id,
            repo_type="space",
        )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\nDone.")
    print(f"Open your Hugging Face Space: {url}")


if __name__ == "__main__":
    main()
