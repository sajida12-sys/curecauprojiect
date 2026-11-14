---
title: BERT GPT Balochi App
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Encoder BERT NER + Decoder GPT Balochi Q&A
---

# BERT + GPT Streamlit Assignment

One Streamlit app with **two sections**:

| Section | Model | Language | Task |
|--------|--------|----------|------|
| BERT | Encoder-only (`bert-base-multilingual-cased`) | High-resource / multilingual | Named Entity Recognition (NER) |
| GPT | Decoder-only (`gpt2`) | **Balochi only** (low-resource) | Text generation / Q&A |

This follows the course PDF: fine-tune both models, upload to **your** Hugging Face account, put both in **one** Streamlit app, deploy via GitHub → Streamlit Cloud / Hugging Face Spaces.

**Accounts**
- GitHub: https://github.com/sajida12-sys/curecauprojiect
- Hugging Face username: `sajida12`

---

## Project files

- `app.py` — Streamlit app (BERT tab + GPT tab)
- `config.py` — model names / Hugging Face username
- `balochi_qa.py` — Balochi question → correct answer matcher
- `train_bert_ner.py` — fine-tune multilingual BERT on WikiANN
- `train_gpt_balochi.py` — fine-tune GPT-2 on Balochi instructions
- `upload_models.py` — upload saved models to Hugging Face
- `push_to_hf.py` — push this app to your Hugging Face Space
- `data/` — WikiANN + Balochi datasets
- `requirements.txt` — Python packages

---

## Beginner steps (start → submit)

### 1) Install packages
```bash
pip install -r requirements.txt
```

### 2) Train BERT (NER)
```bash
python train_bert_ner.py
```
Model saves to: `models/bert-ner-mbert`

### 3) Train GPT (Balochi only)
```bash
python train_gpt_balochi.py
```
Model saves to: `models/gpt2-balochi`

### 4) Login to Hugging Face and upload
```bash
hf auth login
```
Then in PowerShell:
```powershell
$env:HF_USERNAME="sajida12"
python upload_models.py
python push_to_hf.py
```

### 5) Connect models to the app
`config.py` already uses Hugging Face username `sajida12`.
After you upload fine-tuned models, the app will automatically load:
- `sajida12/bert-ner-mbert`
- `sajida12/gpt2-balochi`

Until those repos exist, it safely uses public fallback models.

### 6) Test app on your computer
```bash
streamlit run app.py
```

### 7) GitHub
Repo: https://github.com/sajida12-sys/curecauprojiect

### 8) Deploy
- Streamlit Cloud: https://share.streamlit.io (connect GitHub repo, file `app.py`)
- Hugging Face Spaces: run `python push_to_hf.py` after `hf auth login`

---

## Security
- Never commit Hugging Face or GitHub tokens
- If a token was shared in chat, revoke it immediately
