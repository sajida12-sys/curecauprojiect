# BERT + GPT Streamlit Assignment

One Streamlit app with **two sections**:

| Section | Model | Language | Task |
|--------|--------|----------|------|
| BERT | Encoder-only (`bert-base-multilingual-cased`) | High-resource / multilingual | Named Entity Recognition (NER) |
| GPT | Decoder-only (`gpt2`) | **Balochi only** (low-resource) | Text generation / Q&A |

This follows the course PDF: fine-tune both models, upload to **your** Hugging Face account, put both in **one** Streamlit app, deploy via GitHub → Streamlit Cloud.

---

## Project files

- `app.py` — Streamlit app (BERT tab + GPT tab)
- `config.py` — model names / Hugging Face username
- `train_bert_ner.py` — fine-tune multilingual BERT on WikiANN
- `train_gpt_balochi.py` — fine-tune GPT-2 on Balochi instructions
- `upload_models.py` — upload saved models to Hugging Face
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
huggingface-cli login
```
Then in PowerShell:
```powershell
$env:HF_USERNAME="your_hf_username"
python upload_models.py
```

### 5) Connect models to the app
Edit `config.py`:
1. Set `HF_USERNAME = "your_hf_username"`
2. Set `USE_FINETUNED = True`

### 6) Test app on your computer
```bash
streamlit run app.py
```

### 7) Push this project to GitHub
Use the safe commands in the chat / below. **Never put tokens inside project files.**

### 8) Deploy on Streamlit Cloud
1. Open https://share.streamlit.io
2. Sign in with GitHub
3. New app → select this repo → main file `app.py`
4. Deploy
5. Copy the public link for submission

---

## GitHub repo
https://github.com/sajida12-sys/curecauprojiect

## Security
- Never commit Hugging Face or GitHub tokens
- If a token was shared in chat, revoke it immediately on GitHub Settings → Developer settings → Personal access tokens
