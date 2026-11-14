# BERT + GPT Streamlit Assignment

One Streamlit app with **two sections**:

| Section | Model type | Language | Task |
|--------|------------|----------|------|
| BERT | Encoder-only | Multilingual / high-resource | Named Entity Recognition (NER) |
| GPT / Balochi | Decoder-only path | **Balochi only** (low-resource) | Question → Answer |

## Important files
- `app.py` — main Streamlit application
- `requirements.txt` — Python packages
- `README.md` — this file
- `data/balochi_instructions.csv` — Balochi Q&A dataset used by the GPT section

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
- GitHub: https://github.com/sajida12-sys/curecauprojiect
- Streamlit Cloud: https://share.streamlit.io → select this repo → main file `app.py`

## Hugging Face
- Username: `sajida12`
- Space: https://huggingface.co/spaces/sajida12/balochi-gpt-test
- BERT model card: https://huggingface.co/sajida12/bert-ner-mbert
- GPT model card: https://huggingface.co/sajida12/gpt2-balochi

## How to test
1. **BERT tab:** enter `Ali Khan visited Karachi.` → see Person / Location entities
2. **GPT / Balochi tab:** pick a ready Balochi question → get the matching answer
