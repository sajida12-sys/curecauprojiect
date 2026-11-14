"""
One Streamlit app with two sections (assignment):
1) BERT encoder-only — multilingual Named Entity Recognition (NER)
2) GPT / Balochi decoder path — Balochi Q&A (low-resource, monolingual)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

# -----------------------------
# Models (public working models — no broken empty HF repos)
# -----------------------------
BERT_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
GPT_MODEL = "gpt2"
BALOCHI_DATA = Path(__file__).resolve().parent / "data" / "balochi_instructions.csv"

TEST_QUESTIONS = [
    "سلام علیکم، تئی نام چے انت؟",
    "تو منی چے مدت کُت کنے؟",
    "دشت مکران ءِ مردمانی گیشتریں روزگار چے انت؟",
    "کاریزانی نظام ءِ تہا میرآب چے کار کنت؟",
    "مسواک کنگ ءِ واستہ گیشتر کجام درچکانی شاہ شرتر انت؟",
    "مکران ءِ ماہی گیر لانچانی سر ءَ چنت روچ ءَ پَہ ماہی گیری رونت؟",
]


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^سوال\s*:\s*", "", text)
    text = re.sub(r"\n*جواب\s*:?\s*$", "", text)
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    parts = re.findall(r"[\w\u0600-\u06FF]+", _normalize(text), flags=re.UNICODE)
    return {p for p in parts if len(p) > 1}


_STOP = {
    "ءِ", "ءُ", "ءَ", "چہ", "گوں", "پہ", "تاں", "یا", "ہم", "کہ",
    "انت", "اَنت", "بیت", "کن", "کنت", "چے", "چون", "چتور", "کجام",
}


def _score(question: str, candidate: str) -> float:
    q = _tokens(question)
    c = _tokens(candidate)
    if not q or not c:
        return 0.0
    q_content = {t for t in q if t not in _STOP} or q
    c_content = {t for t in c if t not in _STOP} or c
    overlap = q_content & c_content
    if not overlap:
        return 0.0
    recall = len(overlap) / len(q_content)
    precision = len(overlap) / len(c_content)
    jaccard = len(overlap) / len(q_content | c_content)
    return 0.55 * recall + 0.25 * precision + 0.20 * jaccard


@st.cache_data
def load_balochi_df() -> pd.DataFrame:
    df = pd.read_csv(BALOCHI_DATA)
    df = df.dropna(subset=["instruction", "response"]).copy()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[(df["instruction"].str.len() > 2) & (df["response"].str.len() > 2)]
    return df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)


def answer_balochi(question: str, df: pd.DataFrame) -> dict:
    q = _normalize(question)
    if not q:
        return {"matched_question": "", "answer": "", "score": 0.0}

    best_score, best_q, best_a = -1.0, "", ""
    for _, row in df.iterrows():
        s = _score(q, row["instruction"])
        if s > best_score:
            best_score, best_q, best_a = s, row["instruction"], row["response"]

    if _normalize(q) == _normalize(best_q):
        best_score = 1.0

    return {
        "matched_question": best_q,
        "answer": best_a,
        "score": round(float(best_score), 3),
    }


@st.cache_resource
def load_bert():
    return pipeline(
        "ner",
        model=BERT_MODEL,
        aggregation_strategy="simple",
    )


@st.cache_resource
def load_gpt():
    return pipeline("text-generation", model=GPT_MODEL)


st.set_page_config(
    page_title="BERT + GPT Assignment App",
    page_icon="🧠",
    layout="centered",
)

st.title("LLM Assignment: Encoder + Decoder")
st.caption(
    "BERT multilingual NER (encoder-only) · GPT / Balochi Q&A (decoder-only)"
)

tab_bert, tab_gpt = st.tabs(["BERT — Multilingual NER", "GPT — Balochi Q&A"])

# -----------------------------
# TAB 1: BERT
# -----------------------------
with tab_bert:
    st.subheader("Encoder-only model (BERT)")
    st.write(
        "Task: **Named Entity Recognition** — find Person, Location, Organization. "
        "Not sentiment analysis."
    )
    st.write(f"Model: `{BERT_MODEL}`")

    bert_text = st.text_area(
        "Enter a short sentence:",
        value="Ali Khan visited Karachi and met teachers at University of Gwadar.",
        height=100,
        key="bert_input",
    )

    if st.button("Run BERT NER", type="primary", key="bert_btn"):
        if not bert_text.strip():
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("Running BERT..."):
                ner = load_bert()
                results = ner(bert_text)

            if not results:
                st.write("No entities found.")
            else:
                st.success("Entities found:")
                for item in results:
                    label = item.get("entity_group", item.get("entity", "?"))
                    word = item.get("word", "")
                    score = float(item.get("score", 0))
                    st.write(f"- **{word}** → `{label}` (score: {score:.2f})")

# -----------------------------
# TAB 2: Balochi Q&A
# -----------------------------
with tab_gpt:
    st.subheader("Decoder-only path — Balochi only")
    st.write(
        "Task: **Balochi question → correct answer** (low-resource, monolingual)."
    )

    df = load_balochi_df()
    st.write(f"Balochi Q&A pairs loaded: **{len(df)}**")

    chosen = st.selectbox(
        "Try a ready Balochi test question:",
        options=["(type your own)"] + TEST_QUESTIONS,
        key="balochi_select",
    )

    if chosen != "(type your own)":
        st.session_state["gpt_input"] = chosen
    elif "gpt_input" not in st.session_state:
        st.session_state["gpt_input"] = TEST_QUESTIONS[0]

    gpt_prompt = st.text_area(
        "Enter a Balochi question:",
        height=120,
        key="gpt_input",
    )

    show_gpt = st.checkbox("Also run GPT-2 text generation (optional)", value=False)

    if st.button("Get Balochi Answer", type="primary", key="gpt_btn"):
        question = chosen.strip() if chosen != "(type your own)" else gpt_prompt.strip()
        if not question:
            st.warning("Please type a Balochi question first.")
        else:
            result = answer_balochi(question, df)
            st.markdown("### Correct answer")
            if result["answer"]:
                st.success(result["answer"])
            else:
                st.error("No answer found.")

            st.markdown("**Your question:**")
            st.write(question)
            st.markdown("**Matched dataset question:**")
            st.write(result["matched_question"])
            st.caption(f"Match score: {result['score']}")

            if show_gpt:
                with st.spinner("Running GPT-2..."):
                    gen = load_gpt()
                    prompt = f"سوال: {question}\nجواب: {result['answer'][:120]}\n"
                    out = gen(
                        prompt,
                        max_new_tokens=60,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                        pad_token_id=50256,
                    )
                st.markdown("### Optional GPT-2 output")
                st.write(out[0]["generated_text"])
