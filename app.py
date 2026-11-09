"""
One Streamlit app with two sections:
1) BERT (encoder-only) — multilingual Named Entity Recognition (NER)
2) GPT (decoder-only) — Balochi monolingual text generation
"""

import streamlit as st
from transformers import pipeline

from config import get_bert_model_id, get_gpt_model_id


st.set_page_config(
    page_title="BERT + GPT Assignment App",
    page_icon="🧠",
    layout="centered",
)

st.title("LLM Assignment: Encoder + Decoder")
st.caption(
    "Section 1: BERT multilingual NER (encoder-only) · "
    "Section 2: GPT Balochi generation (decoder-only)"
)


@st.cache_resource
def load_bert_ner(model_id: str):
    return pipeline(
        "ner",
        model=model_id,
        aggregation_strategy="simple",
    )


@st.cache_resource
def load_gpt_generator(model_id: str):
    return pipeline(
        "text-generation",
        model=model_id,
    )


tab_bert, tab_gpt = st.tabs(
    ["BERT — Multilingual NER", "GPT — Balochi Generation"]
)

# -----------------------------
# TAB 1: BERT (Encoder-only)
# -----------------------------
with tab_bert:
    st.subheader("Encoder-only model (BERT / mBERT)")
    st.write(
        "Task: **Named Entity Recognition** — find Person, Location, Organization. "
        "This is text understanding, not sentiment analysis."
    )

    bert_model_id = get_bert_model_id()
    st.write(f"Model: `{bert_model_id}`")

    bert_text = st.text_area(
        "Enter a short sentence (English or another high-resource language):",
        value="Ali Khan visited Karachi and met teachers at University of Gwadar.",
        height=100,
        key="bert_input",
    )

    if st.button("Run BERT NER", type="primary", key="bert_btn"):
        if not bert_text.strip():
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("Loading / running BERT..."):
                ner = load_bert_ner(bert_model_id)
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

                with st.expander("Raw model output"):
                    st.json(results)

# -----------------------------
# TAB 2: GPT (Decoder-only)
# -----------------------------
with tab_gpt:
    st.subheader("Decoder-only model (GPT-2) — Balochi only")
    st.write(
        "Task: **Balochi text generation / Q&A** (low-resource, monolingual). "
        "Use Balochi script only — not Urdu or other languages."
    )

    gpt_model_id = get_gpt_model_id()
    st.write(f"Model: `{gpt_model_id}`")

    gpt_prompt = st.text_area(
        "Enter a short Balochi prompt / question:",
        value="سوال: سلام علیکم، تئی نام چے انت؟\nجواب:",
        height=120,
        key="gpt_input",
    )
    max_tokens = st.slider("Max new tokens", 20, 150, 60)

    if st.button("Run GPT Balochi", type="primary", key="gpt_btn"):
        if not gpt_prompt.strip():
            st.warning("Please type a Balochi prompt first.")
        else:
            with st.spinner("Loading / running GPT..."):
                gen = load_gpt_generator(gpt_model_id)
                out = gen(
                    gpt_prompt,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    pad_token_id=50256,
                )
            text = out[0]["generated_text"]
            st.success("Generated text:")
            st.write(text)

            with st.expander("Raw model output"):
                st.json(out)

st.markdown("---")
st.caption(
    "Assignment: one live Streamlit app · BERT encoder + GPT decoder · "
    "Balochi for low-resource GPT section"
)
