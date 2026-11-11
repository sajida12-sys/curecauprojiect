"""
One Streamlit app with two sections:
1) BERT (encoder-only) — multilingual Named Entity Recognition (NER)
2) GPT / Balochi (decoder-only path) — Balochi Q&A with correct answer based on question
"""

import streamlit as st
from transformers import pipeline

from balochi_qa import answer_balochi_question, load_balochi_qa, sample_test_questions
from config import get_bert_model_id, get_gpt_model_id


st.set_page_config(
    page_title="BERT + GPT Assignment App",
    page_icon="🧠",
    layout="centered",
)

st.title("LLM Assignment: Encoder + Decoder")
st.caption(
    "Section 1: BERT multilingual NER (encoder-only) · "
    "Section 2: GPT Balochi Q&A (decoder-only / low-resource)"
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


@st.cache_data
def get_balochi_df():
    return load_balochi_qa()


@st.cache_data
def get_test_questions():
    return sample_test_questions(12)


tab_bert, tab_gpt = st.tabs(
    ["BERT — Multilingual NER", "GPT — Balochi Q&A"]
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
# TAB 2: GPT / Balochi Q&A
# -----------------------------
with tab_gpt:
    st.subheader("Decoder-only path — Balochi only")
    st.write(
        "Task: **Balochi question → correct answer** (low-resource, monolingual). "
        "The app matches your question to the Balochi dataset and returns the answer based on that question."
    )

    df = get_balochi_df()
    st.write(f"Balochi Q&A pairs loaded: **{len(df)}**")

    test_qs = get_test_questions()
    chosen = st.selectbox(
        "Try a ready Balochi test question:",
        options=["(type your own)"] + test_qs,
        key="balochi_select",
    )

    # Keep text box in sync when user picks a ready question
    if chosen != "(type your own)":
        st.session_state["gpt_input"] = chosen
    elif "gpt_input" not in st.session_state:
        st.session_state["gpt_input"] = "سلام علیکم، تئی نام چے انت؟"

    gpt_prompt = st.text_area(
        "Enter a Balochi question:",
        height=120,
        key="gpt_input",
    )

    show_gpt = st.checkbox("Also run GPT-2 text generation (optional)", value=False)
    gpt_model_id = get_gpt_model_id()
    if show_gpt:
        st.write(f"GPT model: `{gpt_model_id}`")
        max_tokens = st.slider("Max new tokens", 20, 150, 60)

    if st.button("Get Balochi Answer", type="primary", key="gpt_btn"):
        question = gpt_prompt.strip()
        # If a ready question is selected, always use that exact text
        if chosen != "(type your own)":
            question = chosen.strip()

        if not question:
            st.warning("Please type a Balochi question first.")
        else:
            with st.spinner("Matching question to Balochi dataset..."):
                result = answer_balochi_question(question, df)

            st.markdown("### Correct answer (based on matched question)")
            if result["answer"]:
                st.success(result["answer"])
            else:
                st.error("No answer found. Please try another Balochi question.")

            st.markdown("**Your question:**")
            st.write(question)
            st.markdown("**Matched dataset question:**")
            st.write(result["matched_question"])
            st.caption(f"Match score: {result['score']}")

            with st.expander("Top similar questions (for testing)"):
                for i, item in enumerate(result["top_matches"], start=1):
                    st.markdown(f"**{i}. score={item['score']}**")
                    st.write(f"Q: {item['question']}")
                    st.write(f"A: {item['answer'][:220]}...")

            if show_gpt:
                with st.spinner("Running GPT-2 generation..."):
                    prompt = (
                        f"سوال: {question}\n"
                        f"جواب: {result['answer'][:120]}\n"
                    )
                    gen = load_gpt_generator(gpt_model_id)
                    out = gen(
                        prompt,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                        pad_token_id=50256,
                    )
                st.markdown("### Optional GPT-2 output")
                st.write(out[0]["generated_text"])

st.markdown("---")
st.caption(
    "Assignment: one live Streamlit app · BERT encoder + GPT/Balochi decoder path · "
    "Balochi answers matched to questions"
)
