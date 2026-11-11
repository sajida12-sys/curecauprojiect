"""
Balochi Q&A helper: match a user question to the closest dataset question
and return the correct answer based on that question.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "balochi_instructions.csv"

# Extra clean test questions (answer must match the question topic)
EXTRA_TEST_PAIRS = [
    {
        "instruction": "سلام علیکم، تئی نام چے انت؟",
        "response": "وعلیکم السلام واجہ، پہ ہیر بیائے! منی نام بلوچی کومکار انت ءُ من شمارا بلوچی نبشتہ ءُ گپ ءُ تران ءَ کومک کناں۔",
    },
    {
        "instruction": "بلوچی روایتی سَجّی ءِ تہا کجام مسالہ جنگ بنت؟",
        "response": "اصلی ءُ کوھنیں بلوچی سجی ءِ تہا بازار ءِ باز مسالہ لوٹ نہ بنت؛ گیشتر ایوک ءَ سنگینیں واد کارمرز بیت ءُ لیمبو یا انار دانگ ءِ نمب ہم تام دارگ ءَ کُمک کنت۔",
    },
    {
        "instruction": "مکران ءِ ماہی گیر لانچانی سر ءَ چنت روچ ءَ پَہ ماہی گیری رونت؟",
        "response": "مکران ءِ ماہی گیر گیشتر دہ روچ چہ پانزدہ روچ تاں زِر ءِ تہا لانچانی سر ءَ مان انت ءُ پدا گوں شکار ءَ ساحل ءَ واتر بنت۔",
    },
    {
        "instruction": "کاریزانی نظام ءِ تہا میرآب چے کار کنت؟",
        "response": "میرآب کاریزانی آپ ءِ بھر ءُ بانگ کنگ ءِ زمہ وار انت؛ آ روچ ءُ پیالگ ءِ حساب ءَ کشارکاراں برابریں آپ دنت تانکہ کسے ءِ حق زوال مہ بیت۔",
    },
    {
        "instruction": "مسواک کنگ ءِ واستہ گیشتر کجام درچکانی شاہ شرتر انت؟",
        "response": "مئے ڈگار ءَ پیلو، کہیر ءُ نیم ءِ مسواک باز نامدار اَنت؛ پیلو دنتان ءُ دپ ءِ تندروستی ءَ کُمک کنت ءُ نیم ءِ مسواک دپ ءِ کرماں رکینیت۔",
    },
    {
        "instruction": "ناکُو ءُ ماما کہول ءِ تہا چے رول دار اَنت؟",
        "response": "ناکُو، ماما ءُ ترو نزیکیں سیاد اَنت کہ دیوان، سور ءُ پُرسانی تہا کہول ءَ یک جا دار اَنت ءُ کسانیں زھگاں مہر ءُ ادب ءِ سوج دئی اَنت۔",
    },
    {
        "instruction": "دشت مکران ءِ مردمانی گیشتریں روزگار چے انت؟",
        "response": "دشت مکران ءِ مردمانی گیشتریں روزگار مچپالی، کِشت ءُ کشار ءُ مالداری انت کہ کمی آپی ءَ ہم وتی زند ءَ اباد دار اَنت۔",
    },
    {
        "instruction": "تو منی چے مدت کُت کنے؟",
        "response": "سلام واجہ! من تئی بلوچی کومکار آں، نبشتہ کاری ءُ لوزانی دربرگ ءَ تئی کُمک ءَ کُت کناں۔ تو پہ دلبڈی ہر جُستے کہ است ات بہ پُرس۔",
    },
]


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^[سS]وال\s*:\s*", "", text)
    text = re.sub(r"\n*جواب\s*:?\s*$", "", text)
    # Remove Arabic/Balochi diacritics so سَجّی ~= سجی
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> set[str]:
    # Keep Balochi/Persian/Arabic letters and numbers as tokens
    parts = re.findall(r"[\w\u0600-\u06FF]+", _normalize(text), flags=re.UNICODE)
    return {p for p in parts if len(p) > 1}


# Common short words that should not dominate matching
_STOP = {
    "ءِ", "ءُ", "ءَ", "چہ", "گوں", "پہ", "تاں", "یا", "ہم", "کہ",
    "انت", "اَنت", "بیت", "کن", "کنت", "چے", "چون", "چتور", "کجام",
}


def _score(question: str, candidate: str) -> float:
    """Score how well candidate question matches user question.

    Prefers: most user content words appear in the candidate (answer-based match).
    """
    q = _tokens(question)
    c = _tokens(candidate)
    if not q or not c:
        qs, cs = set(_normalize(question)), set(_normalize(candidate))
        if not qs or not cs:
            return 0.0
        return len(qs & cs) / len(qs | cs)

    q_content = {t for t in q if t not in _STOP}
    c_content = {t for t in c if t not in _STOP}
    if not q_content:
        q_content = q
    if not c_content:
        c_content = c

    overlap = q_content & c_content
    # Recall: how much of the user question is covered by the dataset question
    recall = len(overlap) / len(q_content)
    # Precision-like: avoid matching unrelated long questions
    precision = len(overlap) / len(c_content) if c_content else 0.0
    jaccard = len(overlap) / len(q_content | c_content)
    # Weighted blend — recall matters most for short user questions
    return 0.55 * recall + 0.25 * precision + 0.20 * jaccard


def load_balochi_qa() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["instruction", "response"]).copy()
    extra = pd.DataFrame(EXTRA_TEST_PAIRS)
    df = pd.concat([extra, df], ignore_index=True)
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    # Drop empty / tiny rows
    df = df[(df["instruction"].str.len() > 2) & (df["response"].str.len() > 2)]
    df = df.drop_duplicates(subset=["instruction"], keep="first").reset_index(drop=True)
    return df


def answer_balochi_question(user_question: str, df: pd.DataFrame | None = None, top_k: int = 3):
    """
    Return the best dataset answer for the user question.
    Also returns top similar questions for transparency/testing.
    """
    if df is None:
        df = load_balochi_qa()

    q = _normalize(user_question)
    if not q:
        return {
            "matched_question": "",
            "answer": "",
            "score": 0.0,
            "top_matches": [],
        }

    scored = []
    for _, row in df.iterrows():
        s = _score(q, row["instruction"])
        scored.append((s, row["instruction"], row["response"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_q, best_a = scored[0]
    top = [
        {"score": round(s, 3), "question": qq, "answer": aa}
        for s, qq, aa in scored[:top_k]
    ]

    # If match is very weak, still return best but mark low confidence
    return {
        "matched_question": best_q,
        "answer": best_a,
        "score": round(best_score, 3),
        "confident": best_score >= 0.35,
        "top_matches": top,
    }


def sample_test_questions(n: int = 12) -> list[str]:
    """Ready-made Balochi questions for testing the app."""
    base = [p["instruction"] for p in EXTRA_TEST_PAIRS]
    df = load_balochi_qa()
    # Prefer longer knowledge questions
    knowledge = df[df["instruction"].str.contains("چے|چون|چتور|کجام", na=False)]
    picked = knowledge.sample(min(n, len(knowledge)), random_state=7)["instruction"].tolist()
    # Keep extras first, then unique knowledge samples
    out = []
    for q in base + picked:
        if q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    # Quick self-test
    df = load_balochi_qa()
    tests = sample_test_questions(8)
    print(f"Dataset size: {len(df)}")
    for q in tests:
        result = answer_balochi_question(q, df)
        print("-" * 60)
        print("Q:", q)
        print("Matched:", result["matched_question"][:80])
        print("Score:", result["score"], "confident:", result["confident"])
        print("A:", result["answer"][:160])
