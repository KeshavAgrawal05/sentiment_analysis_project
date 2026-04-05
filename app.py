"""
app.py
------
Streamlit web UI for Twitter Sentiment Analysis.

Run:
    streamlit run app.py
"""

import os
import sys
import json
import time

import streamlit as st

# ── Allow imports from project root ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analyser",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; }
.main-header p  { color: #6b7280; font-size: 1rem; }

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    text-align: center;
    border: 1.5px solid transparent;
}
.positive { background: #f0fdf4; border-color: #86efac; }
.negative { background: #fff1f2; border-color: #fca5a5; }
.neutral  { background: #eff6ff; border-color: #93c5fd; }

.result-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.positive .result-label { color: #16a34a; }
.negative .result-label { color: #dc2626; }
.neutral  .result-label { color: #2563eb; }

.result-confidence { font-size: 0.95rem; color: #6b7280; margin-top: 0.3rem; }

/* ── Probability bar ── */
.prob-bar-container { margin: 0.3rem 0; }
.prob-label { font-size: 0.85rem; font-weight: 500; color: #374151; }
.prob-bar-bg {
    background: #e5e7eb;
    border-radius: 99px;
    height: 10px;
    margin-top: 3px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.4s ease;
}
.fill-positive { background: #4ade80; }
.fill-negative { background: #f87171; }
.fill-neutral  { background: #60a5fa; }

/* ── Tweet pill ── */
.tweet-pill {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.92rem;
    color: #374151;
    margin-bottom: 1rem;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── History item ── */
.history-item {
    background: #f9fafb;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    border-left: 4px solid;
}
.hi-positive { border-color: #4ade80; }
.hi-negative { border-color: #f87171; }
.hi-neutral  { border-color: #60a5fa; }
.hi-text { font-size: 0.88rem; color: #374151; line-height: 1.4; }
.hi-badge {
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 99px;
    padding: 2px 9px;
    white-space: nowrap;
    flex-shrink: 0;
}
.badge-positive { background: #dcfce7; color: #15803d; }
.badge-negative { background: #fee2e2; color: #b91c1c; }
.badge-neutral  { background: #dbeafe; color: #1d4ed8; }

/* ── Example tweets ── */
.example-section { margin-top: 0.4rem; }
.stButton button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the SentimentPredictor once and cache it for the session."""
    from src.predict import SentimentPredictor
    return SentimentPredictor(model_path=MODEL_PATH)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {text, result} dicts


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐦 Tweet Sentiment Analyser</h1>
    <p>Classify any tweet as <b>Positive</b>, <b>Negative</b>, or <b>Neutral</b> using NLP + ML</p>
</div>
""", unsafe_allow_html=True)
st.divider()


# ── Load model ────────────────────────────────────────────────────────────────
model_ready = os.path.exists(MODEL_PATH)

if not model_ready:
    st.error(
        "⚠️  Model not found. Please train the model first:\n\n"
        "```bash\npython src/train.py\n```",
        icon="🚨"
    )
    st.stop()

try:
    predictor = load_predictor()
    st.success("✅  Model loaded successfully!", icon="🤖")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ── Input area ───────────────────────────────────────────────────────────────
st.subheader("📝 Enter a Tweet")

# Example tweets quick-fill
EXAMPLES = {
    "😊 Positive": "I absolutely love this new feature! Works perfectly every time 🎉",
    "😠 Negative": "Worst customer service ever. Waited 2 hours and got no help at all.",
    "😐 Neutral":  "The package was delivered this afternoon.",
}

st.markdown('<div class="example-section"><b>Quick examples:</b></div>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for col, (label, example_text) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["prefill"] = example_text

# Text area (use prefill from example buttons if set)
default_text = st.session_state.pop("prefill", "")
tweet_text = st.text_area(
    label="tweet_input",
    label_visibility="collapsed",
    value=default_text,
    placeholder="Type or paste a tweet here …",
    height=110,
    max_chars=280,
)

char_count = len(tweet_text)
st.caption(f"{char_count} / 280 characters")

analyze_btn = st.button("🔍  Analyse Sentiment", type="primary", use_container_width=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not tweet_text.strip():
        st.warning("Please enter some tweet text first.", icon="⚠️")
    else:
        with st.spinner("Analysing …"):
            time.sleep(0.3)   # brief pause for UX
            result = predictor.predict(tweet_text)

        sentiment = result["sentiment"]
        confidence = result["confidence"]
        conf_level = result["confidence_level"]
        probs      = result["probabilities"]

        # Emoji map
        emoji_map = {"positive": "😊", "negative": "😠", "neutral": "😐"}
        emoji = emoji_map.get(sentiment, "🤔")

        # Result card
        st.markdown(f"""
        <div class="result-card {sentiment}">
            <div style="font-size:2.8rem">{emoji}</div>
            <div class="result-label">{sentiment}</div>
            <div class="result-confidence">
                {conf_level} confidence &nbsp;·&nbsp; {confidence*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("**Class probabilities**")
        fill_class = {"positive": "fill-positive", "negative": "fill-negative", "neutral": "fill-neutral"}
        for cls in ["positive", "negative", "neutral"]:
            p = probs.get(cls, 0.0)
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-label">{cls.capitalize()} &nbsp; <b>{p*100:.1f}%</b></div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill {fill_class[cls]}" style="width:{p*100:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Preprocessed text expander
        with st.expander("🔬 Show preprocessed text"):
            st.code(result["cleaned_text"] or "(empty after cleaning)", language=None)

        # Add to history
        st.session_state.history.insert(0, {"text": tweet_text, "result": result})
        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[:10]


# ── Analysis history ─────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    col1, col2 = st.columns([5, 1])
    col1.subheader("🕑 Recent Analyses")
    if col2.button("Clear", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    for entry in st.session_state.history:
        s   = entry["result"]["sentiment"]
        txt = entry["text"][:120] + ("…" if len(entry["text"]) > 120 else "")
        st.markdown(f"""
        <div class="history-item hi-{s}">
            <span class="hi-badge badge-{s}">{s.upper()}</span>
            <span class="hi-text">{txt}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar: about ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Twitter Sentiment Analysis**  
An NLP-powered classifier built with:
- **Preprocessing**: NLTK (tokenise, stopwords, lemmatise)
- **Features**: TF-IDF (unigrams + bigrams)
- **Models**: Logistic Regression & Naive Bayes
- **UI**: Streamlit
    """)
    st.divider()
    st.markdown("**Model info**")
    if model_ready:
        try:
            from src.utils import load_artifact
            bundle = load_artifact(MODEL_PATH)
            st.write(f"Algorithm : `{bundle.get('best_model_name','?')}`")
            st.write(f"Accuracy  : `{bundle.get('test_accuracy', 0)*100:.2f}%`")
            st.write(f"Classes   : `{bundle.get('label_names', [])}`")
        except Exception:
            st.write("Run training to see model info.")
    st.divider()
    st.caption("Built as a portfolio ML project · GitHub ready")
