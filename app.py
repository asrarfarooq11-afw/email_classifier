import string
import pickle

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam or Nah?",
    page_icon="📬",
    layout="centered",
)

# ----------------------------------------------------------------------------
# ONE-TIME SETUP: nltk data, model, vectorizer
# ----------------------------------------------------------------------------
@st.cache_resource
def load_nltk():
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    # stopwords lives under corpora, not tokenizers — make sure it's there too
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


load_nltk()
model, vectorizer = load_artifacts()
ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))


# ----------------------------------------------------------------------------
# THIS MUST MATCH THE NOTEBOOK EXACTLY — it's the #1 reason deployed spam
# classifiers give garbage predictions: training preprocesses the text,
# deployment doesn't, and the vectorizer sees totally different input.
# ----------------------------------------------------------------------------
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


def predict(message: str):
    cleaned = transform_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]  # [P(ham), P(spam)]
    return pred, proba, cleaned


# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Poppins:wght@400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #1b1035 0%, #2d1b4e 35%, #3a1c5e 65%, #241242 100%);
        color: #f4f1ff;
    }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff6ec7, #7c5cff, #5ce1e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .hero-sub {
        text-align: center;
        color: #cbbdf5;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.6rem;
    }

    .char-row { display: flex; justify-content: space-between; gap: 14px; margin-bottom: 1.6rem; }

    .char-card {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 14px 12px;
        text-align: center;
        backdrop-filter: blur(6px);
    }

    .char-emoji { font-size: 2.4rem; display: block; margin-bottom: 6px; animation: float 2.6s ease-in-out infinite; }
    .char-card:nth-child(2) .char-emoji { animation-delay: .3s; }
    .char-card:nth-child(3) .char-emoji { animation-delay: .6s; }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .char-label { font-weight: 700; font-size: 0.85rem; color: #fff; }
    .char-quote { font-size: 0.78rem; color: #d8cdf7; margin-top: 4px; font-style: italic; }

    .verdict-card {
        border-radius: 20px;
        padding: 28px 22px;
        text-align: center;
        margin-top: 1.2rem;
        animation: pop 0.35s ease-out;
    }

    @keyframes pop {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    .verdict-spam {
        background: linear-gradient(135deg, #ff4d6d, #c9184a);
        box-shadow: 0 8px 30px rgba(255,77,109,0.35);
    }

    .verdict-ham {
        background: linear-gradient(135deg, #3ddc97, #1a936f);
        box-shadow: 0 8px 30px rgba(61,220,151,0.30);
    }

    .verdict-emoji { font-size: 3rem; }
    .verdict-headline { font-size: 1.5rem; font-weight: 800; color: white; margin-top: 4px; }
    .verdict-sub { font-size: 0.95rem; color: rgba(255,255,255,0.9); margin-top: 6px; }

    .confidence-wrap { margin-top: 14px; font-size: 0.85rem; color: rgba(255,255,255,0.85); }

    .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        color: #fff !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        font-size: 1rem !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #ff6ec7, #7c5cff);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        padding: 10px 28px;
        font-size: 1rem;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover { transform: scale(1.04); color: white; }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">📬 SPAM OR NAH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">welcome bestie — drop a text and let Naive Bayes read the room 👀</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="char-row">
        <div class="char-card">
            <span class="char-emoji">😇</span>
            <div class="char-label">the ham</div>
            <div class="char-quote">"hey, running 5 mins late!"</div>
        </div>
        <div class="char-card">
            <span class="char-emoji">😈</span>
            <div class="char-label">the spam</div>
            <div class="char-quote">"u WON $$$ click NOW"</div>
        </div>
        <div class="char-card">
            <span class="char-emoji">🧠</span>
            <div class="char-label">naive bayes</div>
            <div class="char-quote">"bet. let me cook."</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------------
message = st.text_area(
    "your message",
    height=130,
    placeholder="paste the suspicious text/email here...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1])
with col1:
    judge_clicked = st.button("judge it 🔨", use_container_width=True)
with col2:
    clear_clicked = st.button("clear", use_container_width=True)

if clear_clicked:
    st.rerun()

# ----------------------------------------------------------------------------
# RESULT
# ----------------------------------------------------------------------------
if judge_clicked:
    if not message.strip():
        st.warning("bestie... you gotta type something first 💀")
    else:
        pred, proba, cleaned = predict(message)

        if pred == 1:
            confidence = f"{proba[1]*100:.1f}%" if proba is not None else None
            st.markdown(
                f"""
                <div class="verdict-card verdict-spam">
                    <div class="verdict-emoji">💀</div>
                    <div class="verdict-headline">unfortunately... you got SPAM'd</div>
                    <div class="verdict-sub">this one's giving scam energy. do not click, do not reply.</div>
                    {f'<div class="confidence-wrap">confidence: {confidence} spam</div>' if confidence else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            confidence = f"{proba[0]*100:.1f}%" if proba is not None else None
            st.markdown(
                f"""
                <div class="verdict-card verdict-ham">
                    <div class="verdict-emoji">✅</div>
                    <div class="verdict-headline">congrats, this is HAM</div>
                    <div class="verdict-sub">certified real one. safe to trust (probably still read carefully lol).</div>
                    {f'<div class="confidence-wrap">confidence: {confidence} ham</div>' if confidence else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("see what naive bayes actually saw"):
            st.code(cleaned if cleaned else "(nothing left after cleaning)", language=None)

# ----------------------------------------------------------------------------
# FOOTER / ABOUT
# ----------------------------------------------------------------------------
with st.expander("how does this work?"):
    st.markdown(
        """
        This is a **Multinomial Naive Bayes** classifier trained on labeled SMS/email
        messages (`ham` = 0, `spam` = 1).

        Every message goes through the *exact* same cleanup used during training before
        it's judged:
        1. lowercase everything
        2. tokenize into words
        3. drop non-alphanumeric tokens
        4. remove stopwords + punctuation
        5. stem each word (Porter Stemmer)

        The cleaned text is turned into word-count vectors with a fitted
        `CountVectorizer`, and Naive Bayes does the rest.
        """
    )
