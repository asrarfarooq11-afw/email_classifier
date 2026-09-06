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
        padding: 18px 12px 16px;
        text-align: center;
        backdrop-filter: blur(6px);
    }

    .char-icon {
        width: 42px;
        height: 42px;
        margin: 0 auto 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        animation: float 3s ease-in-out infinite;
    }
    .char-card:nth-child(1) .char-icon { background: rgba(61,220,151,0.15); }
    .char-card:nth-child(2) .char-icon { background: rgba(255,77,109,0.15); }
    .char-card:nth-child(3) .char-icon { background: rgba(124,92,255,0.18); }

    .char-card:nth-child(2) .char-icon { animation-delay: .3s; }
    .char-card:nth-child(3) .char-icon { animation-delay: .6s; }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
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

    .verdict-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: rgba(255,255,255,0.85);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 999px;
        padding: 3px 14px;
        margin-bottom: 10px;
    }
    .verdict-headline { font-size: 1.6rem; font-weight: 800; color: white; margin-top: 4px; }
    .verdict-sub { font-size: 0.95rem; color: rgba(255,255,255,0.9); margin-top: 6px; }

    .confidence-wrap { margin-top: 14px; font-size: 0.85rem; color: rgba(255,255,255,0.85); }

    .stTextArea textarea {
        background-color: #2a1a4a !important;
        color: #f4f1ff !important;
        caret-color: #f4f1ff !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea::placeholder {
        color: #a99bd6 !important;
        opacity: 1 !important;
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
            <div class="char-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3ddc97" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2.5" y="5.5" width="19" height="13" rx="2.2"/>
                    <path d="M3 7l9 6 9-6"/>
                    <path d="M9.5 15.2l1.8 1.8 3.2-3.4" stroke="#3ddc97"/>
                </svg>
            </div>
            <div class="char-label">the ham</div>
            <div class="char-quote">"hey, running 5 mins late!"</div>
        </div>
        <div class="char-card">
            <div class="char-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff4d6d" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2.5" y="5.5" width="19" height="13" rx="2.2"/>
                    <path d="M3 7l9 6 9-6"/>
                    <path d="M12 11.5v2.4" stroke="#ff4d6d"/>
                    <circle cx="12" cy="16.3" r="0.15" stroke="#ff4d6d" stroke-width="2.2"/>
                </svg>
            </div>
            <div class="char-label">the spam</div>
            <div class="char-quote">"u WON $$$ click NOW"</div>
        </div>
        <div class="char-card">
            <div class="char-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c5cff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="5" r="2"/>
                    <circle cx="5.5" cy="18" r="2"/>
                    <circle cx="18.5" cy="18" r="2"/>
                    <path d="M12 7v4M12 11l-5.2 5M12 11l5.2 5"/>
                </svg>
            </div>
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
                    <div class="verdict-tag">VERDICT</div>
                    <div class="verdict-headline">unfortunately... this is SPAM</div>
                    <div class="verdict-sub">giving scam energy. don't click, don't reply.</div>
                    {f'<div class="confidence-wrap">{confidence} confident</div>' if confidence else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            confidence = f"{proba[0]*100:.1f}%" if proba is not None else None
            st.markdown(
                f"""
                <div class="verdict-card verdict-ham">
                    <div class="verdict-tag">VERDICT</div>
                    <div class="verdict-headline">congrats, this is HAM</div>
                    <div class="verdict-sub">certified real one — safe to trust.</div>
                    {f'<div class="confidence-wrap">{confidence} confident</div>' if confidence else ''}
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
