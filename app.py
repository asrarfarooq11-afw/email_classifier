import streamlit as st
import textwrap
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ham or Spam?",
    page_icon="✉",
    layout="centered"
)

# ---------------------------------------------------------
# NLTK
# ---------------------------------------------------------
@st.cache_resource
def setup_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

    return PorterStemmer()

ps = setup_nltk()

# ---------------------------------------------------------
# LOAD TRAINED MODEL + VECTORIZER
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

try:
    model, cv = load_artifacts()
except Exception as e:
    st.error(
        "Model files could not be loaded. Make sure model.pkl and "
        "vectorizer.pkl are in the same folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------
# SAME PREPROCESSING USED DURING TRAINING
# ---------------------------------------------------------
def transform_text(text):
    text = text.lower()

    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Fallback tokenizer if NLTK's punkt resource is unavailable.
        tokens = re.findall(r"\b\w+\b", text)

    tokens = [token for token in tokens if token.isalnum()]

    stop_words = set(stopwords.words("english"))
    tokens = [
        token for token in tokens
        if token not in stop_words and token not in string.punctuation
    ]

    tokens = [ps.stem(token) for token in tokens]

    return " ".join(tokens)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Space+Mono:wght@400;700&display=swap');

    .stApp {
        background: #eee8da;
        color: #171717;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .masthead {
        text-align: center;
        border-top: 4px solid #171717;
        border-bottom: 4px solid #171717;
        padding: 18px 10px 14px 10px;
        margin-bottom: 18px;
    }

    .tiny {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .title {
        font-family: 'Libre Baskerville', serif;
        font-size: clamp(42px, 7vw, 76px);
        font-weight: 700;
        letter-spacing: -3px;
        line-height: 1;
        margin: 8px 0;
    }

    .subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
        margin-top: 10px;
    }

    .comic-strip {
        border: 3px solid #171717;
        background: #f5f0e4;
        padding: 25px 20px 18px 20px;
        margin: 24px 0;
        box-shadow: 7px 7px 0 #171717;
    }

    .scene {
        display: flex;
        align-items: stretch;
        justify-content: center;
        gap: 18px;
    }

    .character {
        flex: 1;
        min-height: 230px;
        border: 2px solid #171717;
        padding: 18px;
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: #eee8da;
    }

    .character-label {
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 700;
    }

    .speech {
        background: #fffdf7;
        border: 2px solid #171717;
        border-radius: 45% 45% 45% 12%;
        padding: 14px;
        font-family: 'Libre Baskerville', serif;
        font-size: 15px;
        line-height: 1.4;
        text-align: center;
        min-height: 78px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .cartoon {
        height: 92px;
        position: relative;
        margin: 8px auto;
        width: 130px;
    }

    .head {
        position: absolute;
        width: 58px;
        height: 58px;
        border: 3px solid #171717;
        border-radius: 50%;
        left: 36px;
        top: 4px;
        background: #f4d7b5;
    }

    .eye-left, .eye-right {
        position: absolute;
        width: 6px;
        height: 8px;
        background: #171717;
        border-radius: 50%;
        top: 25px;
    }

    .eye-left { left: 50px; }
    .eye-right { left: 74px; }

    .body {
        position: absolute;
        width: 70px;
        height: 43px;
        border: 3px solid #171717;
        border-radius: 45% 45% 8px 8px;
        left: 30px;
        top: 61px;
        background: #d8d0bd;
    }

    .arm-left, .arm-right {
        position: absolute;
        width: 45px;
        height: 3px;
        background: #171717;
        top: 76px;
    }

    .arm-left {
        left: 1px;
        transform: rotate(25deg);
    }

    .arm-right {
        right: 1px;
        transform: rotate(-25deg);
    }

    .envelope {
        position: absolute;
        right: 2px;
        top: 38px;
        width: 40px;
        height: 28px;
        border: 2px solid #171717;
        background: #fffdf7;
        transform: rotate(-8deg);
    }

    .envelope:after {
        content: "";
        position: absolute;
        width: 25px;
        height: 2px;
        background: #171717;
        left: 5px;
        top: 11px;
        transform: rotate(28deg);
    }

    .magnifier {
        position: absolute;
        right: 2px;
        top: 36px;
        width: 28px;
        height: 28px;
        border: 4px solid #171717;
        border-radius: 50%;
    }

    .magnifier:after {
        content: "";
        position: absolute;
        width: 28px;
        height: 4px;
        background: #171717;
        right: -21px;
        bottom: -13px;
        transform: rotate(45deg);
    }

    .vs {
        align-self: center;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 18px;
        border: 2px solid #171717;
        background: #fffdf7;
        padding: 10px 8px;
        transform: rotate(-3deg);
    }

    .judge {
        margin-top: 20px;
        border-top: 3px double #171717;
        padding-top: 17px;
        text-align: center;
    }

    .judge-line {
        font-family: 'Libre Baskerville', serif;
        font-weight: 700;
        font-size: 25px;
    }

    .judge-small {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 1px;
        margin-top: 7px;
    }

    .input-label {
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 28px 0 7px 0;
    }

    div[data-testid="stTextArea"] textarea {
        background: #fffdf7;
        color: #171717;
        border: 2px solid #171717;
        border-radius: 0;
        font-family: 'Space Mono', monospace;
        font-size: 14px;
        box-shadow: 4px 4px 0 #171717;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #171717;
        box-shadow: 4px 4px 0 #171717;
    }

    div.stButton > button {
        width: 100%;
        border: 2px solid #171717;
        border-radius: 0;
        background: #171717;
        color: #eee8da;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 13px 10px;
        margin-top: 14px;
        box-shadow: 5px 5px 0 #77705f;
    }

    div.stButton > button:hover {
        background: #2a2a2a;
        color: #fffdf7;
        border-color: #171717;
    }

    .verdict {
        margin-top: 30px;
        border: 3px solid #171717;
        background: #fffdf7;
        padding: 28px 22px;
        text-align: center;
        box-shadow: 7px 7px 0 #171717;
    }

    .verdict-kicker {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .verdict-word {
        font-family: 'Libre Baskerville', serif;
        font-size: clamp(42px, 8vw, 72px);
        font-weight: 700;
        margin: 10px 0;
    }

    .verdict-text {
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        line-height: 1.7;
    }

    .footer-note {
        border-top: 1px solid #171717;
        margin-top: 35px;
        padding-top: 12px;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
    }

    @media (max-width: 700px) {
        .scene {
            flex-direction: column;
        }

        .vs {
            margin: -8px auto;
        }

        .character {
            min-height: 215px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
))

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <div class="masthead">
        <div class="tiny">THE DAILY CLASSIFIER · MACHINE LEARNING EDITION</div>
        <div class="title">HAM OR SPAM?</div>
        <div class="subtitle">ONE MESSAGE. TWO SUSPECTS. ONE FINAL VERDICT.</div>
    </div>
    """,
    unsafe_allow_html=True
))

# ---------------------------------------------------------
# COMIC / STORY
# ---------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <div class="comic-strip">
        <div class="scene">

            <div class="character">
                <div class="character-label">THE GUEST</div>

                <div class="speech">
                    “GUEST HAS ARRIVED!<br>
                    I HAVE A MESSAGE.”
                </div>

                <div class="cartoon">
                    <div class="head">
                        <div class="eye-left"></div>
                        <div class="eye-right"></div>
                    </div>
                    <div class="body"></div>
                    <div class="arm-left"></div>
                    <div class="arm-right"></div>
                    <div class="envelope"></div>
                </div>
            </div>

            <div class="vs">VS</div>

            <div class="character">
                <div class="character-label">THE DETECTIVE</div>

                <div class="speech">
                    “HAM? SPAM?<br>
                    BRING ME THE EVIDENCE.”
                </div>

                <div class="cartoon">
                    <div class="head">
                        <div class="eye-left"></div>
                        <div class="eye-right"></div>
                    </div>
                    <div class="body"></div>
                    <div class="arm-left"></div>
                    <div class="arm-right"></div>
                    <div class="magnifier"></div>
                </div>
            </div>

        </div>

        <div class="judge">
            <div class="judge-line">LET NAIVE BAYES DECIDE.</div>
            <div class="judge-small">
                THE MESSAGE GOES IN · THE PROBABILITY COMES OUT
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
))

# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------
st.markdown('<div class="input-label">THE MESSAGE UNDER INVESTIGATION</div>',
            unsafe_allow_html=True)

message = st.text_area(
    "",
    height=145,
    placeholder="Type or paste an SMS here...",
    label_visibility="collapsed"
)

# ---------------------------------------------------------
# CLASSIFY
# ---------------------------------------------------------
if st.button("SUBMIT FOR CLASSIFICATION"):

    if not message.strip():
        st.warning("Please enter a message before submitting.")
    else:
        processed = transform_text(message)
        vector = cv.transform([processed])

        prediction = model.predict(vector)[0]

        # Notebook mapping: ham = 0, spam = 1
        if prediction == 1:
            verdict = "SPAM"
            explanation = (
                "Naive Bayes has classified this message as spam. "
                "The message's learned word pattern is more strongly "
                "associated with the spam class."
            )
        else:
            verdict = "HAM"
            explanation = (
                "Naive Bayes has classified this message as ham. "
                "The message's learned word pattern is more strongly "
                "associated with the legitimate class."
            )

        st.markdown(
            textwrap.dedent(f"""
            <div class="verdict">
                <div class="verdict-kicker">FINAL VERDICT</div>
                <div class="verdict-word">{verdict}</div>
                <div class="verdict-text">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
            ))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <div class="footer-note">
        COUNT VECTORIZER → MULTINOMIAL NAIVE BAYES → CLASSIFICATION
    </div>
    """,
    unsafe_allow_html=True
))
