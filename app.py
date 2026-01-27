import streamlit as st
import pandas as pd
import joblib
import time
import json
import os
import re
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Depression Detection", page_icon="logo.png", layout="wide")
REPORT_FILE = "reports.json"

# ================== CSS ==================
st.markdown("""
<style>
body { background: radial-gradient(circle at top left, #0f172a 0%, #020617 45%, #000000 100%); color: #E5E7EB; }
body::before { content: ""; position: fixed; inset: 0; background: radial-gradient(circle at 80% 20%, rgba(56,189,248,0.06), transparent 40%), radial-gradient(circle at 20% 80%, rgba(139,92,246,0.05), transparent 40%); z-index: -1; }

.stContainer, .stExpander, .report-card, div[data-testid="stVerticalBlock"] > div {
    background: rgba(17, 25, 40, 0.55) !important; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 8px 32px rgba(0,0,0,0.45); padding: 14px;
}
div[data-testid="stVerticalBlock"] > div:first-child { background: none !important; border: none !important; box-shadow: none !important; backdrop-filter: none !important; padding: 0 !important; }
div[data-testid="stVerticalBlock"] > div:first-child * { background: none !important; }

.stTextArea textarea { background: rgba(2, 6, 23, 0.7); color: #E5E7EB; border-radius: 12px; border: 1px solid rgba(148,163,184,0.25); }
.stTextArea textarea:focus { border-color: #38BDF8; box-shadow: 0 0 8px rgba(56,189,248,0.35); }

.stButton > button { background: linear-gradient(135deg, #020617, #111827); color: #E5E7EB; border: 1px solid rgba(56,189,248,0.35); border-radius: 12px; padding: 0.6em 1.4em; font-weight: 600; letter-spacing: 0.3px; transition: all 0.3s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 0 12px rgba(56,189,248,0.45); border-color: #38BDF8; }
.stButton > button:active { transform: scale(0.97); }

.skeleton-cell { height: 18px; margin: 6px 0; background: linear-gradient(90deg, #020617 25%, #1e293b 37%, #020617 63%); background-size: 400% 100%; animation: shimmer 2.5s infinite; border-radius: 6px; }
@keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }

.stProgress > div > div { background: linear-gradient(90deg, #38BDF8, #8B5CF6); box-shadow: 0 0 8px rgba(56,189,248,0.6); }

.highlight { background: rgba(239,68,68,0.85); border-radius: 4px; padding: 1px 4px; color: #fff; }

summary { font-weight: 600; color: #E5E7EB; }

button[kind="secondary"] { border-color: rgba(239,68,68,0.5) !important; }
button[kind="secondary"]:hover { box-shadow: 0 0 10px rgba(239,68,68,0.6) !important; }

.st-emotion-cache-zy6yx3 { width: 100% !important; padding: 3.5rem 3rem 10rem !important; max-width: initial !important; min-width: auto !important; }

#upload-dataset-optional, #report-history, #analyze-new-text { font-size: 2.02rem !important; font-weight: 600; letter-spacing: 0.2px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
# ----------------- DEPRESSION WORD EXTRACTION -----------------
def extract_depression_words(text):
    """
    Detects depression-related words and phrases in the text.
    Returns a list of unique found words/phrases.
    Multi-word phrases are detected first for priority.
    """
    dep_dict = [
        "depression","depressive","depressed","depressing",
        "sad","sadness","unhappy","unhappiness","down","low","gloomy","gloom",
        "miserable","misery","melancholy","melancholic","hopeless","hopelessness","hopeful",
        "despair","despairing","desperate","emptiness","empty","meaningless","worthless","worthlessness","useless",
        "alone","lonely","loneliness","isolated","isolation","withdrawn","withdrawal","abandoned","rejected","ignored","unwanted","disconnected",
        "numb","numbness","emotionless","detached","detachment","apathetic","apathy","unfeeling","blunted","blunting",
        "tired","fatigue","fatigued","exhausted","exhaustion","drained","burnout","burnedout","lethargic","lethargy",
        "weak","weakness","sleepy","drowsy","confused","confusion","overthinking","ruminating","rumination",
        "foggy","brainfog","forgetful","unfocused","distracted","guilt","guilty","shame","shameful","selfblame","blame",
        "inferior","inadequate","failure","failed","failing","selfhatred","selfdoubt","anhedonia","unmotivated",
        "motivationless","disinterested","bored","boredom","indifferent","indifference","hurt","hurting","pain","painful","suffering",
        "distress","anguish","torment","frustrated","frustration","dysphoria","psychological","psychologicalpain","emotionalpain",
        "mooddisorder","mentalhealth","mentalillness","lowselfesteem","emptymood","flatmood","always","constant","constantly",
        "persistent","persisting","chronic","longterm","endless","neverending"
    ]

    dep_phrases = [
        "i feel depressed","i feel sad","i feel very sad","i feel low","i feel down","i feel empty","i feel numb",
        "i feel broken","i feel lost","i feel hopeless","i feel worthless","i feel useless","i feel miserable","i feel alone",
        "i feel lonely","i have no one","no one cares","no one understands me","i feel isolated","i feel disconnected",
        "i feel abandoned","i feel unwanted","i feel invisible","i feel mentally exhausted","i feel emotionally exhausted",
        "i feel drained","i am tired all the time","i feel burned out","my mind feels tired","my mind feels heavy",
        "i have no energy","i have no motivation","i lost motivation","i lost interest","i lost interest in everything",
        "nothing makes me happy","nothing excites me","i dont enjoy anything","nothing matters anymore",
        "everything feels meaningless","life feels empty","i feel stuck in life","i feel trapped",
        "i feel hopeless about life","i cant focus","i cant think clearly","my thoughts never stop",
        "i overthink everything","my mind feels foggy","i feel like this every day","i feel this all the time",
        "this feeling never goes away","i always feel sad","i have felt this way for a long time"
    ]

    found = set()
    text_lower = text.lower()

    # Check multi-word phrases first (priority)
    for phrase in dep_phrases:
        if phrase in text_lower:
            found.add(phrase)

    # Check single words, ignore if already in multi-word matches
    for word in dep_dict:
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            found.add(word)

    return list(found)


# ----------------- HIGHLIGHT FUNCTION -----------------
def highlight_words(text, words):
    """
    Highlights words/phrases in the original text.
    Handles punctuation, multi-word phrases, and is case-insensitive.
    """
    if not words:
        return text

    # Sort words by length descending to prioritize longer phrases first
    words_sorted = sorted(words, key=lambda x: len(x), reverse=True)

    # Escape regex special chars
    words_escaped = [re.escape(w) for w in words_sorted]

    # Combine into single regex pattern
    pattern = r'(' + '|'.join(words_escaped) + r')'

    # Replace matches with highlight span
    highlighted_text = re.sub(pattern, r'<span class="highlight">\1</span>', text, flags=re.IGNORECASE)

    return highlighted_text



def load_reports():
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_reports(reports):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

if 'reports' not in st.session_state:
    st.session_state['reports'] = load_reports()

def get_base64_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_logo("logo.png")
st.markdown(f"""
   <img src="data:image/png;base64,{logo_base64}" 
     alt="Logo" 
     style="width:154px; height:auto; border-radius:50%; 
     box-shadow:0 0 15px #38BDF8, 0 0 25px #8B5CF6, 0 0 35px #F472B6;
     transition: transform 0.3s ease; display: block; margin: 0 auto;
     margin-top: 24px; margin-bottom: 37px;">
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("""
<h1 style="text-align: center; margin-bottom: 0.2em; padding: 0.25rem 0px 0px 0rem">
AI-Based Depression Detection
</h1>
<p style="text-align: center; margin-top:0; margin-bottom:24px;">
Train model on social data or analyze new thoughts.
</p>
""", unsafe_allow_html=True)

# ================== LAYOUT ==================
left_col, right_col = st.columns([2,3])


# ---------- FULLSCREEN DATASET MODAL ----------
# ================== LEFT PANEL ==================
with left_col:
    st.header("📂 Upload Dataset (Optional)")
    uploaded_file = st.file_uploader(
         "CSV/XLSX with 'post_text' & 'label'", 
         type=["csv","xlsx"], 
         key="upload_dataset"
    )
    
    if uploaded_file is not None:
        placeholder = st.empty()
        with placeholder.container():
            for _ in range(4):
                cols = st.columns(4)
                for c in cols:
                    c.markdown('<div class="skeleton-cell"></div>', unsafe_allow_html=True)
        time.sleep(1.5)

        # Load dataset
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        data = data[["post_text","label"]]
        data.columns = ["text","label"]
        placeholder.empty()
        st.success("Dataset loaded successfully")

        # ---------- SHOW FIRST 5 ROWS FULL WIDTH ----------
        st.dataframe(data.head(), use_container_width=True)

        # ---------- FULL SCREEN MODAL BUTTON ----------
        # ---------- FULL SCREEN MODAL BUTTON ----------
        # ---------- FULL SCREEN DATASET BUTTON ----------
        if st.button("🔍 View Full Dataset"):
            st.markdown("### Full Dataset View")
            st.dataframe(data, use_container_width=True)





        # ---------- TRAIN MODEL BUTTON ----------
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                time.sleep(1)
                X_train, X_test, y_train, y_test = train_test_split(
                    data["text"], data["label"], test_size=0.2, random_state=42
                )
                vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2))
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_vec, y_train)
                acc = accuracy_score(y_test, model.predict(X_test_vec)) * 100
                joblib.dump(model, "model.pkl")
                joblib.dump(vectorizer, "vectorizer.pkl")
                st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}%")


# ================== ANALYZE NEW TEXT ==================
with left_col:
    st.header("📝 Analyze New Text")
    user_text = st.text_area("Write your thoughts here", height=220)
    if st.button("🔍 Analyze"):
        try:
            model = joblib.load("model.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
        except:
            st.error("❌ Train the model first")
            st.stop()
        if user_text.strip():
            cleaned = user_text.lower()
            vec = vectorizer.transform([cleaned])
            base_prob = model.predict_proba(vec)[0][1]*100
            dep_words = extract_depression_words(cleaned)

            # ---------- POSITIVE WORDS ----------
            positive_words = [
                "happy","blessed","excited","fun","love","smile","yay","awesome","amazing","joy",
                "grateful","best day","good vibes","lit","😍","😂","😎","✨","🥰","💖","🥳","💯","😁","😃"
            ]

            # ---------- SCORE CALCULATION ----------
            dep_score = len(dep_words) / max(len(user_text.split()), 1)
            pos_score = len([p for p in positive_words if p in cleaned]) / max(len(user_text.split()), 1)

            # ---------- PROBABILITY ADJUST ----------
            prob = base_prob + dep_score*100 - pos_score*50
            prob = max(min(prob, 100), 0)

            # ---------- MOOD DECISION ----------
            if pos_score > 0.05:
                mood = "Positive"
            elif prob > 60:
                mood = "Depressed"
            else:
                mood = "Neutral"

            # ---------- SAVE REPORT ----------
            user_title = f"User {len(st.session_state.reports)+1}"
            report = {"title": user_title, "text": user_text, "prob": prob, "words": dep_words, "mood": mood}
            st.session_state.reports.insert(0, report)
            save_reports(st.session_state.reports)


# ================== RIGHT PANEL ==================
with right_col:
    st.header("📊 Report History")
    if st.session_state.reports:
        for idx, r in enumerate(st.session_state.reports):
            # Highlight depressive words in text
            highlighted_text = highlight_words(r["text"], r["words"])
            with st.expander(f"{r['title']} - {r['mood']} ({r['prob']:.1f}%)"):
                st.markdown(highlighted_text, unsafe_allow_html=True)
                st.progress(int(r["prob"]))
                if st.button("Delete Report", key=f"del_{idx}"):
                    st.session_state.reports.pop(idx)
                    save_reports(st.session_state.reports)
                    st.rerun()
    else:
        st.info("No reports yet. Analyze text to see results here.")


st.markdown("---")
st.caption("🎓 FYP – AI-Based Depression Detection")
