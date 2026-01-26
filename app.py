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
from utils import clean_text

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Depression Detection", page_icon="logo.png", layout="wide")

REPORT_FILE = "reports.json"

# ================== CSS ==================
st.markdown("""
<style>

/* ---------- GLOBAL BACKGROUND ---------- */
body {
    background: radial-gradient(
        circle at top left,
        #0f172a 0%,
        #020617 45%,
        #000000 100%
    );
    color: #E5E7EB;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 80% 20%, rgba(56,189,248,0.06), transparent 40%),
        radial-gradient(circle at 20% 80%, rgba(139,92,246,0.05), transparent 40%);
    z-index: -1;
}

/* ---------- GLASSMORPHISM PANELS ---------- */
.stContainer,
.stExpander,
.report-card,
div[data-testid="stVerticalBlock"] > div {
    background: rgba(17, 25, 40, 0.55) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    padding: 14px;
}

/* ---------- TEXT AREA ---------- */
.stTextArea textarea {
    background: rgba(2, 6, 23, 0.7);
    color: #E5E7EB;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.25);
}

.stTextArea textarea:focus {
    border-color: #38BDF8;
    box-shadow: 0 0 8px rgba(56,189,248,0.35);
}

/* ---------- FUTURISTIC BUTTONS ---------- */
.stButton > button {
    background: linear-gradient(135deg, #020617, #111827);
    color: #E5E7EB;
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 12px;
    padding: 0.6em 1.4em;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 12px rgba(56,189,248,0.45);
    border-color: #38BDF8;
}

.stButton > button:active {
    transform: scale(0.97);
}

/* ---------- SKELETON LOADER ---------- */
.skeleton-cell {
    height: 18px;
    margin: 6px 0;
    background: linear-gradient(
        90deg,
        #020617 25%,
        #1e293b 37%,
        #020617 63%
    );
    background-size: 400% 100%;
    animation: shimmer 2.5s infinite;
    border-radius: 6px;
}

@keyframes shimmer {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

/* ---------- PROGRESS BAR ---------- */
.stProgress > div > div {
    background: linear-gradient(90deg, #38BDF8, #8B5CF6);
    box-shadow: 0 0 8px rgba(56,189,248,0.6);
}

/* ---------- HIGHLIGHT WORDS ---------- */
.highlight {
    background: rgba(239,68,68,0.85);
    border-radius: 4px;
    padding: 1px 4px;
    color: #fff;
}

/* ---------- EXPANDER HEADER ---------- */
summary {
    font-weight: 600;
    color: #E5E7EB;
}

/* ---------- DELETE / DANGER BUTTON ---------- */
button[kind="secondary"] {
    border-color: rgba(239,68,68,0.5) !important;
}

button[kind="secondary"]:hover {
    box-shadow: 0 0 10px rgba(239,68,68,0.6) !important;
}

</style>
""", unsafe_allow_html=True)



# ================== HELPERS ==================
def highlight_words(text, words):
    for w in words:
        text = re.sub(f"({re.escape(w)})", r'<span class="highlight">\1</span>', text, flags=re.IGNORECASE)
    return text

# ----------- EXPANDED DEPRESSION WORD EXTRACTION -----------
def extract_depression_words(text):
    dep_dict = [
        # Single words
        "depressed", "hopeless", "worthless", "alone", "deadinside", "mentallydone", "fakefine",
        "numb", "empty", "lonelyaf", "selfhate", "broken", "despair",
        "helpless", "unloved", "guilt", "shame", "isolated", "defeated",
        "dejected", "melancholy", "grief", "heartache", "desolation",
        "anxious", "overwhelmed", "fragile", "disconnected", "trapped",
        "paralyzed", "tormented", "regret", "paininside", "selfdoubt",
        "cryingallnight", "darkthoughts", "soulache", "losthope",
        "mentalbreakdown", "hopelessmind", "emptiness", "stressful",
        "lowspirits", "brokenhearted", "discouraged", "overloaded",
        "unmotivated", "inadequate", "failure", "rejected", "ignored",
        "exhausted", "miserable", "resentful", "anguish", "desperate",
        "loneliness", "heartbroken", "sorrow", "unworthy", "fatigue",

        # Multi-word phrases
        "i hate myself", "i am sad", "i feel empty", "i feel numb",
        "i am alone", "cant cope", "feel dead inside", "no energy",
        "i feel worthless", "nothing matters", "i am broken", "feel disconnected",
        "i am helpless", "everything is hopeless", "i have no motivation",
        "i cant sleep", "i am trapped", "my life is pointless",
        "everything is grey", "i feel rejected", "i am a failure",
        "i am failing", "i feel empty inside", "life is painful",
        "nothing feels right", "i am unloved", "i cant go on",
        "i feel suffocated", "i am disappointed with myself",
        "no one understands me", "i am lonely", "i am stuck",
        "i feel broken", "i am worthless", "i am disconnected",
        "i feel hopeless", "i am torn apart", "i cant do this anymore",
        "i feel lost", "i am overwhelmed", "i am exhausted", "my heart hurts",
        "i feel anxious", "i feel helpless", "i have no hope",
        "i feel trapped", "my life is meaningless", "i feel despair",
        "i feel sorrow", "i feel miserable"
    ]

    text_lower = text.lower()
    found = []

    # Exact match for phrases first
    for phrase in dep_dict:
        if " " in phrase:  # multi-word phrase
            if phrase in text_lower:
                found.append(phrase.replace(" ", "_"))
        else:  # single word, whole word match
            if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
                found.append(phrase)

    # Extract hashtags and emojis separately
    hashtags = re.findall(r"#\w+", text)
    emojis = re.findall(r"[^\w\s,]", text)

    return list(set(found + hashtags + emojis))

# ------------------- REPORT HELPERS --------------------
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






# --- Encode logo to base64 for inline HTML ---
def get_base64_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_logo("logo.png")  # folder me logo.png hona chahiye

# --- Display logo at top center with glow effect ---
st.markdown(f"""
<div style="display:flex; justify-content:center; align-items:center; margin-bottom:15px; 
            background: none !important; border: none !important; box-shadow: none !important; padding:0 !important;">
    <img src="data:image/png;base64,{logo_base64}" 
         alt="Logo" 
         style="width:120px; height:auto; border-radius:50%; 
                box-shadow:0 0 15px #38BDF8, 0 0 25px #8B5CF6, 0 0 35px #F472B6;
                transition: transform 0.3s ease;">
</div>
""", unsafe_allow_html=True)




# ================== HEADER ==================
st.title("AI-Based Depression Detection")
st.write("Train model on social data or analyze new thoughts.")

# ================== LAYOUT ==================
left_col, right_col = st.columns([2,3])

# ================== LEFT PANEL ==================
with left_col:
    st.header("📂 Step 1: Upload Dataset (Optional)")
    uploaded_file = st.file_uploader("Upload CSV/XLSX with post_text & label", type=["csv","xlsx"])
    
    if uploaded_file is not None:
        placeholder = st.empty()
        with placeholder.container():
            for _ in range(6):
                cols = st.columns(4)
                for c in cols: c.markdown('<div class="skeleton-cell"></div>', unsafe_allow_html=True)
        time.sleep(2.0)
        
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        data = data[["post_text","label"]]; data.columns=["text","label"]
        placeholder.empty()
        st.success("Dataset loaded successfully")
        st.dataframe(data.head())
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training on real social data..."):
                time.sleep(1.2)
                data["text"] = data["text"].apply(clean_text)
                data["label"] = pd.to_numeric(data["label"], errors="coerce")
                data = data.dropna()
                X = data["text"]; y = data["label"]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2))
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_vec, y_train)
                
                acc = accuracy_score(y_test, model.predict(X_test_vec)) * 100
                
                joblib.dump(model, "model.pkl")
                joblib.dump(vectorizer, "vectorizer.pkl")
                
                st.success(f"✅ Model trained successfully")
                st.write(f"🎯 Test Accuracy: **{acc:.2f}%**")
    
    st.header("📝 Step 2: Analyze New Text")
    user_text = st.text_area("Write real thoughts (slang, emoji allowed)", height=220)
    
    if st.button("🔍 Analyze"):
        try:
            model = joblib.load("model.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
        except:
            st.error("❌ Train the model first")
            st.stop()
        
        if user_text.strip():
            result_placeholder = st.empty()
            with result_placeholder.container():
                for _ in range(4):
                    st.markdown('<div class="skeleton-cell"></div>', unsafe_allow_html=True)
                    time.sleep(0.5)
            
            cleaned = clean_text(user_text)
            vec = vectorizer.transform([cleaned])
            base_prob = model.predict_proba(vec)[0][1]*100

            # ----------- UPDATED WEIGHTED SCORING WITH POSITIVE WORDS -----------
            dep_words = extract_depression_words(cleaned)

            positive_dict = [
                "happy","excited","blessed","smile","yay","awesome","best day","fun","friends",
                "love","joy","sunny","good vibes","lit","laugh","favorite","cheerful","amazing",
                "grateful","delighted","thrilled","fantastic","wonderful","ecstatic","funny","yayyy"
            ]

            strong_weight = 1.0       # depression word weight
            emoji_weight = 0.1        # emojis contribute weakly
            hashtag_weight = 0.2      # hashtags contribute weakly
            positive_weight = 1.0     # positive word weight

            score = 0.0
            positive_score = 0.0

            for w in dep_words:
                if w.startswith("#"):
                    score += hashtag_weight
                elif re.match(r"[^\w\s,]", w):  # emoji
                    score += emoji_weight
                else:
                    score += strong_weight

            for pw in positive_dict:
                if re.search(rf"\b{re.escape(pw)}\b", cleaned.lower()):
                    positive_score += positive_weight

            # Length factor for short sentences
            length_factor = min(len(cleaned.split()) / 10, 1)
            max_score = max(len(dep_words), 1)
            weighted_prob = (score / max_score) * 100 * length_factor

            max_positive = max(len(positive_dict), 1)
            weighted_positive = (positive_score / max_positive) * 100

            prob = (base_prob + weighted_prob) / 2

            # ----------- MULTI-LEVEL MOOD WITH POSITIVE DETECTION -----------
            if weighted_positive > 40:          # adjust threshold if needed
                mood = "Positive"
            elif prob >= 60:
                mood = "Depressed"
            else:
                mood = "Neutral"

            user_title = f"User {len(st.session_state.reports)+1}"
            
            report = {"title": user_title, "text": user_text, "prob": prob, "words": dep_words, "mood": mood}
            st.session_state.reports.insert(0, report)
            save_reports(st.session_state.reports)
            result_placeholder.empty()

# ================== RIGHT PANEL ==================
with right_col:
    st.header("📊 Report History (Click to expand)")
    if st.session_state.reports:
        for idx, r in enumerate(st.session_state.reports):
            with st.expander(f"{r['title']} - {r['mood']} ({r['prob']:.1f}%)", expanded=False):
                st.markdown(highlight_words(r["text"], r["words"]), unsafe_allow_html=True)
                st.progress(int(r["prob"]))
                if st.button("Delete Report", key=f"del_{idx}"):
                    st.session_state.reports.pop(idx)
                    save_reports(st.session_state.reports)
                    st.rerun()
    else:
        st.info("No reports yet. Analyze text to see results here.")

st.markdown("---")
st.caption("🎓 FYP – AI-Based Depression Detection | Click to expand report details")
