import streamlit as st
import pandas as pd
import joblib
import time
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Depression Detection", layout="wide")

REPORT_FILE = "reports.json"

# ================== CSS ==================
st.markdown("""
<style>
/* ============ BASE THEME & TYPOGRAPHY ============ */
:root {
    --primary: #6c63ff;
    --primary-dark: #564fd3;
    --secondary: #ff6584;
    --accent: #36d1dc;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-input: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --border-color: #475569;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
}

* {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    transition: background-color 0.3s ease, border-color 0.3s ease;
}

body {
    background: linear-gradient(135deg, var(--bg-dark) 0%, #1e1b4b 100%);
    color: var(--text-primary);
    min-height: 100vh;
    margin: 0;
    padding: 16px;
}

/* ============ IMPROVED TEXT AREAS ============ */
.stTextArea textarea {
    background-color: var(--bg-input);
    color: var(--text-primary);
    border: 2px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    font-size: 15px;
    resize: vertical;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTextArea textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.2);
    background-color: #2d3748;
}

/* ============ ENHANCED BUTTONS ============ */
.stButton>button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    padding: 12px 24px;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    background: linear-gradient(135deg, var(--primary-dark) 0%, #453fcc 100%);
}

.stButton>button:active {
    transform: translateY(0);
}

.stButton>button::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(255, 255, 255, 0.5);
    opacity: 0;
    border-radius: 100%;
    transform: scale(1, 1) translate(-50%);
    transform-origin: 50% 50%;
}

.stButton>button:focus:not(:active)::after {
    animation: ripple 1s ease-out;
}

@keyframes ripple {
    0% {
        transform: scale(0, 0);
        opacity: 0.5;
    }
    100% {
        transform: scale(20, 20);
        opacity: 0;
    }
}

/* ============ SOPHISTICATED SKELETON LOADER ============ */
.skeleton-cell {
    height: 20px;
    margin: 8px 0;
    background: linear-gradient(
        90deg,
        var(--bg-card) 25%,
        #2d3748 37%,
        var(--bg-card) 63%
    );
    background-size: 400% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
    border-radius: var(--radius-sm);
    position: relative;
    overflow: hidden;
}

.skeleton-cell::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.05),
        transparent
    );
    animation: shine 2s ease-in-out infinite;
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

@keyframes shine {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ============ MODERN REPORT CARD ============ */
.report-card {
    background: var(--bg-card);
    padding: 20px;
    margin-bottom: 16px;
    border-radius: var(--radius-lg);
    border-left: 4px solid var(--primary);
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.report-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    opacity: 0.7;
}

.report-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-left-color: var(--accent);
}

/* ============ HIGHLIGHT EFFECTS ============ */
.highlight {
    background: linear-gradient(120deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
    border-radius: var(--radius-sm);
    padding: 2px 6px;
    color: #ff6b6b;
    font-weight: 600;
    position: relative;
    display: inline-block;
}

.highlight::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 10%;
    width: 80%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff6b6b, transparent);
}

/* ============ ANIMATED PROGRESS BAR ============ */
.progress-container {
    margin: 16px 0;
}

.progress-bar {
    background-color: var(--bg-input);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-top: 8px;
    height: 20px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    position: relative;
}

.progress-fill {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    height: 100%;
    border-radius: var(--radius-lg);
    position: relative;
    overflow: hidden;
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    right: 0;
    background-image: linear-gradient(
        -45deg,
        rgba(255, 255, 255, 0.1) 25%,
        transparent 25%,
        transparent 50%,
        rgba(255, 255, 255, 0.1) 50%,
        rgba(255, 255, 255, 0.1) 75%,
        transparent 75%,
        transparent
    );
    background-size: 20px 20px;
    animation: move-stripes 1s linear infinite;
}

@keyframes move-stripes {
    0% { background-position: 0 0; }
    100% { background-position: 20px 0; }
}

/* ============ EXPANDER WITH SMOOTH ANIMATION ============ */
.expander-container {
    margin: 16px 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.expander-header {
    font-weight: 700;
    font-size: 16px;
    cursor: pointer;
    padding: 16px 20px;
    background: var(--bg-card);
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.3s ease;
    border-left: 4px solid transparent;
}

.expander-header:hover {
    background: #2d3748;
    padding-left: 24px;
    border-left-color: var(--primary);
}

.expander-header::after {
    content: '▼';
    font-size: 12px;
    transition: transform 0.3s ease;
    color: var(--text-muted);
}

.expander-header.expanded::after {
    transform: rotate(180deg);
}

.expander-content {
    max-height: 0;
    overflow: hidden;
    background: #1a2234;
    transition: max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    padding: 0 20px;
}

.expander-content.open {
    max-height: 1000px;
    padding: 20px;
}

/* ============ MODERN CLOSE BUTTON ============ */
.close-btn {
    float: right;
    color: white;
    background: linear-gradient(135deg, var(--error), #dc2626);
    border-radius: 50%;
    width: 28px;
    height: 28px;
    text-align: center;
    line-height: 28px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.close-btn:hover {
    transform: scale(1.1) rotate(90deg);
    box-shadow: var(--shadow-md);
}

.close-btn:active {
    transform: scale(0.95);
}

/* ============ ADDITIONAL MODERN ELEMENTS ============ */
/* Card grid layout */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    margin: 24px 0;
}

.stat-card {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: 20px;
    box-shadow: var(--shadow-sm);
    border-top: 3px solid var(--primary);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}

.stat-value {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 8px 0;
}

/* Tooltip */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted var(--text-muted);
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: var(--bg-input);
    color: var(--text-primary);
    text-align: center;
    border-radius: var(--radius-md);
    padding: 10px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    box-shadow: var(--shadow-lg);
    font-size: 14px;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 0 4px;
}

.badge-primary {
    background: rgba(108, 99, 255, 0.2);
    color: var(--primary);
}

.badge-success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-input);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--primary), var(--accent));
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(var(--primary-dark), var(--accent));
}

/* ============ RESPONSIVE ADJUSTMENTS ============ */
@media (max-width: 768px) {
    .card-grid {
        grid-template-columns: 1fr;
    }
    
    .report-card {
        padding: 16px;
    }
    
    .stat-value {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)
# ----------- ADD THIS FUNCTION HERE -----------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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



