import streamlit as st
import joblib
import numpy as np
import re
import nltk
import json
from typing import List, Pattern, Set
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from scipy.sparse import hstack
from streamlit_extras.colored_header import colored_header

# ====== KONFIGURASI PATH ======
VECT_PATH   = "model/vectorizer-fiks-1.pkl"
XGB_PATH    = "model/xgboost_model-fiks-1.pkl"
KEYW_PATH   = "model/aduan_keywords-1.npy"
REGEX_PATH  = "model/regex_booster.json"
SLANG_PATH  = "source/slang-kamus.txt"
THRESHOLD   = 0.4

nltk.download('stopwords')
nltk.download('punkt')

@st.cache_data
def load_slang_dict():
    d = {}
    try:
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    d[k] = v
    except Exception as e:
        st.error(f"Gagal memuat slang dictionary: {e}")
    return d

@st.cache_data
def load_aduan_keywords(npy_path: str) -> Set[str]:
    try:
        return set(np.load(npy_path, allow_pickle=True))
    except Exception as e:
        st.error(f"Gagal memuat aduan keywords: {e}")
        return set()

@st.cache_data
def load_regex_boosters(json_path: str, max_gap: int = 6) -> List[Pattern]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            booster_patterns = json.load(f)

        compiled_patterns = []
        for p in booster_patterns:
            head = re.escape(p["head"])
            tail = p["tail"]
            pattern = rf"{head}(?:\\s+\\w+){{0,{max_gap}}}\\s+{tail}"
            compiled_patterns.append(re.compile(pattern, flags=re.IGNORECASE))
        return compiled_patterns
    except Exception as e:
        st.error(f"Gagal memuat regex boosters: {e}")
        return []

@st.cache_resource
def load_model():
    try:
        vect = joblib.load(VECT_PATH)
        model = joblib.load(XGB_PATH)
        return vect, model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# Normalisasi teks
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing teks

def preprocess_text(text, slang_dict, stemmer, stop_words):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [slang_dict.get(t, t) for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Deteksi aduan berbasis frasa dan regex booster

def is_keyword_aduan(text: str, aduan_keywords: Set[str], regex_boosters: List[Pattern]) -> int:
    clean_text = text.lower()
    clean_text = re.sub(r"[^\w\s]", "", clean_text)
    tokens = word_tokenize(clean_text)

    for n in range(1, 5):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            if phrase in aduan_keywords:
                return 1

    for pattern in regex_boosters:
        if pattern.search(clean_text):
            return 1

    return 0

# Fungsi prediksi utama

def predict_aduan(text, vectorizer, xgb_model, aduan_keywords, regex_boosters, slang_dict, stemmer, stop_words, threshold=0.4):
    keyword_flag = is_keyword_aduan(text, aduan_keywords, regex_boosters)
    txt_clean = preprocess_text(text, slang_dict, stemmer, stop_words)
    Xv = vectorizer.transform([txt_clean])
    Xk = np.array([keyword_flag]).reshape(-1, 1)
    X = hstack([Xv, Xk])
    prob = xgb_model.predict_proba(X)[0, 1]
    lbl = int(prob >= threshold)
    return lbl, prob, txt_clean

# Fungsi utama tampilan Streamlit

def main():
    st.set_page_config("Klasifikasi Aduan Transportasi", layout="wide")
    colored_header("🌟 Deteksi Aduan Lalu Lintas Surabaya", description="Ketik teks keluhan atau komentar masyarakat", color_name="violet-70")

    slang_dict = load_slang_dict()
    aduan_keywords = load_aduan_keywords(KEYW_PATH)
    regex_boosters = load_regex_boosters(REGEX_PATH)
    vectorizer, xgb_model = load_model()

    with st.form("form_aduan"):
        teks_input = st.text_area("Masukkan komentar:", height=150)
        submit = st.form_submit_button("Klasifikasikan")

    if submit and teks_input:
        label, prob, teks_bersih = predict_aduan(
            teks_input, vectorizer, xgb_model,
            aduan_keywords, regex_boosters, slang_dict,
            stemmer, stop_words, threshold=THRESHOLD
        )

        st.markdown("---")
        st.subheader("📊 Hasil Klasifikasi")
        st.write(f"**Teks Asli:** {teks_input}")
        st.write(f"**Teks Bersih:** {teks_bersih}")
        st.write(f"**Probabilitas Aduan:** {prob:.2f}")
        st.success("✅ Termasuk Aduan") if label else st.info("ℹ️ Bukan Aduan")

if __name__ == "__main__":
    main()
