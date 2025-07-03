import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from scipy.sparse import hstack
from streamlit_extras.colored_header import colored_header

# ====== KONFIGURASI PATH ======
VECT_PATH = "model/vectorizer-fiks-1.pkl"
XGB_PATH  = "model/xgboost_model-fiks-1.pkl"
KEYW_PATH = "model/aduan_keywords-1.npy"
SLANG_PATH = "source/slang-kamus.txt"
THRESHOLD = 0.4

nltk.download('stopwords')

@st.cache_data
def load_slang_dict():
    d = {}
    try:
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    d[k] = v
    except:
        pass
    return d

def normalize_slang(text: str, slang_dict) -> str:
    return " ".join(slang_dict.get(w, w) for w in word_tokenize(text.lower()))

def preprocess_text(text: str, slang_dict, stemmer, stop_words, aduan_keywords):
    t = normalize_slang(text, slang_dict)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\d+", "", t)
    tokens = [w for w in word_tokenize(t) if w not in stop_words]
    cleaned = []
    for w in tokens:
        stem = stemmer.stem(w)
        cleaned.append(stem)
    return " ".join(cleaned)

def is_keyword_aduan(text: str, aduan_keywords) -> int:
    return int(bool(set(word_tokenize(text.lower())) & aduan_keywords))

def predict_aduan(text, vectorizer, xgb_model, aduan_keywords, slang_dict, stemmer, stop_words, threshold=0.4):
    txt_clean = preprocess_text(text, slang_dict, stemmer, stop_words, aduan_keywords)
    Xv = vectorizer.transform([txt_clean])
    Xk = np.array([is_keyword_aduan(txt_clean, aduan_keywords)]).reshape(-1, 1)
    X = hstack([Xv, Xk])
    prob = xgb_model.predict_proba(X)[0, 1]
    lbl = int(prob >= threshold)
    return lbl, prob, txt_clean

def main():
    vectorizer = joblib.load(VECT_PATH)
    xgb_model = joblib.load(XGB_PATH)
    aduan_keywords = set(np.load(KEYW_PATH, allow_pickle=True))
    slang_dict = load_slang_dict()
    stemmer = StemmerFactory().create_stemmer()
    stop_words = set(stopwords.words("indonesian")) | set(stopwords.words("english"))
    stop_words.update(["iya"])

#======================= UI HALAMAN PREDIKSI =============
    colored_header(
        label="🕵️ Sampaikan Aduan Anda",
        description="Sistem ini akan mendeteksi apakah teks Anda merupakan aduan atau bukan.",
        color_name="violet-70"
    )

    user_input = st.text_area(
        "📝 Masukkan teks aduan:",
        height=150, placeholder="Ketik aduan Anda di sini…"
    )

    if st.button("🔍 Prediksi"):
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
        else:
            pred, prob, txt_cleaned = predict_aduan(
                user_input,
                vectorizer, xgb_model, aduan_keywords,
                slang_dict, stemmer, stop_words,
                THRESHOLD
            )
            st.markdown("---")
            st.subheader("📊 Hasil Prediksi:")
            if pred == 1:
                st.success(f"✅ Aduan Terdeteksi!")
            else:
                st.error(f"❌ Bukan Aduan ")

    if st.button("⬅️ Kembali ke Beranda"):
        st.query_params["page"] = "about"
        st.rerun()

if __name__ == "__main__":
    main()
