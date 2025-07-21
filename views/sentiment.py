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
VECT_PATH   = "model/vectorizer-fiks-1.pkl"
XGB_PATH    = "model/xgboost_model-fiks-1.pkl"
KEYW_PATH   = "model/aduan_keywords-1.npy"
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
def load_keywords():
    try:
        return set(np.load(KEYW_PATH, allow_pickle=True))
    except Exception as e:
        st.error(f"Gagal memuat aduan keywords: {e}")
        return set()

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
def preprocess_text(text, slang_dict, stemmer, stop_words, keywords):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [slang_dict.get(t, t) for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def is_keyword_aduan(text: str, aduan_keywords: set) -> int:
    text_lower = text.lower()
    return int(any(keyword in text_lower for keyword in aduan_keywords))

def predict_aduan(text, vectorizer, xgb_model, aduan_keywords, slang_dict, stemmer, stop_words, threshold=0.4):
    keyword_flag = is_keyword_aduan(text, aduan_keywords)
    txt_clean = preprocess_text(text, slang_dict, stemmer, stop_words, aduan_keywords)
    Xv = vectorizer.transform([txt_clean])
    Xk = np.array([keyword_flag]).reshape(-1, 1)
    X = hstack([Xv, Xk])
    prob = xgb_model.predict_proba(X)[0, 1]
    lbl = int(prob >= threshold)
    return lbl, prob, txt_clean

def main():
    st.set_page_config("🚦 Klasifikasi Aduan Transportasi", layout="wide")
    colored_header("\ud83c\udf1f Deteksi Aduan Lalu Lintas Surabaya", description="Ketik teks keluhan atau komentar masyarakat", color_name="violet-70")

    slang_dict = load_slang_dict()
    aduan_keywords = load_keywords()
    vectorizer, xgb_model = load_model()

    with st.form("form_aduan"):
        teks_input = st.text_area("Masukkan komentar:", height=150)
        submit = st.form_submit_button("Klasifikasikan")

    if submit and teks_input:
        label, prob, teks_bersih = predict_aduan(
            teks_input, vectorizer, xgb_model,
            aduan_keywords, slang_dict, stemmer,
            stop_words, threshold=THRESHOLD
        )

        st.markdown("---")
        st.subheader("\ud83d\udd22 Hasil Klasifikasi")
        st.write(f"**Teks Asli:** {teks_input}")
        st.write(f"**Teks Bersih:** {teks_bersih}")
        st.write(f"**Probabilitas Aduan:** {prob:.2f}")
        st.success("✅ Termasuk Aduan") if label else st.info("ℹ️ Bukan Aduan")

if __name__ == "__main__":
    main()
