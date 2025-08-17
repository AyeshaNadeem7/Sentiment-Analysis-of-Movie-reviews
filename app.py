# app.py

import streamlit as st
import pickle
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords

nltk.download("stopwords")

# Load Model and Vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928dab); /* purplish gradient */
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #928dab;
        font-size: 16px;
    }
    .stButton button {
        border-radius: 10px;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 0.6em 1.2em;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%);
        color: #fff;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>✨ Discover if your review is  👍 Positive or 👎 Negative✨</h4>", unsafe_allow_html=True)

# Review Input
review_text = st.text_area("✍️ Write your detailed movie review here...", height=150)

if st.button("🔎 Analyze Sentiment"):
    if review_text.strip() != "":
        cleaned = clean_text(review_text)

        # 🚨 Validation: check if input has at least one alphabetic word
        if not re.search(r"[a-zA-Z]", review_text):
            st.error("⚠️ Please enter a valid text review (letters required). Numbers or symbols are not allowed.")
        else:
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

            # Try to get probabilities (Logistic Regression & Naive Bayes support it, SVM doesn’t by default)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = np.max(proba) * 100
            else:
                # fallback: use decision_function (for SVM) and convert to pseudo-confidence
                if hasattr(model, "decision_function"):
                    decision = model.decision_function(vec)
                    confidence = (1 / (1 + np.exp(-abs(decision))))[0] * 100
                else:
                    confidence = 100.0  # if not supported

            # Display result
            if prediction == 1:
                st.success(f"🌟 Positive Review – Audience will love it! \n\n🔮 Confidence: **{confidence:.2f}%**")
            else:
                st.error(f"💔 Negative Review – Critics might not enjoy it. \n\n🔮 Confidence: **{confidence:.2f}%**")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Extra Info
with st.expander("ℹ️ About this app"):
    st.write("""
    This sentiment analyzer is trained on the **IMDb 50K Movie Reviews Dataset**.  
    It uses **Machine Learning (Logistic Regression, Naive Bayes, SVM)**, and the best model was chosen for prediction.  
    - ✨ Positive reviews highlight enjoyment, strong acting, or emotional impact.  
    - 💔 Negative reviews highlight boredom, weak story, or poor acting.  
    Enter your review above and see how the model interprets it! 🎥  
    """)
