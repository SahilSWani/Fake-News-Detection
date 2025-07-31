# app.py
import streamlit as st
import joblib
import re
import string
import nltk

nltk.download('stopwords')
stopwords_set = set(nltk.corpus.stopwords.words('english'))

# Load trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords_set]
    return " ".join(words)

# Streamlit app UI
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter news content below to check if it's **Fake or Real**.")

user_input = st.text_area("🗞️ Enter News Article Text:", height=250)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("🚨 This news seems **FAKE**.")
