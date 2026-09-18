import streamlit as st
import re
import numpy as np
import pickle
from gensim.models import FastText

# Load models
ft_model = FastText.load("fasttext_model.model", mmap="r")
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Sentence vector
def sentence_vector(tokens, model):
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Prediction function
def predict(text):
    text = clean_text(text)
    tokens = text.split()
    vec = sentence_vector(tokens, ft_model).reshape(1, -1)
    return model.predict(vec)[0]

# Streamlit UI
st.title("🧠 Mental Health Text Classifier")
user_input = st.text_area("Enter a statement:", "")

if st.button("Predict"):
    if user_input.strip():
        prediction = predict(user_input)
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")
