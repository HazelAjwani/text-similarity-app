%%writefile app.py
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("Text Similarity Checker")

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input sentences
sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

if st.button("Check Similarity"):
    if sentence1 and sentence2:
        embedding1 = model.encode(sentence1, convert_to_tensor=True)
        embedding2 = model.encode(sentence2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        st.success(f"Similarity Score: {similarity:.4f}")
    else:
        st.warning("Please enter both sentences.")
