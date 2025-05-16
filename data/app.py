import streamlit as st
from utils import predict
import os

st.title("🧅 Onion Image Classifier")
uploaded_file = st.file_uploader("Upload an Onion Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
    label = predict("temp.jpg")
    st.success(f"Predicted class: **{label}**")
