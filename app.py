import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
from PIL import Image

# Title
st.title("📷 Image URL Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with image URLs", type=["xlsx"])

# Model selection
model_choice = st.selectbox("Select Model", ["mobilenet_model_final.h5 (128x128)", "mobilenet_model_v2.h5 (224x224)"])

# Load selected model
if model_choice == "mobilenet_model_final.h5 (128x128)":
    model_path = "mobilenet_model_final.h5"
    img_size = 128
else:
    model_path = "mobilenet_model_v2.h5"
    img_size = 224

@st.cache_resource
def load_selected_model(path):
    model = load_model(path)
    return model

model = load_selected_model(model_path)
st.success("✅ Model loaded successfully")

# If Excel file is uploaded
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    urls = df.iloc[:, 0].tolist()  # Assumes image URLs are in first column

    results = []

    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((img_size, img_size))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)[0]
            label = "Valid" if prediction[0] > 0.5 else "Invalid"
            confidence = float(prediction[0]) if label == "Valid" else 1 - float(prediction[0])
            results.append([url, label, round(confidence * 100, 2)])
        except Exception as e:
            results.append([url, "❌ Error", str(e)])

    result_df = pd.DataFrame(results, columns=["URL", "Prediction", "Confidence/Error"])
    st.dataframe(result_df)

    # Download button
    output_filename = "prediction_results.xlsx"
    result_df.to_excel(output_filename, index=False)
    with open(output_filename, "rb") as f:
        st.download_button("📥 Download Results", f, file_name=output_filename)

