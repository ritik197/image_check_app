import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model

# Load models
@st.cache_resource
def load_models():
    models = {
        "MobileNet Final (128x128)": load_model("mobilenet_model_final.h5"),
        "MobileNet V2 (224x224)": load_model("mobilenet_model_v2.h5"),
     
    }
    return models

models = load_models()

# App UI
st.title("📸 Image Prediction from Excel (Valid / Invalid Meter)")
st.markdown("Upload an Excel file with image URLs. Choose a model to classify.")

# Upload Excel
uploaded_file = st.file_uploader("📤 Upload Excel with image URLs (1st column)", type=["xlsx"])

# Select model
model_name = st.selectbox("🤖 Choose a Model", list(models.keys()))
model = models[model_name]
input_shape = model.input_shape[1:3]

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    output = []

    st.write("✅ Processing Images...")
    progress = st.progress(0)
    for i, row in df.iterrows():
        try:
            url = row[0]
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize(input_shape)
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            prediction = model.predict(img_array)[0]
            label = "Valid" if prediction[0] > 0.5 else "Invalid"
            confidence = round(float(prediction[0]) * 100, 2)

            output.append({"URL": url, "Prediction": label, "Confidence (%)": confidence})
        except Exception as e:
            output.append({"URL": url, "Prediction": "❌ Error", "Confidence (%)": 0})
        progress.progress((i + 1) / len(df))

    result_df = pd.DataFrame(output)
    st.success("✅ Done!")
    st.write(result_df)

    # Download output
    def convert_df(df):
        return df.to_excel(index=False, engine='openpyxl')

    st.download_button("📥 Download Result", convert_df(result_df), "prediction_results.xlsx")