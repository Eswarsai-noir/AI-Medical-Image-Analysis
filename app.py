import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("models/model.h5")

st.title("🩺 AI Medical Image Analysis")
st.write("Upload a Chest X-ray image to detect Pneumonia")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ FIRST create img
    img = np.array(image)

    # ✅ THEN apply fixes
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    if prediction > 0.5:
        result = "🛑 PNEUMONIA DETECTED"
        color = "red"
    else:
        result = "✅ NORMAL"
        color = "green"

    st.markdown(f"<h2 style='color:{color};'>{result}</h2>", unsafe_allow_html=True)