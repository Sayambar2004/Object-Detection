import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import os

# Load model
model = YOLO("saved_model.pt")

# Streamlit config
st.set_page_config(page_title="📦 Inventory Object Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
    .subtext {
        text-align: center;
        font-size: 1.1rem;
        color: #bbbbbb;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .uploaded-img, .result-img {
        border: 2px solid #333;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-title">📦 Inventory Object Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload an inventory image and detect objects using a YOLOv8 model.</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to RGB
    image = image.convert("RGB")
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    st.markdown("### 🔍 Detecting objects...")

    # Predict
    results = model.predict(source=temp_path, save=True, conf=0.3)
    result_path = Path(results[0].save_dir) / Path(results[0].path).name

    # Show result
    if result_path.exists():
        st.image(str(result_path), caption="Detected Objects", use_container_width=True)
    else:
        st.error("⚠️ Detection failed: Annotated image not found.")

    # Count total objects
    total_objects = len(results[0].boxes.cls)
    st.markdown("### 📊 Count:")
    st.write(f"📦 **Count:** {total_objects}")

    # Cleanup
    os.remove(temp_path)
