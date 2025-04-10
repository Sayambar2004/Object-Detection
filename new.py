import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import os

# Load YOLO model
model = YOLO("saved_model.pt")

# Configure Streamlit layout
st.set_page_config(
    page_title="📦 Inventory Object Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: #f1f1f1;
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #00ffd5;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #cccccc;
        margin-bottom: 2rem;
    }
    .instructions {
        background-color: #1e1e2f;
        border-left: 6px solid #00c6ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,198,255,0.1);
    }
    .stFileUploader label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f1f1;
    }
    .stButton>button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #0072ff, #00c6ff);
    }
    .image-box {
        border: 2px solid #00c6ff;
        border-radius: 10px;
        box-shadow: 0 0 15px #00c6ff33;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .object-count-box {
        background-color: #1c1e26;
        border-left: 6px solid #00c6ff;
        padding: 1rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,198,255,0.15);
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title and Subtitle
st.markdown('<div class="main-title">📦 Inventory Object Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect and count items in your inventory image using AI</div>', unsafe_allow_html=True)

# Instructions Box
st.markdown("""
    <div class="instructions">
    <h4>🔍 How to Use:</h4>
    <ul>
        <li>📁 Upload an image of your inventory or storage</li>
        <li>🤖 Let our AI model detect and annotate objects</li>
        <li>📊 Instantly view the count of detected items</li>
    </ul>
    <p>Supports JPG, JPEG, PNG. Max size: 200MB</p>
    </div>
""", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📤 Uploaded Image", use_container_width=True, output_format="JPEG", clamp=True)

    image = image.convert("RGB")
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    st.markdown("### 🧠 AI is analyzing your image...")

    # Predict
    results = model.predict(source=temp_path, save=True, conf=0.3)
    result_path = Path(results[0].save_dir) / Path(results[0].path).name

    # Display result image
    if result_path.exists():
        st.image(str(result_path), caption="✅ Detected Objects", use_container_width=True, output_format="JPEG", clamp=True)
    else:
        st.error("⚠️ Detection failed: Annotated image not found.")

    # Show object count
    total_objects = len(results[0].boxes.cls)
    st.markdown('<div class="object-count-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Count:")
    st.markdown(f"<span style='font-size: 1.6rem;'>📦 <strong>{total_objects}</strong> object(s) detected</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    os.remove(temp_path)

# Footer
st.markdown('<div class="footer">Crafted with 💙 using AI and Streamlit</div>', unsafe_allow_html=True)
