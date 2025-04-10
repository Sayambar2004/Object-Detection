import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import os

# Load the trained YOLOv8 model (make sure saved_model.pt is in the same folder)
model = YOLO("saved_model.pt")

st.set_page_config(page_title="📦 Inventory Object Detector", layout="centered")
st.title("📦 Inventory Object Detector")
st.write("Upload an image of your inventory and detect objects")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to RGB (to avoid errors when saving PNGs as JPEGs)
    image = image.convert("RGB")

    # Save image to disk
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    st.write("🔍 Detecting objects...")

    # Run YOLOv8 prediction
    results = model.predict(source=temp_path, save=True, conf=0.3)

    # Get path to saved annotated image
    result_path = Path(results[0].save_dir) / Path(results[0].path).name

    # Display the annotated image
    if result_path.exists():
        st.image(str(result_path), caption="Detected Objects", use_column_width=True)
    else:
        st.error("⚠️ Annotated image not found.")

    # Clean up the temp input image
    os.remove(temp_path)

