import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import uuid

# Paths
MODEL_PATH = "C:/Users/aswin/automated/damage-detector.pt"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Automated Vehicle Damage Detector")
st.title(" Automated Vehicle Damage Detection ")
st.write("Upload an image of a vehicle to detect visible damage using YOLOv8.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    file_ext = os.path.splitext(uploaded_file.name)[-1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    input_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.image(input_path, caption="Uploaded Image", use_container_width=True)
    st.write("Running YOLOv8 damage detection")

    # Run YOLO prediction
    results = model.predict(input_path)

    # Save result image
    result_path = os.path.join(RESULT_FOLDER, unique_filename)
    results[0].save(filename=result_path)

    # Show result image
    st.image(result_path, caption="Detected Damage", use_container_width=True)
