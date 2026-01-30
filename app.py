import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import uuid

# Set Page Config MUST be the first Streamlit command
st.set_page_config(
    page_title="Automated Vehicle Damage Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, deep learning aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border-color: #ff6b6b;
    }
    h1 {
        color: #fafafa;
        text-align: center;
        font-weight: 700;
    }
    h2, h3 {
        color: #e0e0e0;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Inter', sans-serif;
    }
    .uploaded-img {
        border: 2px solid #333;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Paths
# Dynamically find the model in the current directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "damage-detector.pt")
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Sidebar
with st.sidebar:
    st.title("🚗 Damage AI")
    st.info(
        """
        **Automated Vehicle Damage Detection**
        
        This system uses state-of-the-art Deep Learning (YOLOv8) to identify vehicle damages such as:
        - Dents
        - Scratches
        - Broken Glass
        - Smashes
        
        **How to use:**
        1. Upload an image.
        2. Wait for the analysis.
        3. View the results.
        """
    )
    st.write("---")
    st.caption("Powered by YOLOv8 & Streamlit")

# Main Content
st.title("🛡️ Automated Vehicle Damage Detection")
st.write("### Upload a vehicle image to instantly detect visible damages.")

# Layout: 2 Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose a file (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded file
        file_ext = os.path.splitext(uploaded_file.name)[-1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(input_path, caption="Original Image", use_container_width=True)

        if st.button("Analyze Damage 🔍"):
            with st.spinner("Analyzing image for damages..."):
                # Load YOLO model
                if not os.path.exists(MODEL_PATH):
                    st.error(f"Model file not found at: {MODEL_PATH}. Please ensure 'damage-detector.pt' is in the project directory.")
                else:
                    try:
                        model = YOLO(MODEL_PATH)
                        
                        # Run YOLO prediction
                        results = model.predict(input_path)

                        # Save result image
                        result_filename = f"result_{unique_filename}"
                        result_path = os.path.join(RESULT_FOLDER, result_filename)
                        
                        # Plot and save
                        # YOLOv8 results object has a plot() method that returns a numpy array (BGR)
                        # or we can use save() method. 
                        # To display in streamlit, it's best to get the plotted image array.
                        res_plotted = results[0].plot()
                        res_image = Image.fromarray(res_plotted[..., ::-1]) # RGB
                        res_image.save(result_path)

                        # Show result in col2
                        with col2:
                            st.subheader("2. Detection Results")
                            st.image(result_path, caption="Detected Damages", use_container_width=True)
                            
                            # Show detections count if possible
                            boxes = results[0].boxes
                            if len(boxes) > 0:
                                st.success(f"Detected {len(boxes)} issue(s)!")
                                # Optional: List classes
                                # for box in boxes:
                                #     cls = int(box.cls[0])
                                #     st.write(f"- {model.names[cls]}")
                            else:
                                st.info("No visible damage detected.")
                                
                    except Exception as e:
                        st.error(f"Error during detection: {e}")

    else:
        # Placeholder or empty state
        with col2:
            st.info("Results will appear here after upload.")

