# Automated-Vehicle-Damage-Detection-Using-Deep-Learning
Automated Vehicle Damage Detection Using Deep Learning is a system that detects and classifies vehicle damages like dents, scratches, and broken glass from images using deep learning models. It enables faster, accurate, and cost-effective insurance claim processing and assessment.

# Automated Vehicle Damage Detection Using Deep Learning

This project is a web-based application that detects visible vehicle damages (such as dents, scratches, and broken glass) from uploaded images using a YOLOv8 deep learning model. It is built with Streamlit for a simple and interactive user interface and uses the Ultralytics YOLOv8 model for real-time object detection.

## Features

- Upload any vehicle image (.jpg, .jpeg, .png)
- Automatically detect and highlight damages
- Uses a pre-trained YOLOv8 model for accuracy and speed
- Saves both uploaded and result images
- Simple, lightweight, and easy to use interface with Streamlit

## How It Works

1. User uploads a vehicle image via the Streamlit interface.
2. The image is saved locally and passed to a pre-trained YOLOv8 model.
3. The model runs inference and detects types of damage (e.g., dent, scratch, broken glass).
4. The result image with bounding boxes is displayed and saved.

## Tech Stack

- **Frontend/UI**: Streamlit
- **Deep Learning Model**: YOLOv8 (`ultralytics` Python library)
- **Programming Language**: Python
- **Image Processing**: PIL
- **File Handling**: `os`, `uuid`
1. Create Virtual Environment and Activate
 python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
2. Install Requirements
pip install -r requirements.txt
3. Place Your Trained Model
4.Replace the placeholder model path with your YOLOv8 model:
5.Running the App
6.streamlit run app.py
Sample Output
Input:
Vehicle image uploaded by the user

Output:
Image with detected damages (e.g., bounding boxes around dents, scratches, etc.)

