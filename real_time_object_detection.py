import streamlit as st
import cv2
import numpy as np
import random
from ultralytics import YOLO
import tempfile
import os
import warnings
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit App Title and Description
st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🎥", layout="wide")

# Inject custom CSS for styling
st.markdown("""
    <style>
        .css-18e3th9 {
            padding: 20px;
            background-color: #F1F1F1;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1v3fvcr {
            background-color: #1E1E1E;
            color: white;
            border-radius: 15px;
            padding: 20px;
        }
        .stFileUploader>label {
            background-color: #008CBA;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stFileUploader>label:hover {
            background-color: #007B9D;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Title and Description
st.title("YOLOv8 Object Detection App")

# Create two columns
col1, col2 = st.columns([1, 2])  # The first column will have less width, and the second will have more width for the video

with col1:
    st.markdown("""
    ## How it Works:
    Welcome to the YOLOv8 Object Detection App! Follow these steps to use the application:

    1. **Upload a video** (supported formats: MP4, AVI, MOV).
    2. The app will perform object detection using YOLOv8 and display the results **frame-by-frame**.
    3. You can **re-upload another video** to analyze.
    """)

    # Check if class file exists
    class_file_path = r"C:\Users\chand\OneDrive\Desktop\yolo.txt"
    if not os.path.exists(class_file_path):
        st.error("Class file not found at the specified path. Please ensure the file exists and try again.")
        st.stop()

    # Load class names
    with open(class_file_path, 'r') as f:
        class_list = f.read().splitlines()

    # Generate random colors for each class
    colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list
    ]

    # Load YOLO model
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

    # File uploader for video
    uploaded_video = st.file_uploader("Upload a video for analysis", type=["mp4", "avi", "mov"])

# Function to handle video processing
def process_video(video_path):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        st.error("Unable to open the video. Please check the file and try again.")
        os.unlink(video_path)  # Ensure the temporary file is deleted
        return False

    stframe = st.empty()
    st.success("Processing video. Please wait...")

    # Progress bar to show video processing progress
    progress_bar = st.progress(0)

    frame_count = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            st.info("End of video reached or unable to read the frame.")
            break

        # Perform object detection
        results = model.predict(source=[frame], conf=0.45, save=False, verbose=False)
        detections = results[0]

        object_count = {}  # Dictionary to hold the count of each detected class

        if len(detections) > 0:
            for box in detections.boxes:
                # Extract detection information
                clsID = int(box.cls.numpy()[0])
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                # Count the objects
                class_name = class_list[clsID]
                if class_name not in object_count:
                    object_count[class_name] = 0
                object_count[class_name] += 1

                # Draw bounding boxes
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    colors[clsID],
                    3,
                )

                # Display class name and confidence
                cv2.putText(
                    frame,
                    f"{class_name}: {round(conf * 100, 2)}%",
                    (int(bb[0]), int(bb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        # Show detected object count on the frame
        y_offset = 30
        for obj, count in object_count.items():
            cv2.putText(
                frame,
                f"{obj}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            y_offset += 30

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Release video capture resources
    capture.release()
    return True

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Create a column for the video display
    with col2:
        # Process video
        if process_video(video_path):
            st.success("Processing complete. You can upload a new video for analysis.")
        else:
            st.error("An error occurred during video processing.")
        
        # Delete the temporary video file after processing
        try:
            os.unlink(video_path)
        except PermissionError:
            st.warning("Permission error while deleting the temporary video file. Please close any applications using the file and try again.")

else:
    st.info("Please upload a video file to get started.")
