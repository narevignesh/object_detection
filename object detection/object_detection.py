import streamlit as st
from ultralytics import YOLO
import cv2
import math
import torch
import torch_directml
import numpy as np
import tempfile
import os

# Initialize session state for camera control
if 'camera_on' not in st.session_state:
    st.session_state['camera_on'] = False
if 'cap' not in st.session_state:
    st.session_state['cap'] = None

# Page config
st.set_page_config(page_title="Object Detection", layout="wide")

# Title
st.title("Real-time Object Detection")

def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        st.sidebar.success("Using DirectML device")
        return device
    except Exception as e:
        st.sidebar.error(f"DirectML not available: {e}")
        st.sidebar.warning("Falling back to CPU")
        return "cpu"

device = get_device()

# Model loading
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Sidebar controls
st.sidebar.title("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_labels = st.sidebar.checkbox("Show Labels", True)

# Add detection mode selection
detection_mode = st.sidebar.radio("Detection Mode", 
    ["Real-time Camera", "Image Upload", "Video Upload"])

# Main content
col1, col2 = st.columns([3,1])

with col1:
    if detection_mode == "Real-time Camera":
        # Camera controls
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            start_button = st.button("Start Camera", type="primary")
        with col_btn2:
            stop_button = st.button("Stop Camera", type="secondary")

        if start_button:
            st.session_state.camera_on = True
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    st.session_state.cap = cv2.VideoCapture(1)
                    if not st.session_state.cap.isOpened():
                        st.error("Error: Cannot open camera")
                        st.session_state.camera_on = False
                        st.stop()
                
                st.session_state.cap.set(3, 640)
                st.session_state.cap.set(4, 640)

        if stop_button:
            st.session_state.camera_on = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()

        # Create placeholder for video feed
        stframe = st.empty()

        # Video feed loop
        if st.session_state.camera_on and st.session_state.cap is not None:
            while st.session_state.camera_on:
                success, frame = st.session_state.cap.read()
                if not success:
                    st.error("Failed to grab frame")
                    break

                try:
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run detection
                    results = model(frame_rgb, stream=True, conf=confidence)

                    for r in results:
                        boxes = r.boxes
                        if len(boxes) > 0:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)

                                if show_labels:
                                    conf = math.ceil((box.conf[0] * 100)) / 100
                                    cls = int(box.cls[0])

                                    if 0 <= cls < len(classNames):
                                        label = f'{classNames[cls]} {conf:.2f}'
                                        cv2.putText(frame_rgb, label, 
                                                  (max(0, x1), max(20, y1)), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Display the frame
                    stframe.image(frame_rgb, channels="RGB", use_column_width=True)

                except Exception as e:
                    st.error(f"Error processing frame: {e}")
                    continue

    elif detection_mode == "Image Upload":
        uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            # Convert uploaded image to numpy array
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = model(image_rgb, conf=confidence)
            
            # Process results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    if show_labels:
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        
                        if 0 <= cls < len(classNames):
                            label = f'{classNames[cls]} {conf:.2f}'
                            cv2.putText(image_rgb, label, 
                                      (max(0, x1), max(20, y1)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the processed image
            st.image(image_rgb, channels="RGB", use_column_width=True)

    elif detection_mode == "Video Upload":
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            tfile.close()  # Close the file handle immediately
            
            # Video controls
            col_vid1, col_vid2, col_vid3 = st.columns(3)
            
            with col_vid1:
                start_vid = st.button("Start Processing")
            with col_vid2:
                stop_vid = st.button("Stop")
            with col_vid3:
                restart_vid = st.button("Restart")

            try:
                # Create video capture object
                vid_cap = cv2.VideoCapture(video_path)
                
                # Create placeholder for video frames
                stframe = st.empty()
                
                if start_vid:
                    st.session_state['video_running'] = True
                    
                if stop_vid:
                    st.session_state['video_running'] = False
                    
                if restart_vid:
                    st.session_state['video_running'] = True
                    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Process video frames
                while getattr(st.session_state, 'video_running', False):
                    success, frame = vid_cap.read()
                    if not success:
                        st.session_state['video_running'] = False
                        break
                    
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model(frame_rgb, conf=confidence)
                        
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                
                                if show_labels:
                                    conf = math.ceil((box.conf[0] * 100)) / 100
                                    cls = int(box.cls[0])
                                    
                                    if 0 <= cls < len(classNames):
                                        label = f'{classNames[cls]} {conf:.2f}'
                                        cv2.putText(frame_rgb, label, 
                                                  (max(0, x1), max(20, y1)), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Error processing video frame: {e}")
                        continue
                
            except Exception as e:
                st.error(f"Error processing video: {e}")
            
            finally:
                # Cleanup
                if 'vid_cap' in locals():
                    vid_cap.release()
                try:
                    os.unlink(video_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {e}")

with col2:
    # Update model info
    st.subheader("Model Information")
    st.info("""
    - Model: YOLOv8n
    - Device: DirectML/CPU
    - Resolution: 640x640
    """)
    
    # Update status section based on mode
    st.subheader("Status")
    if detection_mode == "Real-time Camera":
        if st.session_state.camera_on:
            st.success("Camera is running")
        else:
            st.warning("Camera is stopped")
    elif detection_mode == "Image Upload":
        if 'uploaded_image' in locals():
            st.success("Image processed")
        else:
            st.warning("No image uploaded")
    elif detection_mode == "Video Upload":
        if getattr(st.session_state, 'video_running', False):
            st.success("Video is playing")
        else:
            st.warning("Video is stopped")
    
    # Update instructions based on mode
    st.subheader("Instructions")
    if detection_mode == "Real-time Camera":
        st.write("""
        1. Click 'Start Camera' to begin
        2. Adjust confidence threshold as needed
        3. Toggle labels on/off
        4. Click 'Stop Camera' to end
        """)
    elif detection_mode == "Image Upload":
        st.write("""
        1. Upload an image file
        2. Adjust confidence threshold as needed
        3. Toggle labels on/off
        4. Results will display automatically
        """)
    elif detection_mode == "Video Upload":
        st.write("""
        1. Upload a video file
        2. Click 'Start Processing' to begin
        3. Use controls to stop or restart
        4. Adjust settings as needed
        """)

# Cleanup on session end
if not st.session_state.camera_on and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
    if device != "cpu":
        import gc
        gc.collect()

# This is your working PyTorch DirectML version
# Keep this file and continue using it

