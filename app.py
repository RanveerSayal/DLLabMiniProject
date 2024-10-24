
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2
import tempfile
import os

# Force UTF-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load your trained model
model = models.load_model('best_yoga_model.keras')

# Define the function to preprocess the input video
def preprocess_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)  # Sample evenly from the video if it has more frames than needed
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)  # Set frame position
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))  # Resize to the input size expected by the model
        frames.append(frame)
    
    cap.release()
    
    # If fewer than num_frames were read, duplicate the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    frames = np.array(frames)
    frames = frames / 255.0  # Normalize pixel values
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    return frames

st.title("Yoga Pose Detection")
st.write("Upload a video to classify yoga poses.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Use tempfile to handle temporary video file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(uploaded_file.read())
        temp_video_path = temp_video_file.name
    
    # Preprocess the video
    video_data = preprocess_video(temp_video_path)
    
    # Make predictions
    predictions = model.predict(video_data)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Define class names for the yoga poses
    pose_names = ['Padamasana', 'Tadasana', 'Vrikshasana', 'Trikasana', 'Bhujasana']
    st.write(f"Predicted Pose: {pose_names[predicted_class]}")
    
    # Clean up the temporary video file
    os.remove(temp_video_path)

if __name__ == "__main__":
    st.write("Streamlit app is running...")

