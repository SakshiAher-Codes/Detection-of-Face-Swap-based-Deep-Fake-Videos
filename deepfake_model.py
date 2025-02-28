import tensorflow as tf
import numpy as np
import cv2
import os
# Load trained model

# Adjust the model path
model_path = os.path.join(os.path.dirname(__file__), "model", "deepfake_model.h5")

# Debugging: Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file NOT found at {model_path}")

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise OSError(f"🚨 Error loading model: {e}")

# Function to extract frames and predict deepfake probability
def detect_deep_fake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video file."}

    frame_scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (224, 224))  # Resize to match model input
        frame = frame.astype("float32") / 255.0  # Normalize
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        prediction = model.predict(frame)[0][0]  # Model output
        frame_scores.append(prediction)

    cap.release()

    # Compute overall deepfake confidence score
    confidence_score = np.mean(frame_scores) * 100  # Convert to percentage
    detection = "Fake" if confidence_score > 50 else "Real"

    return {
        "detection": detection,
        "confidence_score": round(confidence_score, 2),
        "frame_scores": frame_scores
    }
