import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model_path = "deepfake_model.h5"  # Update with your actual model path
model = tf.keras.models.load_model(model_path)


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
