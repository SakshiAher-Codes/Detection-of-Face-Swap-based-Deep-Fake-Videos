import tempfile
import streamlit as st
import cv2
import os
import time
import numpy as np
from deepfake_model import detect_deep_fake  # Import function from deepfake_model.py

st.set_page_config(page_title="Deep Fake Detection", layout="wide")

st.title("🔍 Deep Fake Detection Application")

st.sidebar.title("Navigation")
page = st.sidebar.radio("📌 Select a page:",
                        ("Home", "Upload Video", "Detection Results", "How It Works", "Feedback", "About"))

# Home Page
if page == "Home":
    st.header("🏠 Welcome to the Deep Fake Detection Application!")
    st.write(
        "This application uses advanced deep learning models to analyze videos and detect deepfake content. "
        "It provides an easy-to-use interface for uploading videos and getting predictions."
    )
    st.image("deepfake_example.png", caption="Example of a deepfake video")

    st.subheader("🔹 Features:")
    st.markdown("""
    - 🎥 Upload and analyze videos effortlessly  
    - 📊 View confidence scores and frame-by-frame analysis  
    - 🧠 AI-powered deepfake detection using CNN & LSTMs  
    - 🔍 Real-time feedback  
    """)

# Upload Video Page
elif page == "Upload Video":
    st.header("📤 Upload Video for Detection")
    uploaded_file = st.file_uploader("🎬 Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.success("✅ Video uploaded successfully! Showing Preview...")

        # **🔹 LIVE Video Preview Using OpenCV**
        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (Streamlit requires RGB format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame in Streamlit webpage without deprecation warning
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            time.sleep(0.03)  # Adjust for smooth playback

        cap.release()

        # ✅ **NOW the "Analyze" button appears AFTER video preview**
        if st.button("🚀 Analyze Video"):
            st.success("Processing your video... ⏳")
            result = detect_deep_fake(video_path)

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state["detection_result"] = result
                st.rerun()

# Detection Results Page
elif page == "Detection Results":
    st.header("📊 Detection Results")

    if "detection_result" in st.session_state:
        result = st.session_state["detection_result"]
        st.write("**⚠️ Deep Fake Detection:**", result["detection"])
        st.write("**✅ Confidence Score:**", result["confidence_score"], "%")

        st.subheader("📈 Frame-by-Frame Analysis")
        st.line_chart(result["frame_scores"])  # Confidence per frame

        st.subheader("📢 Interpretation of Results")
        if result["confidence_score"] > 80:
            st.error("⚠️ High confidence that this video is a deepfake!")
        elif 50 < result["confidence_score"] <= 80:
            st.warning("⚠️ Moderate confidence of deepfake content detected.")
        else:
            st.success("✅ Low confidence in deepfake presence. The video appears genuine.")
    else:
        st.warning("⚠️ No detection results available. Please analyze a video first.")

# How It Works Page
elif page == "How It Works":
    st.header("🛠 How Deep Fake Detection Works")
    st.write("""
    Our detection system uses:
    - 🧠 **ResNext + LSTM** for facial feature analysis  
    - 🎞 **Frame-wise Confidence Scoring** to detect motion inconsistencies  
    - 🔬 **Deep Learning Models** trained on real and fake datasets  
    """)
    st.image("model_architecture.png", caption="Model Architecture")

# Feedback Page
elif page == "Feedback":
    st.header("💬 User Feedback")
    st.write("We value your feedback to improve our application.")

    feedback = st.text_area("✍️ Your feedback here:")
    if st.button("📩 Submit Feedback"):
        st.success("🙏 Thank you for your feedback!")

# About Page
elif page == "About":
    st.header("📚 About This Application")
    st.write(
        "This Deep Fake Detection Application helps users identify deepfake videos using state-of-the-art AI models. "
        "It provides an interactive interface for video analysis and insights."
    )
    st.write("For more details, visit our [GitHub Repository](https://github.com/example-repo).")

st.sidebar.write("👨‍💻 Developed by Sakshi Aher and Team")
