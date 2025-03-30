# Deepfake Detection Using CNN & RNN

## Project Overview
This project detects **face-swap-based deepfake videos** using a deep learning model built with **CNN (ResNext) and RNN (LSTM)**. The model analyzes video frames to determine whether a video is **real or fake**.

## Features
- **Deepfake Detection:** Analyzes videos frame by frame.
- **Confidence Score:** Provides a probability of whether the video is fake.
- **PDF Report Generation:** Generates a detailed report on the analysis.
- **User-Friendly Web Interface:** Allows easy video uploads and model predictions.

---

## 1️⃣ Installation & Setup
### **Prerequisites:**
- Python 3.8+
- pip (Python package manager)
- Virtual environment (optional but recommended)

### **Step 1: Clone the Repository**
```sh
git clone https://github.com/SakshiAher-Codes/Detection-of-Face-Swap-based-Deep-Fake-Videos.git
cd Detection-of-Face-Swap-based-Deep-Fake-Videos
```

### **Step 2: Install Dependencies**
Ensure you have all required dependencies installed by running:
```sh
pip install -r requirements.txt
```

### **Step 3: Run the Application Locally**
```sh
streamlit run main.py
```
This will open the Streamlit web app in your browser.

---

## 2️⃣ Understanding the Deepfake Detection Process
### **1. Video Processing**
- Extracts frames from the uploaded video.
- Preprocesses frames by resizing and normalizing them.

### **2. Model Prediction**
- Each frame is passed through the trained **ResNext + LSTM** model.
- The model assigns a confidence score indicating if the frame is fake or real.

### **3. Report Generation**
- Calculates an overall confidence score for the video.
- Generates a **detailed PDF report** with frame-wise analysis.

---

## 3️⃣ File Structure
```
📁 Detection-of-Face-Swap-based-Deep-Fake-Videos
│── 📄 main.py  # Streamlit app interface
│── 📄 deepfake_model.py  # Deepfake detection functions
│── 📄 requirements.txt  # Required Python packages
│── 📄 README.txt  # Project documentation
```

---

## 4️⃣ Troubleshooting
### **Error: ModuleNotFoundError: No module named 'gdown'**
Fix: Install `gdown` manually:
```sh
pip install gdown
```

### **Error: NameError: name 'st' is not defined**
Fix: Ensure `import streamlit as st` is present at the top of `main.py`.

### **Error: Yowza, that’s a big file (>25MB)!**
Fix: Store the model on **Google Drive** and use:
```python
import gdown
gdown.download("https://drive.google.com/uc?id=1UiYjPQBC-mZO4qETov4C6oFlnSNeSf2i", "deepfake_model.h5", quiet=False)
```

---

## 📩 Contact
For any queries, feel free to reach out at **biz.sakshiaher@gmail.com**.

Happy Coding! 🚀




