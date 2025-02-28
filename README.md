# Detection-of-Face-Swap-based-Deep-Fake-Videos

📌 Repository Description

This project focuses on detecting deepfake videos using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. It provides a user-friendly interface for uploading videos and obtaining deepfake detection results.

📄 Features

🎥 Video Frame Extraction: Extracts frames from videos for analysis.

🧠 Deep Learning Model: Uses CNN for spatial feature extraction and LSTM for temporal dependencies.

📊 Confidence Scoring: Provides a confidence score for deepfake detection.

🔍 Frame-by-Frame Analysis: Detects inconsistencies across video frames.

🖼️ Data Augmentation: Enhances model performance with augmented training data.

📁 Project Structure

DeepFake-Detection/
│── data/                # Dataset (real & fake videos)
│── models/              # Trained model files
│── src/                 # Source code for model training & evaluation
│── app.py               # Streamlit-based web interface
│── deepfake_model.py    # Core detection logic
│── requirements.txt     # Required dependencies
│── README.md            # Project documentation

🚀 Installation

Clone the repository:

git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

🛠 Model Training

To train the model, execute:

python src/train_model.py

This will train the CNN-LSTM model on your dataset.

📈 Deep Fake Detection

After training, test the detection using:

python src/detect.py --video test_video.mp4

This will analyze the video and return the prediction.

📢 Contribution

We welcome contributions! You can fix this repository, create a branch, and submit a pull request.

📜 License

This project is licensed under the MIT License.

🔗 Developed by: Sakshi Aher and Team








