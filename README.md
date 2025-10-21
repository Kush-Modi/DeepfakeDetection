# 🔍 Deepfake Detection System

![Deepfake Detection Banner](path/to/banner-image.png)

An AI-powered Streamlit application for **deepfake detection** in videos and live webcam streams, leveraging a pre-trained *Vision Transformer (ViT)* model. This system analyzes faces frame-by-frame for authenticity, providing real-time feedback and comprehensive reports to ensure interview security and media trustworthiness.

---

## Table of Contents

- [About The Project](#about-the-project)  
- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
  - [Video Mode](#video-mode)  
  - [Live Webcam Mode](#live-webcam-mode)  
- [How It Works](#how-it-works)  
- [Configuration](#configuration)  
- [Performance & Limitations](#performance--limitations)  
- [Project Structure](#project-structure)  
- [Deployment](#deployment)  
- [Troubleshooting](#troubleshooting)  
- [Security Considerations](#security-considerations)  
- [Acknowledgments](#acknowledgments)  
- [License](#license)  

---

## About The Project

This system uses a state-of-the-art **Vision Transformer (ViT)** model fine-tuned for binary image classification: detecting whether a face is **real** or a **deepfake**. It analyzes videos uploaded by users or live webcam streams, applying a reliable face detection algorithm (OpenCV Haar Cascade) to isolate faces and then using the transformer model for classification.

![Interface Screenshot](path/to/interface-screenshot.png)

The app features a gorgeous modern UI built with Streamlit, including gradient styles, interactive buttons, real-time video frame overlays, and data visualizations summarizing detection results.

---

## Features

- **Video Analysis**: Upload video files (MP4, AVI, MOV), analyze frames for deepfake presence, with real-time progress and overlay visualizations of predictions.  
- **Live Webcam Detection**: Real-time face detection and deepfake analysis directly from your webcam feed, complete with dynamic confidence scores and session summary metrics.  
- **Majority Voting**: Aggregates frame-level predictions to provide a robust final verdict with average confidence and detection counts.  
- **Intuitive UI/UX**: Custom CSS for clean, modern design with colored alerts, hover effects, metric cards, and easy navigation.  
- **Efficient Model Loading**: Cached model loading for fast startup on repeated sessions.  

---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- Webcam (for live detection mode)  
- Internet connection (for first-time model download)

### Installation

Clone the repo and install dependencies:

git clone https://github.com/yourusername/deepfake-detection-system.git
cd deepfake-detection-system

Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate # macOS/Linux
.venv\Scripts\activate # Windows

Install required packages
pip install -r requirements.txt


If `requirements.txt` is not provided, install core dependencies manually:

pip install streamlit opencv-python torch torchvision transformers pillow numpy



2. Choose **Video Detection** mode.  
3. Upload a video file with visible faces. Supported formats: MP4, AVI, MOV.  
4. Click **Start Analysis** to begin deepfake detection on sampled frames.  
5. View processed video with bounding boxes and labels, detection summary, and charts.

![Video Mode Example](path/to/video-mode-example.gif)

---

### Live Webcam Mode

1. Select **Live Webcam** mode.  
2. Click **Start Live Detection** and position yourself clearly in front of the camera.  
3. See live on-frame verdicts for detected faces with confidence scores.  
4. Stop detection to view summarized session statistics.

![Webcam Mode Example](path/to/webcam-mode-example.gif)

---

## How It Works

- **Face Detection**: Uses OpenCV’s Haar Cascade classifier to detect and crop faces from video frames or webcam feed.  
- **Preprocessing**: Converts face images to RGB and feeds them into the ViTImageProcessor for model compatibility.  
- **Inference**: The Vision Transformer model predicts the probability of the face being "Realism" or "Deepfake."  
- **Aggregation**: Frame predictions are tallied with a majority vote to decide a final verdict and average confidence score.  
- **Display**: Results are shown via color-coded bounding boxes (green for real, red for fake), banners, and detailed metrics.

---

## Configuration

- **Frame Sampling Rate**: Controlled dynamically for videos based on FPS or set fixed for webcam to balance accuracy and speed.  
- **Upload Size Limit**: Default Streamlit limits can be increased via `.streamlit/config.toml` if required.  
- **Session State**: Custom state variables manage mode and detection status across reruns.  

---

## Performance & Limitations

- Model reports ≈ 92% accuracy on evaluation datasets.  
- Performance may drop with poor lighting, occlusions, atypical angles, or heavy video compression.  
- Real-time webcam detection depends on hardware and camera quality.  
- Larger videos require more resources and may increase processing time.  
- Use in combination with other liveness checks for safety-critical applications.

---

## Project Structure

deepfake-detection-system/
│
├── app.py # Main Streamlit app source code
├── requirements.txt # Python dependencies
├── README.md # This documentation
└── assets/ # Placeholder folder for images/screenshot



---

## Deployment

- Easily deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) or platforms supporting Python web apps.  
- Ensure camera permission for live webcam mode during deployment.  
- Manage upload file size and dependencies in deployment configs.

---

## Troubleshooting

- **Cannot access webcam**: Confirm browser and OS camera permissions.  
- **Upload errors for large videos**: Increase Streamlit upload limits.  
- **No faces detected**: Ensure video or webcam captures clear frontal face views.  
- **Slow startup**: Initial model download can take time; cache reduces delays afterward.  

---

## Security Considerations

- Always treat detection results as **probabilistic indicators**. Combine with manual review or multi-factor authentication for high-stakes uses.  
- Do not rely solely on single-frame or single-model detection for security decisions.  

---

## Acknowledgments

- [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) for the Vision Transformer deepfake classifier.  
- Streamlit for rapid UI and app runtime.  
- OpenCV for reliable face detection primitives.  
- Hugging Face Transformers team for model APIs.

---

## License

Refer to the model and dependencies' individual licenses for usage terms.

---

### Screenshots & Demo
<img width="1523" height="913" alt="Screenshot 2025-10-21 172924" src="https://github.com/user-attachments/assets/9fa4d575-e336-4c90-88ea-f919a5c7a285" />

<img width="789" height="632" alt="Screenshot 2025-10-21 173007" src="https://github.com/user-attachments/assets/bb2f3299-c671-4dc7-8123-23df5cbafb43" />


---

> Crafted with ❤️ by [Kush Modi](https://github.com/Kush-Modi)

---

*Feel free to contribute or raise issues to improve this project.*



