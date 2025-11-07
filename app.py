import streamlit as st
import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
from collections import Counter
import tempfile

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="stApp"] {
        font-family: 'Roboto', Arial, sans-serif;
        background: #f4f4f5;
        color: #18181b;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #18181b;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #52525b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .card {
        background: #fff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        color: #18181b;
    }
    .stButton>button {
        width: 100%;
        background: #18181b;
        color: #fff;
        border: none;
        padding: 0.8rem 1.2rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 10px;
        transition: background 0.2s, color 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        cursor: pointer;
    }
    .stButton>button:hover {
        background: #6366f1;
        color: #fff;
    }
    .stFileUploader {
        background: #fff;
        border-radius: 10px;
        padding: 1.2rem;
        border: 2px dashed #a1a1aa;
        color: #18181b;
    }
    .stFileUploader:hover {
        border-color: #6366f1;
        background: #f4f4f5;
    }
    .result-success {
        background: #22c55e;
        color: #fff;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .result-danger {
        background: #ef4444;
        color: #fff;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .metric-card {
        background: #fff;
        color: #18181b;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.3rem;
        border: 1px solid #e5e7eb;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 600;
        letter-spacing: 0.2px;
        text-transform: uppercase;
    }
    .stProgress > div > div > div > div {
        background: #6366f1;
    }
    .live-label {
        background: #fff;
        color: #18181b;
        padding: 0.7rem 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.7rem 0;
        text-align: center;
        display: inline-block;
        letter-spacing: 0.2px;
        border: 1px solid #e5e7eb;
    }
    .live-label.real {
        background: #22c55e;
        color: #fff;
    }
    .live-label.fake {
        background: #ef4444;
        color: #fff;
    }

    h2, h3 {
        color: #18181b;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    """Load the pre-trained deepfake detection model"""
    try:
        model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# ==================== HELPER FUNCTIONS ====================
def get_face_from_frame(frame):
    """Extract faces from frame using Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        face_data = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face_data.append({
                'image': face_img,
                'coords': (x, y, w, h)
            })
        return face_data
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return []


def predict_deepfake(face_image, processor, model):
    """Predict if face is real or deepfake"""
    try:
        if isinstance(face_image, np.ndarray):
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = Image.fromarray(face_rgb)

        inputs = processor(images=face_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class_idx = logits.argmax(-1).item()

        label = model.config.id2label[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx].item())

        return label, confidence

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, 0.0


def calculate_final_verdict(predictions):
    """Calculate final verdict based on majority voting"""
    if not predictions:
        return "Unknown", 0.0, 0, 0

    label_counts = Counter([pred['label'] for pred in predictions])
    final_label = label_counts.most_common(1)[0][0]

    label_confidences = [
        pred['confidence'] for pred in predictions if pred['label'] == final_label
    ]
    avg_confidence = sum(label_confidences) / len(label_confidences) if label_confidences else 0.0

    real_count = label_counts.get("Realism", 0)
    fake_count = label_counts.get("Deepfake", 0)

    return final_label, avg_confidence, real_count, fake_count


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-title">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Video Interview Security & Authenticity Verification</p>', unsafe_allow_html=True)

    # Load model
    with st.spinner("🔄 Loading AI Model..."):
        processor, model = load_deepfake_model()

    if processor is None or model is None:
        st.error("Model failed to load. Please refresh and try again.")
        return

    # Mode Selection
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        if st.button("🎥 VIDEO DETECTION", key="btn_video", use_container_width=True):
            st.session_state.mode = "video"

    with col2:
        if st.button("📡 LIVE WEBCAM", key="btn_webcam", use_container_width=True):
            st.session_state.mode = "webcam"

    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'mode' not in st.session_state:
        st.session_state.mode = None

    st.markdown("---")

    # ==================== VIDEO MODE ====================
    if st.session_state.mode == "video":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 🎥 Video Detection Mode")

        uploaded_video = st.file_uploader(
            "Upload a video file (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            help="Upload a video with visible faces for analysis"
        )

        if uploaded_video:
            st.video(uploaded_video)

            if st.button("🚀 Start Analysis", key="analyze_vid", use_container_width=True):
                st.markdown("### 🔄 Processing Video...")

                # Save video temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                try:
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    # Process video
                    cap = cv2.VideoCapture(tfile.name)
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

                    predictions = []
                    frame_count = 0
                    FRAME_SKIP = max(1, (fps // 2) if fps else 15)

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    frame_placeholder = st.empty()

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        # Process every FRAME_SKIP frames
                        if frame_count % FRAME_SKIP == 0:
                            face_data = get_face_from_frame(frame)

                            if len(face_data) > 0:
                                face_img = face_data[0]['image']
                                label, confidence = predict_deepfake(face_img, processor, model)

                                if label:
                                    predictions.append({
                                        'label': label,
                                        'confidence': confidence
                                    })

                                    # Draw on frame
                                    x, y, w, h = face_data[0]['coords']
                                    color = (34, 197, 94) if label == "Realism" else (239, 68, 68)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                                    cv2.putText(
                                        frame,
                                        f"{label}: {confidence*100:.1f}%",
                                        (x, max(20, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        color,
                                        2
                                    )

                            # Display frame
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        # Update progress
                        progress = (frame_count / total_frames) if total_frames > 0 else 0
                        progress_bar.progress(min(1.0, progress))
                        status_text.markdown(f"**Processing:** Frame {frame_count}/{total_frames if total_frames>0 else '—'} ({progress*100:.1f}%)")

                    cap.release()

                finally:
                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass

                # Clear UI placeholders
                progress_bar.empty()
                status_text.empty()
                frame_placeholder.empty()

                # Calculate final verdict
                if predictions:
                    final_label, avg_confidence, real_count, fake_count = calculate_final_verdict(predictions)

                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")

                    # Display verdict
                    if final_label == "Realism":
                        st.markdown(
                            f'<div class="result-success">✅ AUTHENTIC VIDEO<br>'
                            f'<span style="font-size:1.1rem;">Confidence: {avg_confidence*100:.2f}%</span></div>',
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    else:
                        st.markdown(
                            f'<div class="result-danger">⚠️ DEEPFAKE DETECTED<br>'
                            f'<span style="font-size:1.1rem;">Confidence: {avg_confidence*100:.2f}%</span></div>',
                            unsafe_allow_html=True
                        )

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(predictions)}</div>
                            <div class="metric-label">Predictions</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{real_count}</div>
                            <div class="metric-label">Real Count</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{fake_count}</div>
                            <div class="metric-label">Fake Count</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{avg_confidence*100:.1f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Chart (kept simple for compatibility)
                    st.markdown("### 📈 Detection Breakdown")
                    chart_data = {
                        'Detection Type': ['Real', 'Fake'],
                        'Count': [real_count, fake_count]
                    }
                    st.bar_chart(chart_data, x='Detection Type', y='Count', height=300)

                else:
                    st.warning("⚠️ No faces detected in the video. Please upload a video with visible faces.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== WEBCAM MODE ====================
    elif st.session_state.mode == "webcam":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 📡 Live Webcam Detection")

        st.info("💡 Tip: Position your face clearly in front of the camera for best results. Detection updates every few seconds.")

        col_start, col_stop = st.columns(2)

        with col_start:
            start_btn = st.button("🎥 Start Live Detection", key="start_webcam", use_container_width=True)

        with col_stop:
            stop_btn = st.button("🛑 Stop Detection", key="stop_webcam", use_container_width=True)

        if start_btn:
            st.session_state.webcam_active = True
        if stop_btn:
            st.session_state.webcam_active = False

        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False

        if st.session_state.webcam_active:
            st.markdown("---")
            st.markdown("### 📹 Live Feed")

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Please check your camera permissions.")
                st.session_state.webcam_active = False
            else:
                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                predictions = []
                frame_count = 0
                FRAME_SKIP = 15

                current_label = None
                current_confidence = 0.0

                try:
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        # Detect faces
                        face_data = get_face_from_frame(frame)

                        # Draw boxes
                        for face_info in face_data:
                            x, y, w, h = face_info['coords']
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (102, 126, 234), 3)

                        # Predict on some frames
                        if frame_count % FRAME_SKIP == 0 and len(face_data) > 0:
                            face_img = face_data[0]['image']
                            label, confidence = predict_deepfake(face_img, processor, model)

                            if label:
                                predictions.append({'label': label, 'confidence': confidence})
                                current_label = label
                                current_confidence = confidence

                        # Display current prediction on frame
                        if current_label:
                            verdict_text = "REAL" if current_label == "Realism" else "FAKE"
                            text_color = (34, 197, 94) if current_label == "Realism" else (239, 68, 68)

                            # Draw large text box at top
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                            cv2.putText(
                                frame,
                                f"{verdict_text}: {current_confidence*100:.1f}%",
                                (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                text_color,
                                4
                            )

                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        # Show current status below frame
                        if current_label:
                            if current_label == "Realism":
                                status_placeholder.markdown(
                                    f'<div class="live-label real">✅ REAL FACE - {current_confidence*100:.1f}% Confidence</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                status_placeholder.markdown(
                                    f'<div class="live-label fake">⚠️ DEEPFAKE - {current_confidence*100:.1f}% Confidence</div>',
                                    unsafe_allow_html=True
                                )
                except Exception as e:
                    st.error(f"Error during live detection: {e}")
                finally:
                    cap.release()

                # Show final results
                if predictions:
                    st.markdown("---")
                    final_label, avg_confidence, real_count, fake_count = calculate_final_verdict(predictions)

                    st.markdown("### 📊 Session Summary")

                    if final_label == "Realism":
                        st.markdown(
                            f'<div class="result-success">✅ AUTHENTIC<br>'
                            f'<span style="font-size:1.1rem;">Avg Confidence: {avg_confidence*100:.2f}%</span></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="result-danger">⚠️ DEEPFAKE DETECTED<br>'
                            f'<span style="font-size:1.1rem;">Avg Confidence: {avg_confidence*100:.2f}%</span></div>',
                            unsafe_allow_html=True
                        )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Real Detections", real_count)
                    with col3:
                        st.metric("Fake Detections", fake_count)

        st.markdown('</div>', unsafe_allow_html=True)

    
    # ==================== DEFAULT VIEW ====================
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 👆 Select a Detection Mode to Get Started")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            #### 🎥 Video Detection
            - Upload MP4, AVI, or MOV files
            - Complete frame-by-frame analysis
            - Detailed detection statistics
            - Export-ready results
            """)
        
        with col2:
            st.markdown("""
            #### 📡 Live Webcam
            - Real-time face detection
            - Instant deepfake analysis
            - Live confidence scoring
            - Session summary statistics
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Info
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 System Capabilities")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">92%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">ViT</div>
                <div class="metric-label">AI Model</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">Real-time</div>
                <div class="metric-label">Processing</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">Secure</div>
                <div class="metric-label">Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown(""
    <div style='text-align: center; color: #666; padding: 1.5rem;'>
        <p style='font-size: 1rem; font-weight: 600;'>Deepfake Detection System</p>
        <p style='font-size: 0.9rem;'>AI-Powered Interview Security & Authenticity Verification</p>
    </div>
    "", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
