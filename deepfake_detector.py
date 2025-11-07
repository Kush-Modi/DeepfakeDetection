import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import time
from collections import Counter
import os

# Load Deep-Fake-Detector pre-trained model from Hugging Face
print("Loading model...")
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")


def get_face_from_frame(frame):
    """Extract faces from frame using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    face_data = []
    for (x, y, w, h) in faces:
        # Store face region and coordinates
        face_img = frame[y:y+h, x:x+w]
        face_data.append({
            'image': face_img,
            'coords': (x, y, w, h)
        })
    
    return face_data


def predict_deepfake(face_image):
    """Predict if face is real or deepfake"""
    try:
        # Convert BGR to RGB and then to PIL Image
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Preprocess and predict
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class_idx = logits.argmax(-1).item()
        
        # Get label from model config
        label = model.config.id2label[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        
        return label, confidence
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0


def calculate_final_verdict(predictions):
    """Calculate final verdict based on majority voting"""
    if not predictions:
        return "Unknown", 0.0, 0, 0
    
    # Count occurrences of each label
    label_counts = Counter([pred['label'] for pred in predictions])
    
    # Get majority label
    final_label = label_counts.most_common(1)[0][0]
    
    # Calculate average confidence for the final label
    label_confidences = [pred['confidence'] for pred in predictions if pred['label'] == final_label]
    avg_confidence = sum(label_confidences) / len(label_confidences) if label_confidences else 0.0
    
    # Count real vs fake
    real_count = label_counts.get("Realism", 0)
    fake_count = label_counts.get("Deepfake", 0)
    
    return final_label, avg_confidence, real_count, fake_count


def draw_analysis_ui(frame, time_remaining, predictions, current_verdict, real_count, fake_count, avg_confidence, mode="Live"):
    """Draw analysis UI on frame"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    title = f"DEEPFAKE ANALYSIS - {mode.upper()} MODE"
    cv2.putText(frame, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Time remaining
    cv2.putText(frame, f"Time Remaining: {int(time_remaining)}s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Predictions collected
    cv2.putText(frame, f"Predictions Collected: {len(predictions)}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Current stats
    cv2.putText(frame, f"Real: {real_count} | Fake: {fake_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Current verdict (if available)
    if current_verdict:
        verdict_color = (0, 255, 0) if current_verdict == "Realism" else (0, 0, 255)
        verdict_text = "REAL" if current_verdict == "Realism" else "FAKE"
        cv2.putText(frame, f"Leading: {verdict_text} ({avg_confidence*100:.1f}%)", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2)


def draw_final_verdict(frame, final_label, avg_confidence, real_count, fake_count, total_predictions, wait_text=""):
    """Draw final verdict overlay"""
    height, width = frame.shape[:2]
    
    # Create full overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Determine verdict display
    if final_label == "Realism":
        verdict_text = "REAL"
        verdict_color = (0, 255, 0)
        status = "AUTHENTIC"
    else:
        verdict_text = "FAKE"
        verdict_color = (0, 0, 255)
        status = "DEEPFAKE DETECTED"
    
    # Main verdict text
    cv2.putText(frame, "FINAL VERDICT", (width//2 - 200, height//2 - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.putText(frame, verdict_text, (width//2 - 100, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, verdict_color, 5)
    
    cv2.putText(frame, status, (width//2 - 180, height//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, verdict_color, 2)
    
    # Statistics
    cv2.putText(frame, f"Confidence: {avg_confidence*100:.2f}%", (width//2 - 150, height//2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Analysis: {real_count} Real | {fake_count} Fake", (width//2 - 200, height//2 + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Total Predictions: {total_predictions}", (width//2 - 150, height//2 + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if wait_text:
        cv2.putText(frame, wait_text, (width//2 - 220, height//2 + 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)


def run_video_file_detection(video_path):
    """Run deepfake detection on uploaded video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo Properties:")
    print(f"- FPS: {fps}")
    print(f"- Total Frames: {total_frames}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"\nStarting video analysis. Press 'q' to quit.\n")
    
    # Analysis parameters
    ANALYSIS_WINDOW = min(120, duration)  # Use video duration if less than 2 minutes
    FRAME_SKIP = max(1, fps // 2)  # Process 2 frames per second
    
    # State variables
    predictions = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed_time = frame_count / fps if fps > 0 else 0
        
        # Process frame for face detection (every frame for smooth display)
        face_data = get_face_from_frame(frame)
        
        # Draw face boxes
        for face_info in face_data:
            x, y, w, h = face_info['coords']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Run prediction only on specific frames
        if frame_count % FRAME_SKIP == 0 and len(face_data) > 0:
            # Use the first detected face
            face_img = face_data[0]['image']
            label, confidence = predict_deepfake(face_img)
            
            if label:
                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'timestamp': elapsed_time
                })
                print(f"Frame {frame_count}/{total_frames}: {label} ({confidence*100:.2f}%) - Total: {len(predictions)}")
        
        # Calculate current leading verdict
        current_verdict, temp_confidence, temp_real, temp_fake = calculate_final_verdict(predictions)
        
        # Draw analysis UI
        time_remaining = ANALYSIS_WINDOW - elapsed_time
        draw_analysis_ui(frame, time_remaining, predictions, current_verdict, temp_real, temp_fake, temp_confidence, mode="Video")
        
        # Display frame
        cv2.imshow("Video Deepfake Detection - Press 'q' to quit", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate final verdict
    final_label, avg_confidence, real_count, fake_count = calculate_final_verdict(predictions)
    
    print(f"\n{'='*50}")
    print(f"VIDEO ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Final Verdict: {final_label}")
    print(f"Average Confidence: {avg_confidence*100:.2f}%")
    print(f"Real Count: {real_count}")
    print(f"Fake Count: {fake_count}")
    print(f"Total Predictions: {len(predictions)}")
    print(f"{'='*50}\n")
    
    # Show final verdict for 10 seconds or until key press
    ret, last_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, last_frame = cap.read()
    
    if ret:
        draw_final_verdict(last_frame, final_label, avg_confidence, real_count, fake_count, len(predictions), 
                          wait_text="Press any key to continue...")
        cv2.imshow("Video Deepfake Detection - Press 'q' to quit", last_frame)
        cv2.waitKey(10000)  # Wait 10 seconds or until key press
    
    cap.release()
    cv2.destroyAllWindows()


def run_live_webcam_detection():
    """Run deepfake detection with 2-minute analysis windows on webcam"""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting 2-minute analysis detection. Press 'q' to quit.")
    
    # Analysis parameters
    ANALYSIS_WINDOW = 120  # 2 minutes in seconds
    VERDICT_DISPLAY_TIME = 5  # Show verdict for 5 seconds
    FRAME_SKIP = 15  # Process every 15th frame to reduce computation
    
    # State variables
    predictions = []
    start_time = time.time()
    frame_count = 0
    analysis_round = 1
    show_verdict = False
    verdict_start_time = 0
    
    final_label = None
    avg_confidence = 0.0
    real_count = 0
    fake_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Show verdict screen
        if show_verdict:
            draw_final_verdict(frame, final_label, avg_confidence, real_count, fake_count, len(predictions),
                             wait_text="Starting new analysis in 5 seconds...")
            cv2.imshow("Live Deepfake Detection - Press 'q' to quit", frame)
            
            # Check if verdict display time is over
            if current_time - verdict_start_time >= VERDICT_DISPLAY_TIME:
                show_verdict = False
                predictions = []
                start_time = time.time()
                analysis_round += 1
                print(f"\n--- Starting Analysis Round {analysis_round} ---")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Process frame for face detection (every frame for smooth display)
        face_data = get_face_from_frame(frame)
        
        # Draw face boxes
        for face_info in face_data:
            x, y, w, h = face_info['coords']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Run prediction only on specific frames
        if frame_count % FRAME_SKIP == 0 and len(face_data) > 0:
            # Use the first detected face
            face_img = face_data[0]['image']
            label, confidence = predict_deepfake(face_img)
            
            if label:
                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'timestamp': elapsed_time
                })
                print(f"Frame {frame_count}: {label} ({confidence*100:.2f}%) - Total: {len(predictions)}")
        
        # Calculate current leading verdict
        current_verdict, temp_confidence, temp_real, temp_fake = calculate_final_verdict(predictions)
        
        # Draw analysis UI
        time_remaining = ANALYSIS_WINDOW - elapsed_time
        draw_analysis_ui(frame, time_remaining, predictions, current_verdict, temp_real, temp_fake, temp_confidence, mode="Live")
        
        # Check if analysis window is complete
        if elapsed_time >= ANALYSIS_WINDOW:
            # Calculate final verdict
            final_label, avg_confidence, real_count, fake_count = calculate_final_verdict(predictions)
            
            print(f"\n{'='*50}")
            print(f"ANALYSIS ROUND {analysis_round} COMPLETE")
            print(f"{'='*50}")
            print(f"Final Verdict: {final_label}")
            print(f"Average Confidence: {avg_confidence*100:.2f}%")
            print(f"Real Count: {real_count}")
            print(f"Fake Count: {fake_count}")
            print(f"Total Predictions: {len(predictions)}")
            print(f"{'='*50}\n")
            
            # Show verdict screen
            show_verdict = True
            verdict_start_time = time.time()
        
        # Display frame
        cv2.imshow("Live Deepfake Detection - Press 'q' to quit", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal analysis rounds completed: {analysis_round}")


def display_menu():
    """Display main menu and get user choice"""
    print("\n" + "="*60)
    print(" " * 15 + "DEEPFAKE DETECTION SYSTEM")
    print("="*60)
    print("\nChoose detection mode:")
    print("\n1. Live Webcam Detection (2-minute analysis windows)")
    print("2. Upload Video File Detection (full video analysis)")
    print("3. Exit")
    print("\n" + "="*60)
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main function with menu system"""
    while True:
        choice = display_menu()
        
        if choice == '1':
            print("\n[Starting Live Webcam Detection...]")
            run_live_webcam_detection()
            
        elif choice == '2':
            print("\n[Upload Video File Detection]")
            video_path = input("Enter the full path to your video file: ").strip()
            # Remove quotes if user added them
            video_path = video_path.strip('"').strip("'")
            run_video_file_detection(video_path)
            
        elif choice == '3':
            print("\nExiting Deepfake Detection System. Goodbye!")
            break
        
        # Ask if user wants to continue
        if choice != '3':
            continue_choice = input("\nReturn to main menu? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nExiting Deepfake Detection System. Goodbye!")
                break


if __name__ == "__main__":
    main()
