import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from datetime import datetime

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_ear(eye_landmarks, image_width, image_height):
    # Calculate Eye Aspect Ratio (EAR) for a single eye
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    p1, p2, p3, p4, p5, p6 = [
        (int(eye_landmarks[idx].x * image_width), int(eye_landmarks[idx].y * image_height))
        for idx in [0, 1, 2, 3, 4, 5]
    ]

    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0

def analyze_focus(face_landmarks, image_width, image_height):
    if face_landmarks is None:
        return "No Face Detected"

    # EAR Threshold for blink detection
    EAR_THRESHOLD = 0.25

    # Left and Right eye landmarks
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    # Extract coordinates for both eyes
    left_eye_ear = calculate_ear([face_landmarks.landmark[idx] for idx in left_eye_indices], image_width, image_height)
    right_eye_ear = calculate_ear([face_landmarks.landmark[idx] for idx in right_eye_indices], image_width, image_height)

    # Average EAR for both eyes
    avg_ear = (left_eye_ear + right_eye_ear) / 2.0

    if avg_ear < EAR_THRESHOLD:
        return "Low Attention (Possible Blink or Closed Eyes)"

    # Detect if gaze is not on screen by analyzing head pose
    nose_tip = face_landmarks.landmark[1]
    nose_x = int(nose_tip.x * image_width)
    if nose_x < image_width * 0.3 or nose_x > image_width * 0.7:
        return "Not Focused (Looking Away)"

    # If no issues detected, assume focused
    return "Focused"

# Streamlit Interface
st.title("Focus Monitor for Remote Learning")
st.write("A real-time system to monitor attention during online learning.")

start_monitoring = st.checkbox("Start Monitoring")

if start_monitoring:
    st.write("Monitoring started. Stop the checkbox to quit.")
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from webcam.")
                break

            # Flip image horizontally for a mirror view and resize for performance
            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe
            result = face_mesh.process(image)
            attention_status = "No Face Detected"

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Analyze focus level
                    attention_status = analyze_focus(face_landmarks, image.shape[1], image.shape[0])

                    # Draw landmarks on the face
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    )

            # Convert back to BGR for OpenCV display
            annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(annotated_image, attention_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display in Streamlit
            st.image(annotated_image, channels="BGR", use_column_width=True)

            # Display using OpenCV
            cv2.imshow("Focus Monitor", annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        st.write("Monitoring ended.")