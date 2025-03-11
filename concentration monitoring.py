import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# Initialize Mediapipe Face Mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_ear(eye_landmarks, image_width, image_height):
    """
    Calculates the Eye Aspect Ratio (EAR) for an eye.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    # Expecting 6 landmark points per eye
    points = [
        (int(eye_landmarks[idx].x * image_width), int(eye_landmarks[idx].y * image_height))
        for idx in range(len(eye_landmarks))
    ]
    p1, p2, p3, p4, p5, p6 = points

    vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0

def analyze_focus(face_landmarks, image_width, image_height):
    """
    Analyzes focus based on the eyes (using EAR) and the nose tip.
    Uses normalized coordinates for the nose tip to determine gaze direction.
    """
    if face_landmarks is None:
        return "No Face Detected"
    
    EAR_THRESHOLD = 0.25

    # Define landmark indices for each eye (using six points per eye)
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    left_eye_landmarks = [face_landmarks.landmark[idx] for idx in left_eye_indices]
    right_eye_landmarks = [face_landmarks.landmark[idx] for idx in right_eye_indices]

    left_ear = calculate_ear(left_eye_landmarks, image_width, image_height)
    right_ear = calculate_ear(right_eye_landmarks, image_width, image_height)
    avg_ear = (left_ear + right_ear) / 2.0

    # First, check for possible blink/closed eyes
    if avg_ear < EAR_THRESHOLD:
        return "Low Attention (Possible Blink or Closed Eyes)"

    # Use the nose tip landmark (index 1 is commonly used for the tip)
    nose_tip = face_landmarks.landmark[1]
    nose_x = nose_tip.x  # normalized (0 to 1)
    nose_y = nose_tip.y  # normalized (0 to 1)

    # Set threshold values (adjust these as needed)
    # Check horizontal deviation first.
    if nose_x < 0.3:
        return "Not Focused (Looking Left)"
    elif nose_x > 0.7:
        return "Not Focused (Looking Right)"
    # Then check vertical deviation.
    elif nose_y < 0.3:
        return "Not Focused (Looking Up)"
    elif nose_y > 0.7:
        return "Not Focused (Looking Down)"
    
    return "Focused"

def send_report_email(email, report):
    """
    Send the session report to the provided email address.
    
    """
    print(f"Preparing to send email to {email}") 
    try:
        # You would need to set up your email credentials
        sender_email = "venkateshbadarala98@gmail.com"  # Replace with your email
        password = "wlyg imiy dzze nqkm"  # Replace with your app password
        print("Creating email content...") 
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email
        message["Subject"] = "Focus Monitor Session Report"

        message.attach(MIMEText(report, "plain"))
        print("Email content prepared.")  # Debug print

        # Connect to server and send email
        print("Connecting to Gmail SMTP server...")  # Debug print
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            print("Logging into the server...")  # Debug print
            server.login(sender_email, password)
            print("Login successful.")  # Debug print
            server.sendmail(sender_email, email, message.as_string())
            print("Email sent successfully!")  # Debug print
        return True
    except Exception as e:
        print(f"Failed to send email due to error: {e}")  # Debug print
        st.error(f"Failed to send email: {e}")
        return False
    
def generate_report(focus_data):
    """
    Generate a comprehensive report from the session data
    """
    if not focus_data:
        return "No data was collected during the session."
    
    total_samples = len(focus_data)
    status_counts = {}
    
    # Count occurrences of each status
    for timestamp, status in focus_data:
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1
    
    # Calculate percentages
    percentages = {status: (count / total_samples) * 100 for status, count in status_counts.items()}
    
    # Generate the report
    report = "Focus Monitor Session Report\n"
    report += "=" * 30 + "\n\n"
    report += f"Session Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    report += f"Session Time: {datetime.now().strftime('%H:%M:%S')}\n"
    report += f"Total Monitoring Duration: {len(focus_data)} seconds\n\n"
    
    report += "Focus Status Breakdown:\n"
    report += "-" * 25 + "\n"
    
    for status, count in status_counts.items():
        report += f"{status}: {count} seconds ({percentages[status]:.1f}%)\n"
    
    # Calculate focus percentage
    focus_percentage = percentages.get("Focused", 0)
    report += "\n"
    report += f"Overall Focus Rate: {focus_percentage:.1f}%\n"
    
    # Add a qualitative assessment
    if focus_percentage >= 80:
        report += "\nQualitative Assessment: Excellent focus during the session!\n"
    elif focus_percentage >= 60:
        report += "\nQualitative Assessment: Good focus, with some distractions.\n"
    elif focus_percentage >= 40:
        report += "\nQualitative Assessment: Moderate focus, significant room for improvement.\n"
    else:
        report += "\nQualitative Assessment: Low focus detected. Consider strategies to improve attention.\n"
    
    return report

def validate_email(email):
    """
    Validate email format
    """
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

# Streamlit Interface
st.title("Focus Monitor for Remote Learning")
st.write("A real-time system to monitor attention during online learning.")

# Initialize session state for focus data if it doesn't exist
if 'focus_data' not in st.session_state:
    st.session_state.focus_data = []

# Track if email has been sent for current session
if 'email_sent' not in st.session_state:
    st.session_state.email_sent = False

# 1. Add email input field
email = st.text_input("Enter your email address for the session report:")

# 2. Only allow starting after email is provided
start_button_disabled = not validate_email(email)
if start_button_disabled:
    st.warning("Please enter a valid email address before starting the session.")

start_monitoring = st.checkbox("Start Monitoring", disabled=start_button_disabled)

# Detect when checkbox is unchecked (monitoring ended)
if 'previous_monitoring_state' not in st.session_state:
    st.session_state.previous_monitoring_state = False

# Check for state change from monitoring to not monitoring
monitoring_just_stopped = st.session_state.previous_monitoring_state and not start_monitoring
st.session_state.previous_monitoring_state = start_monitoring

if start_monitoring:
    # Reset focus data when starting a new session
    if 'session_started' not in st.session_state or not st.session_state.session_started:
        st.session_state.focus_data = []
        st.session_state.session_started = True
        st.session_state.email_sent = False
    
    st.write("Monitoring started. Uncheck the box to quit.")
    cap = cv2.VideoCapture(0)

    # For performance: track FPS
    prev_time = time.time()
    frame_count = 0
    fps_sum = 0

    # Add status log
    status_log = st.empty()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from webcam.")
                break

            # Resize for performance and flip for mirror effect
            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # Process image using Mediapipe
            result = face_mesh.process(image)
            attention_status = "Analyzing..."
            
            if result.multi_face_landmarks:
                # For each face detected (usually one)
                for face_landmarks in result.multi_face_landmarks:
                    attention_status = analyze_focus(face_landmarks, image.shape[1], image.shape[0])
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
            
            # Record the current status
            current_time = datetime.now().strftime("%H:%M:%S")
            st.session_state.focus_data.append((current_time, attention_status))
            
            # Update status log (show last 5 records)
            log_text = "Recent Status Log:\n"
            for t, s in st.session_state.focus_data[-5:]:
                log_text += f"{t}: {s}\n"
            status_log.text(log_text)
            
            # Calculate FPS for performance measurement
            current_time = time.time()
            elapsed = current_time - prev_time
            fps = 1 / elapsed if elapsed > 0 else 0
            prev_time = current_time
            fps_sum += fps
            frame_count += 1

            # Convert the image back to BGR for OpenCV display
            annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(annotated_image, f"Status: {attention_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the annotated image in OpenCV
            cv2.imshow("Focus Monitor", annotated_image)
            # Also display in Streamlit
            st.image(annotated_image, channels="BGR", use_column_width=True)
            
            # Check if checkbox is unchecked or user pressed 'q' (for OpenCV window)
            if not start_monitoring:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = fps_sum / frame_count if frame_count > 0 else 0
        st.write(f"Monitoring ended. Average FPS: {avg_fps:.2f}")
        
        # 3. Generate the session report
        if st.session_state.focus_data:
            report = generate_report(st.session_state.focus_data)
            st.subheader("Session Report")
            st.text_area("Report", report, height=400)
            
            # 4. Send email with report
            if st.button("Send Report to Your Email"):
                st.info("Sending report, please wait...")  # Inform user
                if send_report_email(email, report):
                    st.success(f"Report sent successfully to {email}")
                    st.session_state.email_sent = True
                else:
                    st.error("Failed to send the report. Please check the email settings.")

            
            # Reset session started flag
            st.session_state.session_started = False
else:
    # If we have data but monitoring is stopped, show the report option
    if 'focus_data' in st.session_state and st.session_state.focus_data and st.session_state.get('session_started', False) == True:
        report = generate_report(st.session_state.focus_data)
        st.subheader("Session Report")
        st.text_area("Report", report, height=400)
        
        # Automatically send the email when monitoring is stopped (checkbox unchecked)
        if monitoring_just_stopped and not st.session_state.email_sent and validate_email(email):
            st.info("Sending report automatically, please wait...")
            if send_report_email(email, report):
                st.success(f"Report sent automatically to {email}")
                st.session_state.email_sent = True
            else:
                st.error("Failed to automatically send the report. You can try sending it manually.")
                # Still show the manual button as a fallback
                if st.button("Send Report to Your Email"):
                    st.info("Trying to send report again, please wait...")
                    if send_report_email(email, report):
                        st.success(f"Report sent successfully to {email}")
                        st.session_state.email_sent = True
                    else:
                        st.error("Failed to send the report again. Please check the email settings.")
        # If email wasn't sent automatically (perhaps due to an error), still show the manual option
        elif not st.session_state.email_sent:
            if st.button("Send Report to Your Email"):
                st.info("Sending report, please wait...")
                if send_report_email(email, report):
                    st.success(f"Report sent successfully to {email}")
                    st.session_state.email_sent = True
                else:
                    st.error("Failed to send the report. Please check the email settings.")
        
        # Reset session started flag
        st.session_state.session_started = False
