import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pyautogui
import time

# Function to detect and crop eyes from a video frame
def detect_and_crop_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    cropped_eyes = []
    for (x, y, w, h) in eyes:
        eye_img = frame[y:y + h, x:x + w]
        cropped_eyes.append(eye_img)
    return cropped_eyes

# Function to play the video, move the cursor, save cropped eyes, and display the direction
def play_video_and_move_cursor(video_path, model, output_dir):
    video_capture = cv2.VideoCapture(video_path)
    label_mapping = {3: 'Left', 4: 'Right', 5: 'Up', 6: 'Down'}
    
    screen_width, screen_height = pyautogui.size()  # Get screen size
    cursor_speed = 20  # Set cursor movement speed

    # Resize video frames for faster processing
    target_width, target_height = 320, 240  # Smaller size for the video

    frame_count = 0  # Frame count for saving images
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the video frame to a smaller size
        frame_resized = cv2.resize(frame, (target_width, target_height))

        cropped_eyes = detect_and_crop_eyes(frame_resized)
        direction_text = 'No Eye Detected'

        for eye_img in cropped_eyes:
            # Save cropped eye image
            eye_img_path = os.path.join(output_dir, f'eye_{frame_count}.jpg')
            cv2.imwrite(eye_img_path, eye_img)

            # Preprocess the image for prediction
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not already
            eye_img = cv2.resize(eye_img, (50, 50))  # Resize to 50x50 pixels
            eye_img_flatten = eye_img.flatten().reshape(1, -1)

            # Predict the direction
            prediction = model.predict(eye_img_flatten)
            direction = prediction[0]

            # Move the cursor based on the predicted direction with smooth motion
            if direction == 3:  # Move cursor left
                pyautogui.move(-cursor_speed, 0, duration=0.1)
                direction_text = 'Left'
            elif direction == 4:  # Move cursor right
                pyautogui.move(cursor_speed, 0, duration=0.1)
                direction_text = 'Right'
            elif direction == 5:  # Move cursor up
                pyautogui.move(0, -cursor_speed, duration=0.1)
            elif direction == 6:  # Move cursor down
                pyautogui.move(0, cursor_speed, duration=0.1)
                direction_text = 'Down'

            frame_count += 1  # Increment the frame counter

        # Overlay the direction text on the frame
        cv2.putText(frame_resized, f'Direction: {direction_text}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)

        # Show the resized frame in a small OpenCV window
        cv2.imshow('Eye Movement Detection (Small)', frame_resized)

        # Position the window to the top-left corner
        cv2.moveWindow('Eye Movement Detection (Small)', 0, 0)

        # Exit condition for video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main execution
video_path = r"C:\Users\91849\OneDrive\Desktop\Desktop\ds project.mp4"  # Change the path to your video
output_dir = r"C:\Users\91849\OneDrive\Desktop\output_eye_tracking_image"  # Output folder for saved eye images

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the pre-trained model (ensure the model file exists)
model = joblib.load('eye_direction_model.joblib')

# Play the video and move the cursor based on detected eye movements
play_video_and_move_cursor(video_path, model, output_dir)
