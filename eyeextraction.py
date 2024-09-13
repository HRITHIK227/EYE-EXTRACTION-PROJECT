import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pyautogui

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

# Function to label and save eye images
def label_and_save_eyes(output_dir, directions, segment_duration, video_capture):
    frame_count = 0
    current_direction = 0
    
    for direction in directions:
        dir_path = os.path.join(output_dir, direction)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        cropped_eyes = detect_and_crop_eyes(frame)

        # Automatically label based on video segment
        direction = directions[current_direction]
        if frame_count > (current_direction + 1) * segment_duration:
            current_direction += 1
            if current_direction >= len(directions):
                break  # Stop if all directions are processed

        for eye_img in cropped_eyes:
            eye_img_path = os.path.join(output_dir, direction, f'eye_{frame_count}.jpg')
            cv2.imwrite(eye_img_path, eye_img)

        frame_count += 1

# Function to train the eye detection model
def train_model(output_dir, directions):
    label_mapping = {'left': 3, 'right': 4, 'up': 5, 'down': 6}
    data, labels = [], []

    for direction in directions:
        direction_path = os.path.join(output_dir, direction)
        for img_name in os.listdir(direction_path):
            img_path = os.path.join(direction_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))  # Ensure images are resized to 50x50
            data.append(img.flatten())
            labels.append(label_mapping[direction])

    data = np.array(data)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Predict the test set to evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    return model

# Function to play the video, move the cursor, and display the direction
def play_video_and_move_cursor(video_path, model):
    video_capture = cv2.VideoCapture(video_path)
    label_mapping = {3: 'Left', 4: 'Right', 5: 'Up', 6: 'Down'}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        cropped_eyes = detect_and_crop_eyes(frame)
        direction_text = 'No Eye Detected'

        for eye_img in cropped_eyes:
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not already
            eye_img = cv2.resize(eye_img, (50, 50))  # Resize to 50x50 pixels
            eye_img_flatten = eye_img.flatten().reshape(1, -1)

            # Predict the direction
            prediction = model.predict(eye_img_flatten)
            direction = prediction[0]

            # Move the cursor based on the predicted direction
            if direction == 3:  # Move cursor left
                pyautogui.move(-10, 0)
                direction_text = 'Left'
            elif direction == 4:  # Move cursor right
                pyautogui.move(10, 0)
                direction_text = 'Right'
            elif direction == 5:  # Move cursor up
                pyautogui.move(0, -10)
                direction_text = 'Up'
            elif direction == 6:  # Move cursor down
                pyautogui.move(0, 10)
                direction_text = 'Down'

        # Overlay the direction text on the frame
        cv2.putText(frame, f'Direction: {direction_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('Eye Movement Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main execution
video_path = r"C:\Users\91849\OneDrive\Desktop\WhatsApp Video .mp4"
output_dir = r"C:\Users\91849\OneDrive\Desktop\output eye tracking image"
directions = ['left', 'right', 'up', 'down']

# Initialize video capture and calculate segment duration
video_capture = cv2.VideoCapture(video_path)
segment_duration = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) // len(directions)

# Label and save eye images
label_and_save_eyes(output_dir, directions, segment_duration, video_capture)

# Train the model
model = train_model(output_dir, directions)
joblib.dump(model, 'eye_direction_model.joblib')

# Play the video and move the cursor based on detected eye movements
play_video_and_move_cursor(video_path, model)
