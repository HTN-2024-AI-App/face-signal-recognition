import cv2
import numpy as np
import threading
from flask import Flask, jsonify
from app import app

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variable to store the latest result
latest_result = False

def are_eyes_visible(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            return True
    return False

def face_detection_loop():
    global latest_result
    cap = cv2.VideoCapture(0)  # Use default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        latest_result = are_eyes_visible(frame)
        print(f"User is looking at screen: {latest_result}")

    cap.release()

@app.route('/face_detection', methods=['GET'])
def get_latest_result():
    return jsonify({"res": latest_result})


# starting thread
camera_thread = threading.Thread(target=face_detection_loop)
camera_thread.daemon = True  # Set as a daemon thread so it will close when the main program exits
camera_thread.start()



if __name__ == "__main__":
    # Start the camera loop in a separate thread
    camera_thread = threading.Thread(target=face_detection_loop)
    camera_thread.daemon = True  # Set as a daemon thread so it will close when the main program exits
    camera_thread.start()

