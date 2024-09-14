import cv2
import numpy as np
import time
from flask import Flask, jsonify
import threading

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

app = Flask(__name__)

# Global variable to store the latest result
latest_result = False

def are_eyes_visible(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through each face and check for eyes
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) > 0:
            return True  # Eyes detected
    
    return False  # No eyes detected

def camera_loop():
    global latest_result
    cap = cv2.VideoCapture(0)  # Use default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        latest_result = are_eyes_visible(frame)
        print(f"User is looking at screen: {latest_result}")
        
    
    cap.release()

@app.route('/is_looking', methods=['GET'])
def is_looking():
    return jsonify({"is_looking": latest_result})

if __name__ == "__main__":
    # Start the camera loop in a separate thread
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.daemon = True  # Set as a daemon thread so it will close when the main program exits
    camera_thread.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)