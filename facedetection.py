import cv2
import numpy as np
import time

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


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

def main():
    cap = cv2.VideoCapture(0)  # Use default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = are_eyes_visible(frame)
        print(f"User is looking at screen: {result}")
        
    
    cap.release()



if __name__ == "__main__":
    # Example usage
    main()