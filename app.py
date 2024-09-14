from facedetection import face_detection_loop
from gesturedetection import gesture_loop

def start_app():
    # Start the camera loop in a separate thread
    import threading
    camera_thread = threading.Thread(target=face_detection_loop)
    camera_thread.daemon = True  # Set as a daemon thread so it will close when the main program exits
    camera_thread.start()

    # Start the gesture loop in a separate thread
    gesture_thread = threading.Thread(target=gesture_loop)
    gesture_thread.daemon = True  # Set as a daemon thread so it will close when the main program exits
    gesture_thread.start()


if __name__ == "__main__":
    start_app()
