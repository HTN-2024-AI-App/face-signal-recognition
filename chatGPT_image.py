import cv2
import base64
import requests
import os
import openai

# Initialize the OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def capture_and_query_openai(prompt, frame, model="text-davinci-003", max_tokens=300):
    # Convert the image to base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Set up the request parameters for OpenAI
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "image": f"data:image/jpeg;base64,{image_base64}"
            }
        ],
        "max_tokens": max_tokens
    }
    
    try:
        # Send the request to the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens
        )
        
        # Return the content of the response
        result = response.choices[0].message['content']
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def camera_loop():
    global latest_result
    cap = cv2.VideoCapture(0)  # Use default camera

    prompt = "Is the person in the picture looking in front of him? Is he holding up his hand in the gesture of praying? Return a json with the following format {\"is_looking\": true, \"is_praying\": true}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show the image in a window
        cv2.imshow('Camera Feed', frame)
        
        # Capture and query Groq
        latest_result = capture_and_query_openai(prompt, frame)
        print(f"User is looking at screen: {latest_result}")

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    camera_loop()
