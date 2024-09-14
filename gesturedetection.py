from groq import Groq
import base64
import cv2
import base64
import requests
import os
import time
import openai
import json
from openai import OpenAI



latest_result_2 = False

# Function to encode the image
def encode_image(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

def capture_and_query_chatgpt(prompt, image_base64, model="gpt-4o", max_tokens=300):
    # Initialize the OpenAI client
    client = OpenAI()

    # Prepare the messages for the API request
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    try:
        # Send the request to the ChatGPT API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )

        # Return the content of the response
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def query_groq(prompt, base64_image):

    client = Groq()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llava-v1.5-7b-4096-preview",
    )
    response_content = chat_completion.choices[0].message.content

    try:
        response_json = json.loads(response_content)
    except json.JSONDecodeError as e:
        # Handle cases where the content is not valid JSON
        print(e)
        print(response_content)
        response_json = {"error": "Invalid JSON response"}

    return response_json

def gesture_loop():
    global latest_result_2
    cap = cv2.VideoCapture(0)  # Use default camera


    prompt = """Analyze the image and provide a JSON string with the following information: Determine if the person in the image has their hands positioned together in a gesture resembling prayer. This includes recognizing situations where: - The hands may be partially visible, possibly being cut off by the edges of the image. - The hands are joined or touching in a manner that resembles a prayer position, where the palms or fingers are pressed together. Please ensure that your analysis considers various possible orientations and positions of the hands to accurately detect if they are kept together in a prayer-like gesture. Return the results in the following JSON format: {"handsPrayer": true or false}. Ensure that the JSON string strictly adheres to this format and contains no additional text, escape characters, or deviations from the specified format."""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show the image in a window
        # cv2.imshow('Camera Feed', frame)
        
        # Capture and query Groq
        base64_image = encode_image(frame)
        latest_result_2 = capture_and_query_chatgpt(prompt, base64_image)
        print(latest_result_2)

    
    cap.release()
    cv2.destroyAllWindows()


@app.route('/praying', methods=['GET'])
def get_latest_result_2():
    return jsonify({"res": latest_result_2})


if __name__ == "__main__": 
    from app import app
    gesture_loop()