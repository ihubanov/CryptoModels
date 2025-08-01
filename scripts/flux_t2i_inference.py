
import base64
import os
import requests
from datetime import datetime

# API configuration
api_key = "no-need"
base_url = "http://localhost:8080"
endpoint = f"{base_url}/v1/images/generations"

# Request headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "model": "local-image-generation",
    "prompt": "A beautiful sunset over a calm ocean",
    "n": 1,
    "steps": 4,
    "size": "1024x1024"
}

# Make the POST request
response = requests.post(endpoint, headers=headers, json=payload)

# Handle the response and save the image
if response.status_code == 200:
    response_data = response.json()
    
    if response_data.get("data") and len(response_data["data"]) > 0:
        # Get the first image from the response
        image_data = response_data["data"][0]
        
        # Check if the image has base64 data
        if "b64_json" in image_data and image_data["b64_json"]:
            # Decode the base64 data
            image_bytes = base64.b64decode(image_data["b64_json"])
            
            # Create output directory if it doesn't exist
            output_dir = "tests/generated_images"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            print(f"Image saved successfully to: {filepath}")
        else:
            print("No base64 data found in the response")
    else:
        print("No images generated in the response")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(f"Response: {response.text}")
