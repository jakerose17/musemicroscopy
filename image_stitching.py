import cv
import numpy as np
from PIL import Image

# Set up OpenAI API key
openai.api_key = "your_openai_api_key"

# Function to stitch images with OpenCV
def stitch_images(image_list):
    # Read images
    images = [cv2.imread(img_path) for img_path in image_list]
    
    # Initialize OpenCV stitcher
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        return stitched
    else:
        print("Stitching failed!")
        return None
'''
# Function to send stitched image to GPT-4 Vision
def analyze_with_gpt4(image, description="Microscopic kidney tissue stitching"):
    # Convert OpenCV image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save image to a temporary file
    image_pil.save("stitched_image.png")
    
    # Open file and send to GPT-4 Vision
    with open("stitched_image.png", "rb") as img_file:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision",
            messages=[
                {"role": "system", "content": "You are a medical imaging expert."},
                {"role": "user", "content": f"Analyze this stitched image of {description} and check if it looks accurate."}
            ],
            files={"image": img_file}
        )
    return response["choices"][0]["message"]["content"]

# Example usage
if __name__ == "__main__":
    image_paths = ["kidney_zoom1.jpg", "kidney_zoom2.jpg", "kidney_zoom3.jpg"]
    stitched_image = stitch_images(image_paths)
    
    if stitched_image is not None:
        feedback = analyze_with_gpt4(stitched_image)
        print("GPT-4 Feedback:", feedback)
        '''