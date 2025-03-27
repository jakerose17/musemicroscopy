import os
import re
import numpy as np
import cv2

def load_images(folder_path):
    pattern = r'X(\d+\.\d+)_Y(\d+\.\d+)'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    images = []
    for img_file in image_files:
        match = re.search(pattern, img_file)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((x, y, img, img_file))
    return images

def detect_and_outline_features(img, min_area=300):
    """
    Detect dark/light shapes in a grayscale image, return contours.
    """
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    # Minimal blur
    img_blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    
    # Dual thresholding for dark and light shapes
    _, thresh_dark = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY_INV)  # Dark shapes
    _, thresh_light = cv2.threshold(img_blurred, 200, 255, cv2.THRESH_BINARY)    # Light shapes
    thresh_combined = cv2.bitwise_or(thresh_dark, thresh_light)
    
    # Dilate to connect big shapes
    kernel = np.ones((5, 5), np.uint8)
    thresh_dilated = cv2.dilate(thresh_combined, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return valid_contours

def match_contours(contours1, contours2, similarity_threshold=0.85):
    """
    Compare contours between two images for duplicates.
    """
    matches = []
    for c1 in contours1:
        for c2 in contours2:
            moments1 = cv2.HuMoments(cv2.moments(c1)).flatten()
            moments2 = cv2.HuMoments(cv2.moments(c2)).flatten()
            diff = np.abs(moments1 - moments2)
            similarity = 1 / (1 + np.sum(diff))
            if similarity > similarity_threshold:
                matches.append((c1, c2))
    return matches

def process_and_match_images(folder_path, output_dir="outlined_features", distance_threshold=1.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = load_images(folder_path)
    print(f"Loaded {len(images)} images")
    
    # Step 1: Detect features in each image
    image_contours = {}
    for x, y, img, fname in images:
        print(f"Processing {fname}...")
        contours = detect_and_outline_features(img, min_area=300)
        image_contours[fname] = contours
        
        # Save with green outlines
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, img_color)
        print(f"Saved outlined image to {output_path} with {len(contours)} features")
    
    # Step 2: Find duplicates across images
    duplicate_matches = {}
    for i, (x1, y1, img1, fname1) in enumerate(images):
        for j, (x2, y2, img2, fname2) in enumerate(images[i+1:], start=i+1):
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx <= distance_threshold and dy <= distance_threshold and (dx > 0 or dy > 0):
                print(f"\nComparing {fname1} vs {fname2}")
                matches = match_contours(image_contours[fname1], image_contours[fname2])
                if matches:
                    duplicate_matches[(fname1, fname2)] = matches
                    print(f"Found {len(matches)} matching features between {fname1} and {fname2}")
    
    # Step 3: Highlight duplicates in red
    for (fname1, fname2), matches in duplicate_matches.items():
        img1_color = cv2.imread(os.path.join(output_dir, fname1))
        img2_color = cv2.imread(os.path.join(output_dir, fname2))
        
        matched_c1 = [c1 for c1, _ in matches]
        matched_c2 = [c2 for _, c2 in matches]
        cv2.drawContours(img1_color, matched_c1, -1, (0, 0, 255), 2)  # Red for duplicates
        cv2.drawContours(img2_color, matched_c2, -1, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, fname1), img1_color)
        cv2.imwrite(os.path.join(output_dir, fname2), img2_color)
        print(f"Updated {fname1} and {fname2} with red outlines for duplicates")

# Main execution
if __name__ == "__main__":
    folder_path = "/Users/alansaj/Downloads/musemicroscopy-main/new_images"
    
    print("Step 1: Detecting, outlining, and matching features in individual images...")
    process_and_match_images(folder_path)
