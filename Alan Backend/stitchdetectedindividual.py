import os
import re
import numpy as np
import cv2

def stitch_images_from_folder(folder_path, output_path="stitched_output_features.png"):
    # Regex to parse X and Y coords from filenames
    pattern = r'X(\d+\.\d+)_Y(\d+\.\d+)'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    # Collect coordinates and file paths
    images = []
    for img_file in image_files:
        match = re.search(pattern, img_file)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            img_path = os.path.join(folder_path, img_file)
            images.append((x, y, img_file))
    
    if not images:
        print("No valid images with X, Y coordinates found.")
        return
    
    # Get image dimensions (assuming all are the same size)
    first_img = cv2.imread(os.path.join(folder_path, images[0][2]))
    if first_img is None:
        print(f"Failed to load {images[0][2]}")
        return
    img_width, img_height = first_img.shape[1], first_img.shape[0]
    
    # Set up the grid
    x_coords = sorted(set(c[0] for c in images))
    y_coords = sorted(set(c[1] for c in images), reverse=True)  # Reverse Y for top-to-bottom
    canvas_width = len(x_coords) * img_width
    canvas_height = len(y_coords) * img_height
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Color canvas
    
    # Map coordinates to grid indices
    x_to_index = {x: i for i, x in enumerate(x_coords)}
    y_to_index = {y: i for i, y in enumerate(y_coords)}
    
    # Place each image on the canvas
    for x, y, fname in images:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            px = x_to_index[x] * img_width
            py = y_to_index[y] * img_height
            canvas[py:py+img_height, px:px+img_width] = img
            print(f"Placed {fname} at grid position ({x_to_index[x]}, {y_to_index[y]})")
        else:
            print(f"Failed to load {fname}")
    
    # Save the stitched image
    cv2.imwrite(output_path, canvas)
    print(f"Stitched image saved to {output_path}")

# Main execution
if __name__ == "__main__":
    folder_path = "outlined_features"  # Use your pre-processed folder
    print("Step 2: Stitching images from output_features...")
    stitch_images_from_folder(folder_path)
