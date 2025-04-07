import os
import re
import numpy as np
import cv2

def stitch_original_images_with_blending(folder_path, output_path="stitched_final_clean_blended.png"):
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
            img = cv2.imread(img_path)  # Load in color (BGR)
            if img is not None:
                images.append((x, y, img, img_file))
    
    if not images:
        print("No valid images with X, Y coordinates found.")
        return
    
    # Get image dimensions (assuming all are the same size)
    img_width, img_height = images[0][2].shape[1], images[0][2].shape[0]
    overlap_x = img_width // 2  # 50% overlap in X
    overlap_y = img_height // 2  # 50% overlap in Y
    
    # Set up the grid with 50% overlap
    x_coords = sorted(set(c[0] for c in images))
    y_coords = sorted(set(c[1] for c in images), reverse=True)  # Reverse Y for top-to-bottom
    grid_width = len(x_coords)
    grid_height = len(y_coords)
    canvas_width = img_width + (grid_width - 1) * overlap_x
    canvas_height = img_height + (grid_height - 1) * overlap_y
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)  # Float for blending
    weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)  # Track weights for blending
    print(f"Canvas size: {canvas_width}x{canvas_height} px")
    
    # Map coordinates to grid indices
    x_to_index = {x: i for i, x in enumerate(x_coords)}
    y_to_index = {y: i for i, y in enumerate(y_coords)}
    
    # Create weight masks for blending
    mask_x = np.linspace(1, 0, overlap_x, dtype=np.float32)  # Left-to-right fade for X overlap
    mask_y = np.linspace(1, 0, overlap_y, dtype=np.float32)  # Top-to-bottom fade for Y overlap
    
    # Place and blend each image
    for x, y, img, fname in images:
        grid_x = x_to_index[x]
        grid_y = y_to_index[y]
        px = grid_x * overlap_x
        py = grid_y * overlap_y
        
        # Convert image to float for blending
        img_float = img.astype(np.float32)
        
        # Create a weight mask for this tile
        tile_weight = np.ones((img_height, img_width), dtype=np.float32)
        
        # Apply X-direction blending
        if grid_x > 0:  # Left overlap with previous tile
            tile_weight[:, :overlap_x] *= mask_x[::-1]  # Fade in from left
        if grid_x < grid_width - 1:  # Right overlap with next tile
            tile_weight[:, -overlap_x:] *= mask_x  # Fade out to right
        
        # Apply Y-direction blending
        if grid_y > 0:  # Top overlap with previous tile
            tile_weight[:overlap_y, :] *= mask_y[::-1][:, np.newaxis]  # Fade in from top
        if grid_y < grid_height - 1:  # Bottom overlap with next tile
            tile_weight[-overlap_y:, :] *= mask_y[:, np.newaxis]  # Fade out to bottom
        
        # Add weighted image to canvas
        try:
            canvas[py:py+img_height, px:px+img_width] += img_float * tile_weight[:, :, np.newaxis]
            weight_map[py:py+img_height, px:px+img_width] += tile_weight
            print(f"Blended {fname} at grid position ({grid_x}, {grid_y}), canvas pos ({px}, {py})")
        except ValueError as e:
            print(f"Error blending {fname}: {e}. Likely due to canvas size mismatch.")
            return
    
    # Normalize by weight map to avoid over-brightening
    mask = weight_map > 0
    canvas[mask] /= weight_map[mask, np.newaxis]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)  # Convert back to uint8
    
    # Save the stitched image
    cv2.imwrite(output_path, canvas)
    print(f"Final blended stitched image saved to {output_path}")
    if np.all(canvas == 0):
        print("Warning: Stitched image is completely black!")

# Main execution
if __name__ == "__main__":
    folder_path = "/Users/alansaj/musemicroscopyalan/AutoCapture_20250402_161903/Exported"  # Original images
    print("Stitching original images with 50% overlap and blending...")
    stitch_original_images_with_blending(folder_path)
