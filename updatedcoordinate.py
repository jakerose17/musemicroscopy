import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_images_on_grid(folder_path, output_path=None, figsize=(15, 15), dpi=300):
    """
    Create a composite image from files based on X and Y coordinates extracted from filenames,
    merging images into a cohesive image without axes or labels.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the images
    output_path : str, optional
        Path to save the output image. If None, the image is not saved
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the figure
    
    Returns:
    --------
    None
    """
    # Regular expression to extract X and Y coordinates from filenames
    pattern = r'X(\d+\.\d+)_Y(\d+\.\d+)'
    
    # Get all image files from the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    # Extract coordinates and corresponding image paths
    coordinates = []
    for img_file in image_files:
        match = re.search(pattern, img_file)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            coordinates.append((x, y, os.path.join(folder_path, img_file)))
        else:
            print(f"Could not extract coordinates from {img_file}")
    
    if not coordinates:
        print("No valid coordinates extracted from filenames.")
        return
    
    # Get image size from first image (assume all images are the same size)
    first_img = Image.open(coordinates[0][2])
    img_width, img_height = first_img.size
    
    # Get unique X and Y coordinates to determine grid size
    x_coords = sorted(set(c[0] for c in coordinates))
    y_coords = sorted(set(c[1] for c in coordinates), reverse=True)  # Reverse Y for top-to-bottom placement
    
    # Calculate canvas size based on the number of unique coordinates
    num_x = len(x_coords)
    num_y = len(y_coords)
    canvas_width = num_x * img_width
    canvas_height = num_y * img_height
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)  # RGBA canvas
    
    # Map coordinates to grid indices
    x_to_index = {x: i for i, x in enumerate(x_coords)}
    y_to_index = {y: i for i, y in enumerate(y_coords)}
    
    # Function to place image on canvas at grid position
    def place_image(canvas, x, y, img_path):
        try:
            img = Image.open(img_path).convert('RGBA')
            img_array = np.array(img)
            
            # Map coordinates to grid indices
            grid_x = x_to_index[x]
            grid_y = y_to_index[y]
            
            # Calculate pixel position on canvas (no scaling, just grid placement)
            px = grid_x * img_width
            py = grid_y * img_height
            
            # Define region on canvas
            y_start = py
            y_end = py + img_height
            x_start = px
            x_end = px + img_width
            
            # Ensure bounds are within canvas (should always be true with this approach)
            if x_start < 0 or x_end > canvas_width or y_start < 0 or y_end > canvas_height:
                print(f"Image at {img_path} exceeds canvas bounds at grid_x:{grid_x}, grid_y:{grid_y}")
                return False
            
            # Place image, overwriting overlapping pixels (latest image takes priority)
            canvas[y_start:y_end, x_start:x_end] = img_array
            return True
        except Exception as e:
            print(f"Error placing image {img_path}: {e}")
            return False
    
    # Place all images on the canvas
    successful_adds = 0
    for x, y, img_path in coordinates:
        if place_image(canvas, x, y, img_path):
            successful_adds += 1
    
    print(f"Successfully added {successful_adds} out of {len(coordinates)} images.")
    
    if successful_adds == 0:
        print("No images were added. Check the coordinate mapping or canvas size.")
        return
    
    # Create figure without any axes or labels
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display the composite image
    ax.imshow(canvas)
    
    # Remove all plot decorations
    ax.axis('off')  # Hide axes, ticks, labels, etc.
    
    # Remove padding and borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(f"Plot saved to {output_path}")
    
    # Close the figure to avoid displaying it
    plt.close(fig)

# Example usage in Cursor
if __name__ == "__main__":
    folder_path = "/Users/alansaj/Downloads/musemicroscopy-main/new_images"  # Adjust to your local path
    output_path = "image_grid_good.png"  # Save to the current directory
    plot_images_on_grid(folder_path, output_path)
