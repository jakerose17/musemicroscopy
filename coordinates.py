import os
import re
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

def plot_images_on_grid(folder_path, output_path=None, figsize=(15, 15), dpi=300):
    """
    Plot images on a grid based on X and Y coordinates extracted from filenames.
    Without coordinate labels and with blended borders between images.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the images
    output_path : str, optional
        Path to save the output plot. If None, the plot is displayed
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the figure
    
    Returns:
    --------
    None
    """
    # Regular expression to extract X and Y coordinates from filenames
    # Pattern matches: xy_X19.800_Y22.100_Z8.293.jpg
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
    
    # Determine grid bounds
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Reduce padding to better focus on the precise coordinates
    x_padding = (x_max - x_min) * 0.02
    y_padding = (y_max - y_min) * 0.02
    
    # Create figure with higher DPI for better precision
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Set up the plot with more precise formatting for axes
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Images Plotted by X and Y Coordinates')
    
    # Use a finer grid for better precision visualization
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    ax.minorticks_on()
    
    # Format axis ticks to show appropriate decimal places
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Determine appropriate decimal precision based on data range
    x_decimals = max(3, int(abs(np.floor(np.log10(x_range / 10)))))
    y_decimals = max(3, int(abs(np.floor(np.log10(y_range / 10)))))
    
    # Format tick labels with appropriate precision
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(f'%.{x_decimals}f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(f'%.{y_decimals}f'))
    
    # Function to add image to plot with blended borders
    def add_image(x, y, img_path, ax, zoom=0.1):
        try:
            # Read the image
            img = Image.open(img_path)
            
            # Apply slight gaussian blur to edges for blending (if PIL is available)
            try:
                # Create a mask for the edges (more transparent at edges)
                width, height = img.size
                mask = Image.new('L', (width, height), 255)  # 'L' mode = 8-bit pixels, black and white
                
                # Define blur width (in pixels)
                blur_width = 5
                
                # Make edges more transparent
                for i in range(blur_width):
                    # Transparency increases as we get closer to the edge
                    alpha = int(255 * (i + 1) / (blur_width + 1))
                    
                    # Top edge
                    for j in range(width):
                        mask.putpixel((j, i), alpha)
                    
                    # Bottom edge
                    for j in range(width):
                        mask.putpixel((j, height - 1 - i), alpha)
                    
                    # Left edge
                    for j in range(height):
                        mask.putpixel((i, j), alpha)
                    
                    # Right edge
                    for j in range(height):
                        mask.putpixel((width - 1 - i, j), alpha)
                
                # Convert to RGBA if not already
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Apply the mask to the alpha channel
                r, g, b, a = img.split()
                a = mask
                img = Image.merge('RGBA', (r, g, b, a))
                
                # Convert to numpy array for matplotlib
                img_array = np.array(img)
            except Exception as e:
                print(f"Error applying edge blending: {e}, using original image")
                img_array = np.array(img)
            
            # Add the image to the plot
            im = OffsetImage(img_array, zoom=zoom)
            ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
            ax.add_artist(ab)
            return True
        except Exception as e:
            print(f"Error adding image {img_path}: {e}")
            return False
    
    # Add all images
    successful_adds = 0
    for x, y, img_path in coordinates:
        if add_image(x, y, img_path, ax):
            successful_adds += 1
            # Removed the coordinate annotation code
    
    print(f"Successfully added {successful_adds} out of {len(coordinates)} images.")
    
    # Add a tight layout to maximize visualization area
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Replace with your actual folder path
    folder_path = "/Users/alansaj/Downloads/musemicroscopy-main/new_images"
    
    # Optional: specify output path to save the plot
    output_path = "image_grid_blended.png"
    
    plot_images_on_grid(folder_path, output_path)
