import os
import re
import numpy as np
import cv2
from collections import defaultdict

def stitch_original_images(folder_path, file_name=None, input_overlap_percent=50):
    # Regex to parse X and Y coords from filenames
    pattern = r'X(\d+\.\d+)_Y(\d+\.\d+)'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    extraction_percent = (1-input_overlap_percent/100)*1.2
    
    # First pass: collect all coordinates to build grid
    raw_coords = []
    for img_file in image_files:
        match = re.search(pattern, img_file)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            raw_coords.append((x, y, img_file))
    
    if not raw_coords:
        print("No valid coordinates found in filenames.")
        return
    
    print(f"\nFound {len(raw_coords)} images with coordinates")
    
    # Function to cluster coordinates that are close together
    def cluster_coordinates(coords, tolerance=1e-6):
        """Cluster coordinates that are very close to each other"""
        if not coords:
            return []
        
        # Sort the coordinates
        sorted_coords = sorted(coords)
        
        # Initialize clusters with the first coordinate
        clusters = [[sorted_coords[0]]]
        
        # Cluster remaining coordinates
        for coord in sorted_coords[1:]:
            if abs(coord - clusters[-1][0]) <= tolerance:
                # Add to the current cluster if within tolerance
                clusters[-1].append(coord)
            else:
                # Start a new cluster
                clusters.append([coord])
        
        # Return the mean of each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    # Extract and cluster X and Y coordinates
    raw_x_coords = [c[0] for c in raw_coords]
    raw_y_coords = [c[1] for c in raw_coords]
    
    # Determine appropriate tolerance based on the precision in the coordinates
    def estimate_tolerance(coords):
        """Estimate appropriate tolerance from the data"""
        if len(coords) < 2:
            return 1e-6
        
        # Find the smallest non-zero difference between coordinates
        diffs = []
        sorted_coords = sorted(set(coords))
        for i in range(len(sorted_coords) - 1):
            diff = sorted_coords[i+1] - sorted_coords[i]
            if diff > 0:
                diffs.append(diff)
        
        if not diffs:
            return 1e-6
        
        # Use 1% of the minimum difference as tolerance
        min_diff = min(diffs)
        return min_diff * 0.01
    
    x_tolerance = estimate_tolerance(raw_x_coords)
    y_tolerance = estimate_tolerance(raw_y_coords)
    
    print(f"Estimated coordinate tolerances: X={x_tolerance:.8f}, Y={y_tolerance:.8f}")
    
    # Cluster coordinates that are very close
    clustered_x = cluster_coordinates(raw_x_coords, x_tolerance)
    clustered_y = cluster_coordinates(raw_y_coords, y_tolerance)
    
    # Sort the clustered coordinates
    unique_x_coords = sorted(clustered_x)
    unique_y_coords = sorted(clustered_y)
    
    print(f"After clustering: {len(unique_x_coords)} unique X values, {len(unique_y_coords)} unique Y values")
    print(f"X values: {unique_x_coords}")
    print(f"Y values: {unique_y_coords}")
    
    # Calculate median spacing to identify grid pattern
    def calculate_grid_spacing(coords):
        if len(coords) < 2:
            return None
        diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
        if not diffs:
            return None
        return np.median(diffs)
    
    x_spacing = calculate_grid_spacing(unique_x_coords)
    y_spacing = calculate_grid_spacing(unique_y_coords)
    
    if x_spacing is None or y_spacing is None:
        print("Not enough coordinates to determine grid spacing")
        return
    
    print(f"Detected grid spacing: X={x_spacing:.4f}, Y={y_spacing:.4f}")
    
    # Extrapolate full grid with potential missing rows/columns
    def extrapolate_grid(coords, spacing, tolerance):
        if not coords:
            return []
        
        # Determine expected range
        min_val = min(coords)
        max_val = max(coords)
        
        # Generate a regular grid within the range
        n_points = round((max_val - min_val) / spacing) + 1
        
        # Create regular grid points
        full_grid = []
        for i in range(int(n_points)):
            grid_point = min_val + i * spacing
            
            # Check if this point matches any actual coordinate (within tolerance)
            matches_actual = any(abs(grid_point - coord) <= tolerance for coord in coords)
            
            # If it matches, use the actual coordinate instead of the calculated one
            if matches_actual:
                # Find the closest actual coordinate
                closest = min(coords, key=lambda c: abs(c - grid_point))
                full_grid.append(closest)
            else:
                full_grid.append(grid_point)
        
        # Check for missing points at beginning/end using tolerance-based comparison
        # Only add points that are significantly distant from the min/max
        if min(coords) - min_val > spacing - tolerance:
            i = 1
            while min_val - i * spacing >= min(coords) - spacing * 3:
                full_grid.insert(0, min_val - i * spacing)
                i += 1
        
        if max_val - max(coords) > spacing - tolerance:
            i = 1
            while max_val + i * spacing <= max(coords) + spacing * 3:
                full_grid.append(max_val + i * spacing)
                i += 1
        
        return sorted(full_grid)
    
    # Create complete grid coordinates with careful extrapolation
    complete_x_coords = extrapolate_grid(unique_x_coords, x_spacing, x_tolerance)
    complete_y_coords = extrapolate_grid(unique_y_coords, y_spacing, y_tolerance)
    
    print(f"Extrapolated grid coordinates:")
    print(f"X={complete_x_coords}")
    print(f"Y={complete_y_coords}")
    
    # Map existing coordinates to the closest grid coordinates
    def map_to_grid(actual_coord, grid_coords, tolerance):
        """Map an actual coordinate to the closest grid coordinate"""
        for grid_coord in grid_coords:
            if abs(actual_coord - grid_coord) <= tolerance:
                return grid_coord
        return None
    
    # Map the raw coordinates to grid positions
    coord_mapping = {}
    for x, y, img_file in raw_coords:
        grid_x = map_to_grid(x, complete_x_coords, x_tolerance)
        grid_y = map_to_grid(y, complete_y_coords, y_tolerance)
        if grid_x is not None and grid_y is not None:
            coord_mapping[(x, y)] = (grid_x, grid_y, img_file)
    
    # Map actual coordinates to grid positions (row and column indices)
    x_to_col = {x: i for i, x in enumerate(complete_x_coords)}
    y_to_row = {y: i for i, y in enumerate(complete_y_coords)}
    
    # For reverse lookup
    col_to_x = {i: x for x, i in x_to_col.items()}
    row_to_y = {i: y for y, i in y_to_row.items()}
    
    grid_width = len(complete_x_coords)
    grid_height = len(complete_y_coords)
    
    print(f"\nComplete grid: {grid_width}×{grid_height}")
    
    # Second pass: collect images and their coordinates
    images = {}
    for x, y, img_file in raw_coords:
        # Get the normalized grid coordinates
        grid_x, grid_y, _ = coord_mapping.get((x, y), (None, None, None))
        if grid_x is None or grid_y is None:
            continue
            
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)  # Load in color (BGR)
        if img is not None:
            # Extract center portion of the image
            height, width = img.shape[:2]
            center_x, center_y = width // 2, height // 2
            crop_width = int(width * extraction_percent)
            crop_height = int(height * extraction_percent)
            start_x = center_x - crop_width // 2
            start_y = center_y - crop_height // 2
            cropped_img = img[start_y:start_y+crop_height, start_x:start_x+crop_width]
            
            # Store with grid position
            col = x_to_col[grid_x]
            row = y_to_row[grid_y]
            images[(col, row)] = (cropped_img, img_file, x, y)
    
    if not images:
        print("No valid images found.")
        return
    
    print(f"\nSuccessfully loaded {len(images)} images out of {grid_width * grid_height} grid positions")
    
    # Get cropped image dimensions (assuming all are the same size)
    img_width, img_height = next(iter(images.values()))[0].shape[1], next(iter(images.values()))[0].shape[0]
    
    # Set consistent overlap values
    overlap_percent = 0.2  # 20% overlap
    overlap_x = int(img_width * overlap_percent)
    overlap_y = int(img_height * overlap_percent)
    
    # Calculate effective tile sizes (accounting for overlap)
    effective_width = int(img_width * (1 - overlap_percent))
    effective_height = int(img_height * (1 - overlap_percent))
    
    # Calculate canvas size based on grid
    canvas_width = effective_width * (grid_width - 1) + img_width
    canvas_height = effective_height * (grid_height - 1) + img_height
    
    # Create canvas and weight map for blending
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    print(f"\nCanvas size: {canvas_width}x{canvas_height} px")
    print(f"Grid dimensions: {grid_width}×{grid_height}")
    print(f"Overlap: {overlap_percent*100}% ({overlap_x} px horizontal, {overlap_y} px vertical)")
    
    # Create weight masks for blending
    mask_x = np.linspace(1, 0, overlap_x, dtype=np.float32)
    mask_y = np.linspace(1, 0, overlap_y, dtype=np.float32)
    
    # Place and blend each available image on the canvas
    missing_positions = []
    
    # Create a grid visualization (invert y axis for display)
    grid_vis = [['.' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Track missing rows and columns
    rows_with_images = set()
    cols_with_images = set()
    
    # Process grid in row-major order with y-axis flipped (bottom to top)
    for row in range(grid_height):
        for col in range(grid_width):
            # Calculate pixel position based on grid coordinates
            px = col * effective_width
            py = (grid_height - 1 - row) * effective_height  # Flip Y axis (bottom row = 0 in original coords)
            
            if (col, row) in images:
                img, fname, orig_x, orig_y = images[(col, row)]
                # Track that this row and column have at least one image
                rows_with_images.add(row)
                cols_with_images.add(col)
                
                # Convert image to float for blending
                img_float = img.astype(np.float32)
                
                # Create a weight mask for this tile
                tile_weight = np.ones((img_height, img_width), dtype=np.float32)
                
                # Apply X-direction blending
                if col > 0:  # Left overlap with previous tile
                    tile_weight[:, :overlap_x] *= mask_x[::-1]
                if col < grid_width - 1:  # Right overlap with next tile
                    tile_weight[:, -overlap_x:] *= mask_x
                
                # Apply Y-direction blending
                if row < grid_height - 1:  # Bottom overlap
                    tile_weight[-overlap_y:, :] *= mask_y[:, np.newaxis]
                if row > 0:  # Top overlap
                    tile_weight[:overlap_y, :] *= mask_y[::-1][:, np.newaxis]
                
                # Add weighted image to canvas
                try:
                    canvas[py:py+img_height, px:px+img_width] += img_float * tile_weight[:, :, np.newaxis]
                    weight_map[py:py+img_height, px:px+img_width] += tile_weight
                    
                    # Use the inverted row for display (to match normal top-to-bottom reading)
                    display_row = grid_height - 1 - row
                    print(f"Placed {fname} at grid position ({col}, {row}), canvas pos ({px}, {py}), original coords X{orig_x:.2f}, Y{orig_y:.2f}")
                    grid_vis[display_row][col] = 'X'  # Mark as placed
                except ValueError as e:
                    print(f"Error placing {fname}: {e}. Likely due to canvas size mismatch.")
                    print(f"Canvas shape: {canvas.shape}, Placement target: [{py}:{py+img_height}, {px}:{px+img_width}]")
                    return
            else:
                # Find the original coordinates for this grid position
                orig_x = col_to_x.get(col)
                orig_y = row_to_y.get(row)
                
                if orig_x is not None and orig_y is not None:
                    missing_positions.append((col, row, orig_x, orig_y))
                    print(f"Missing image at grid position ({col}, {row}), canvas pos ({px}, {py}), original coords X{orig_x:.2f}, Y{orig_y:.2f}")
                else:
                    print(f"Missing image at grid position ({col}, {row}), canvas pos ({px}, {py})")
    
    # Identify missing entire rows and columns
    missing_rows = set(range(grid_height)) - rows_with_images
    missing_cols = set(range(grid_width)) - cols_with_images
    
    if missing_rows:
        print(f"\nMissing entire rows: {sorted(missing_rows)}")
        print(f"Missing Y coordinates: {[row_to_y[r] for r in missing_rows]}")
    
    if missing_cols:
        print(f"\nMissing entire columns: {sorted(missing_cols)}")
        print(f"Missing X coordinates: {[col_to_x[c] for c in missing_cols]}")
    
    # Print grid visualization (flipped to match usual top-down orientation)
    print("\nGrid visualization (X = image placed, . = missing):")
    for i, row in enumerate(grid_vis):
        # Mark rows with no images
        row_marker = " " if grid_height - 1 - i in rows_with_images else ">"
        print(f"{row_marker} {' '.join(row)}")
    
    # Print column markers for missing columns
    col_markers = ['^' if col in cols_with_images else '|' for col in range(grid_width)]
    print(f"  {' '.join(col_markers)}")
    
    if missing_positions:
        print(f"\nMissing {len(missing_positions)} individual positions (excluding entire rows/columns):")
        for col, row, x, y in sorted(missing_positions):
            if row in missing_rows or col in missing_cols:
                continue  # Skip positions in entirely missing rows/columns
            print(f"  - Grid ({col}, {row}), Original coords X{x:.2f}, Y{y:.2f}")
    
    # Normalize by weight map to avoid over-brightening
    mask = weight_map > 0
    canvas[mask] /= weight_map[mask, np.newaxis]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    
    # Save the stitched image
    output_path = os.path.join(folder_path, file_name)
    cv2.imwrite(output_path, canvas)
    print(f"\nStitched image saved to: {output_path}")
    
    if np.all(canvas == 0):
        print("Warning: Stitched image is completely black!")

# Main execution
if __name__ == "__main__":
    folder_path = "C:/Users/jrose17/Downloads/AutoCapture_20250416_162915"  # Original images
    print("Stitching center 60% of images with 20% overlap and blending...")
    stitch_original_images(folder_path, file_name="stitched_final_clean.png", input_overlap_percent=50)
