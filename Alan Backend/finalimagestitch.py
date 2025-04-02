import os
import re
import numpy as np
import cv2
from collections import Counter

def load_stitched_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print(f"Failed to load {file_path}")
        return None
    return img

def detect_duplicates_in_stitched(stitched_path="stitchdetectedindividual.png", min_area=3000):
    stitched_img = load_stitched_image(stitched_path)
    if stitched_img is None:
        return None, None
    
    # Preprocessing (unchanged)
    stitched_gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(stitched_gray)
    img_blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    _, thresh_dark = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY_INV)
    _, thresh_light = cv2.threshold(img_blurred, 200, 255, cv2.THRESH_BINARY)
    thresh_combined = cv2.bitwise_or(thresh_dark, thresh_light)
    kernel = np.ones((5, 5), np.uint8)
    thresh_dilated = cv2.dilate(thresh_combined, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    print(f"Detected {len(valid_contours)} features in stitched image")
    
    # Tile size and grid from original color images
    pattern = r'X(\d+\.\d+)_Y(\d+\.\d+)'
    folder_path = "/Users/alansaj/musemicroscopy/new_images"
    image_files = [f for f in os.listdir(folder_path) if re.search(pattern, f)]
    if not image_files:
        print(f"No images found in {folder_path} to determine size")
        return None, None
    first_img = cv2.imread(os.path.join(folder_path, image_files[0]))
    if first_img is None:
        print(f"Failed to load first tile for size: {image_files[0]}")
        return None, None
    img_width, img_height = first_img.shape[1], first_img.shape[0]
    print(f"Tile size: {img_width}x{img_height} px")
    
    x_coords = sorted(set(float(re.search(pattern, f).group(1)) for f in image_files))
    y_coords = sorted(set(float(re.search(pattern, f).group(2)) for f in image_files), reverse=True)
    grid_width = len(x_coords)
    grid_height = len(y_coords)
    print(f"Grid size: {grid_width}x{grid_height}")
    print(f"X coords: {x_coords}")
    print(f"Y coords: {y_coords}")
    
    # Contour extraction (unchanged)
    contour_origins = []
    edge_threshold = 0.25
    for cnt in valid_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            grid_x = cx // img_width
            grid_y = cy // img_height
            if grid_x < grid_width and grid_y < grid_height:
                offset_x = cx % img_width
                near_edge_x = offset_x < img_width * edge_threshold or offset_x > img_width * (1 - edge_threshold)
                if near_edge_x:
                    contour_origins.append((cnt, (grid_x, grid_y), (cx, cy)))
                    expected_x = grid_x * img_width + offset_x
                    expected_y = grid_y * img_height
                    print(f"Contour at {grid_x},{grid_y}: centroid=({cx}, {cy}), expected_x={expected_x:.0f}, expected_y_base={expected_y:.0f}")
    
    # Pairing (unchanged)
    initial_pairs = []
    paired = set()
    for i, (c1, origin1, pos1) in enumerate(contour_origins):
        if i in paired:
            continue
        best_match = None
        best_score = -1
        best_shape_similarity = -1
        best_pos = None
        best_origin = None
        best_j = None
        
        for j, (c2, origin2, pos2) in enumerate(contour_origins):
            if j == i or j in paired:
                continue
            dx = origin2[0] - origin1[0]
            dy = origin2[1] - origin1[1]
            if abs(dx) == 1 and dy == 0:
                pixel_dx = pos2[0] - pos1[0]
                pixel_dy = pos2[1] - pos1[1]
                distance = np.sqrt(pixel_dx**2 + pixel_dy**2)
                min_x_dist = 400
                max_x_dist = 1600
                max_y_drift = 50
                if min_x_dist <= distance <= max_x_dist and abs(pixel_dx) >= min_x_dist:
                    if abs(pixel_dy) > max_y_drift:
                        print(f"Rejected (Y drift too large): distance={distance:.1f}px, dx={pixel_dx}, dy={pixel_dy}, origins={origin1} vs {origin2}, pos1={pos1}, pos2={pos2}")
                        continue
                    moments1 = cv2.HuMoments(cv2.moments(c1)).flatten()
                    moments2 = cv2.HuMoments(cv2.moments(c2)).flatten()
                    diff = np.abs(moments1 - moments2)
                    similarity = 1 / (1 + np.sum(diff))
                    area1 = cv2.contourArea(c1)
                    area2 = cv2.contourArea(c2)
                    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
                    shape_similarity = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
                    score = similarity * area_ratio * (1 - shape_similarity)
                    if score > 0.5 and similarity > 0.65 and area_ratio > 0.65:
                        if score > best_score or (score == best_score and shape_similarity < best_shape_similarity):
                            best_match = c2
                            best_score = score
                            best_shape_similarity = shape_similarity
                            best_pos = pos2
                            best_origin = origin2
                            best_j = j
    
        if best_match is not None:
            distance = np.sqrt((pos1[0] - best_pos[0])**2 + (pos1[1] - best_pos[1])**2)
            initial_pairs.append((c1, best_match, pos1, best_pos, origin1, best_origin, distance, best_score, best_shape_similarity))
            paired.add(i)
            paired.add(best_j)
            print(f"Initial duplicate: score={best_score:.3f}, shape_similarity={best_shape_similarity:.3f}, origins={origin1} vs {best_origin}, distance={distance:.1f}px, dx={best_pos[0] - pos1[0]}, dy={best_pos[1] - pos1[1]}, pos1={pos1}, pos2={best_pos}")
    
    print(f"Found {len(initial_pairs)} initial duplicate pairs")
    
    # Mode filtering (unchanged)
    if initial_pairs:
        distances = [pair[6] for pair in initial_pairs]
        bin_size = 50
        binned_distances = [int(d // bin_size * bin_size) for d in distances]
        mode_distance = Counter(binned_distances).most_common(1)[0][0]
        tolerance = 150
        min_valid_dist = mode_distance - tolerance
        max_valid_dist = mode_distance + tolerance
        print(f"Mode distance: {mode_distance} px, valid range: {min_valid_dist}-{max_valid_dist} px")
        
        duplicate_pairs = []
        for c1, c2, pos1, pos2, origin1, origin2, distance, score, shape_similarity in initial_pairs:
            if min_valid_dist <= distance <= max_valid_dist:
                duplicate_pairs.append((c1, c2, pos1, pos2, origin1, origin2))
                print(f"Valid duplicate: score={score:.3f}, shape_similarity={shape_similarity:.3f}, origins={origin1} vs {origin2}, distance={distance:.1f}px, dx={pos2[0] - pos1[0]}, dy={pos2[1] - pos1[1]}, pos1={pos1}, pos2={pos2}")
    
    print(f"Found {len(duplicate_pairs)} final unique duplicate pairs")
    
    # Align duplicates: Calculate shifts to match centroids
    shifts = {}
    for _, _, pos1, pos2, origin1, origin2 in duplicate_pairs:
        grid_x1, grid_y1 = origin1
        grid_x2, grid_y2 = origin2
        dx = pos2[0] - pos1[0]  # X distance between centroids
        
        # Determine which tile to shift (right tile moves left)
        if grid_x2 > grid_x1:
            tile_to_shift = (grid_x2, grid_y2)
            reference_tile = (grid_x1, grid_y1)
            shift_amount = dx  # Positive dx means shift left by that amount
        else:
            tile_to_shift = (grid_x1, grid_y1)
            reference_tile = (grid_x2, grid_y2)
            shift_amount = -dx  # Negative dx means shift left by that amount
        
        # Store shift for tiles with duplicates
        shifts[tile_to_shift] = shifts.get(tile_to_shift, 0) + shift_amount
        print(f"Shift tile {tile_to_shift} left by {shift_amount:.1f} px relative to {reference_tile} to align centroids")

    # Calculate average shift from duplicate pairs
    if shifts:
        avg_shift = np.mean(list(shifts.values()))
        print(f"Average shift from duplicate pairs: {avg_shift:.1f} px")
    else:
        avg_shift = 0
        print("No duplicate pairs found; using average shift of 0 px")

    # Apply shifts: Use duplicate shifts where available, avg_shift for others
    adjusted_shifts = {}
    for row in range(grid_height):
        cumulative_shift = 0
        for col in range(grid_width):
            tile = (col, row)
            if tile in shifts:
                cumulative_shift += shifts[tile]  # Use specific shift for duplicate tiles
            else:
                cumulative_shift += avg_shift  # Apply avg_shift to non-duplicate tiles
            adjusted_shifts[tile] = cumulative_shift
            if cumulative_shift != 0:
                print(f"Tile {tile} adjusted shift: {cumulative_shift:.1f} px (duplicate shift: {shifts.get(tile, 0):.1f}, avg shift applied: {avg_shift if tile not in shifts else 0:.1f})")

    # Canvas setup: Adjust width to account for shifts
    max_shift = max(abs(min(adjusted_shifts.values(), default=0)), abs(max(adjusted_shifts.values(), default=0)))
    canvas_width = int(grid_width * img_width - max_shift)  # Reduce width by max shift
    canvas_height = int(grid_height * img_height)
    aligned_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    print(f"Canvas size: {canvas_width}x{canvas_height} px")
    
    # Place tiles with shifts using original color images
    for file in image_files:
        match = re.search(pattern, file)
        grid_x = x_coords.index(float(match.group(1)))
        grid_y = y_coords.index(float(match.group(2)))
        tile_path = os.path.join(folder_path, file)
        tile = cv2.imread(tile_path)
        if tile is None:
            print(f"Failed to load tile {tile_path}")
            continue
        
        base_x = grid_x * img_width
        base_y = grid_y * img_height
        shift_x = adjusted_shifts.get((grid_x, grid_y), 0)
        new_x = int(base_x - shift_x)  # Shift left (negative shift_x reduces x position)
        
        print(f"Placing tile {grid_x},{grid_y}: base_x={base_x}, base_y={base_y}, shift_x={shift_x:.1f}, new_x={new_x}, tile_shape={tile.shape}")
        
        if new_x < 0:
            tile = tile[:, -new_x:]  # Crop left side if shifted off-canvas
            new_x = 0
        if new_x + tile.shape[1] > canvas_width:
            tile = tile[:, :canvas_width - new_x]  # Crop right side
        if base_y + tile.shape[0] > canvas_height:
            tile = tile[:canvas_height - base_y, :]  # Crop bottom
        
        try:
            aligned_img[base_y:base_y + tile.shape[0], new_x:new_x + tile.shape[1]] = tile
            print(f"Shifted tile {grid_x},{grid_y} placed at ({new_x}, {base_y})")
        except ValueError as e:
            print(f"Error placing shifted tile {grid_x},{grid_y}: {e}")
    
    cv2.imwrite("stitched_aligned_clean.png", aligned_img)
    print(f"Clean aligned stitched image saved to stitched_aligned_clean.png")
    if np.all(aligned_img == 0):
        print("Warning: Aligned image is completely black!")
    
    return aligned_img, duplicate_pairs

if __name__ == "__main__":
    print("Step 3: Detecting and aligning X-direction duplicates...")
    aligned_img, duplicate_pairs = detect_duplicates_in_stitched()
