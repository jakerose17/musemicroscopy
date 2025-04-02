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

def detect_duplicates_in_stitched(stitched_path="stitchdetectedindividual.png", output_path="stitched_with_duplicates.png", min_area=3000):
    stitched_img = load_stitched_image(stitched_path)
    if stitched_img is None:
        return None, None
    
    # Preprocessing
    stitched_gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(stitched_gray)
    img_blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)  # Softer blur
    _, thresh_dark = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY_INV)
    _, thresh_light = cv2.threshold(img_blurred, 200, 255, cv2.THRESH_BINARY)
    thresh_combined = cv2.bitwise_or(thresh_dark, thresh_light)
    kernel = np.ones((5, 5), np.uint8)  # Smaller kernel
    thresh_dilated = cv2.dilate(thresh_combined, kernel, iterations=1)  # 1 iteration
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    print(f"Detected {len(valid_contours)} features in stitched image")
    
    # Save intermediate for debugging
    cv2.imwrite("thresh_debug.png", thresh_dilated)
    print("Saved thresholded image to thresh_debug.png for debugging")
    
    # Tile size
    pattern = r'X(\d+\.\d+)_Y(\d+\.\d+)'
    folder_path = "outlined_features"
    image_files = [f for f in os.listdir(folder_path) if re.search(pattern, f)]
    if not image_files:
        print("No images found in output_features to determine size")
        return None, None
    first_img = cv2.imread(os.path.join(folder_path, image_files[0]))
    img_width, img_height = first_img.shape[1], first_img.shape[0]
    print(f"Tile size: {img_width}x{img_height} px")
    min_x_dist = 400
    max_x_dist = 1600
    max_y_drift = 50
    print(f"Initial X-direction distance range: {min_x_dist}-{max_x_dist} px, max Y drift: {max_y_drift} px")
    
    # Grid setup
    x_coords = sorted(set(float(re.search(pattern, f).group(1)) for f in image_files))
    y_coords = sorted(set(float(re.search(pattern, f).group(2)) for f in image_files), reverse=True)
    grid_width = len(x_coords)
    grid_height = len(y_coords)
    
    # Contour extraction
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
    
    # Initial pairing
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
                    else:
                        print(f"Rejected (low score): similarity={similarity:.3f}, area_ratio={area_ratio:.3f}, shape_similarity={shape_similarity:.3f}, score={score:.3f}, origins={origin1} vs {origin2}, distance={distance:.1f}px, pos1={pos1}, pos2={pos2}")
                else:
                    reason = "distance" if not (min_x_dist <= distance <= max_x_dist) else "dx"
                    print(f"Rejected ({reason}): distance={distance:.1f}px, dx={pixel_dx}, dy={pixel_dy}, origins={origin1} vs {origin2}, pos1={pos1}, pos2={pos2}")
        
        if best_match is not None:
            distance = np.sqrt((pos1[0] - best_pos[0])**2 + (pos1[1] - best_pos[1])**2)
            initial_pairs.append((c1, best_match, pos1, best_pos, origin1, best_origin, distance, best_score, best_shape_similarity))
            paired.add(i)
            paired.add(best_j)
            print(f"Initial duplicate: score={best_score:.3f}, shape_similarity={best_shape_similarity:.3f}, origins={origin1} vs {best_origin}, distance={distance:.1f}px, dx={best_pos[0] - pos1[0]}, dy={best_pos[1] - pos1[1]}, pos1={pos1}, pos2={best_pos}")
    
    print(f"Found {len(initial_pairs)} initial duplicate pairs")
    
    # Mode distance filtering
    if initial_pairs:
        distances = [pair[6] for pair in initial_pairs]
        bin_size = 50
        binned_distances = [int(d // bin_size * bin_size) for d in distances]
        mode_distance = Counter(binned_distances).most_common(1)[0][0]
        tolerance = 150
        min_valid_dist = mode_distance - tolerance
        max_valid_dist = mode_distance + tolerance
        print(f"Mode distance: {mode_distance} px, valid range: {min_valid_dist}-{max_valid_dist} px")
        
        # Filter pairs
        duplicate_pairs = []
        for c1, c2, pos1, pos2, origin1, origin2, distance, score, shape_similarity in initial_pairs:
            if min_valid_dist <= distance <= max_valid_dist:
                duplicate_pairs.append((c1, c2, pos1, pos2))
                print(f"Valid duplicate: score={score:.3f}, shape_similarity={shape_similarity:.3f}, origins={origin1} vs {origin2}, distance={distance:.1f}px, dx={pos2[0] - pos1[0]}, dy={pos2[1] - pos1[1]}, pos1={pos1}, pos2={pos2}")
            else:
                print(f"Invalid (distance outlier): score={score:.3f}, shape_similarity={shape_similarity:.3f}, origins={origin1} vs {origin2}, distance={distance:.1f}px, dx={pos2[0] - pos1[0]}, dy={pos2[1] - pos1[1]}, pos1={pos1}, pos2={pos2}")
    else:
        duplicate_pairs = []
        print("No initial pairs to calculate mode distance")
    
    print(f"Found {len(duplicate_pairs)} final unique duplicate pairs")
    
    # Visualize
    output_img = stitched_img.copy()
    for c1, c2, pos1, pos2 in duplicate_pairs:
        cv2.fillPoly(output_img, [c1], (0, 0, 255))  # Red fill
        cv2.fillPoly(output_img, [c2], (0, 0, 255))
        cv2.line(output_img, pos1, pos2, (255, 0, 0), 2)  # Blue line
        cv2.circle(output_img, pos1, 5, (0, 255, 255), -1)  # Yellow dot
        cv2.circle(output_img, pos2, 5, (0, 255, 255), -1)
    
    cv2.imwrite(output_path, output_img)
    print(f"Stitched image with valid duplicates (red fills, blue lines, yellow centroids) saved to {output_path}")
    
    return output_img, duplicate_pairs

if __name__ == "__main__":
    print("Step 3: Detecting X-direction duplicates...")
    stitched_with_duplicates, duplicate_pairs = detect_duplicates_in_stitched()
