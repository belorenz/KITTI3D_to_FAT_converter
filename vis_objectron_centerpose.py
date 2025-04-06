import cv2
import numpy as np # Added numpy for matrix/vector math
import json
import os
from scipy.spatial.transform import Rotation as R # For quaternion to Euler/Matrix conversion

# --- Configuration ---
DATA_DIR = '/home/lore_be/data/shoe_batch-1_3/'
DATA_DIR = '/home/lore_be/data/deleteme2/'
IMAGE_EXTENSION = '.png'
JSON_EXTENSION = '.json'
WINDOW_NAME = '6D Pose Ground Truth Inspector (-Z fwd, X right, Y up)'
# Text display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 255, 255)  # White for general info
LINE_TYPE = 1
TEXT_START_Y = 20
TEXT_LINE_HEIGHT = 18
OBJECT_INFO_OFFSET_Y = 10
# Projected point indices settings
INDEX_FONT_SCALE = 0.4
INDEX_FONT_COLOR = (0, 255, 255) # Yellow
INDEX_OFFSET_X = 5
INDEX_OFFSET_Y = -5
POINT_CIRCLE_RADIUS = 3
POINT_CIRCLE_COLOR = (0, 0, 255) # Red circles for points
# Coordinate Axes Visualization Settings
AXIS_BASE_LENGTH = 0.1  # Increased from 0.05 for longer axes (adjust further if needed)
AXIS_LINE_THICKNESS = 2
AXIS_COLOR_X = (0, 0, 255)  # Red (BGR for OpenCV)
AXIS_COLOR_Y = (0, 255, 0)  # Green
AXIS_COLOR_Z = (255, 0, 0)  # Blue
# New settings for Axis Labels
AXIS_LABEL_FONT_SCALE = 0.7 # Larger font scale for labels
AXIS_LABEL_LINE_TYPE = 2   # Thicker font line type
AXIS_LABEL_OFFSET_X = 6   # Pixel offset for label from axis endpoint
AXIS_LABEL_OFFSET_Y = 6   # Pixel offset for label from axis endpoint

# --- Coordinate System Assumption ---
# -Z : Forward (out of the camera/screen)
# +X : Right
# +Y : Up
# ------------------------------------

# --- (find_data_pairs, load_data, quaternion_to_euler_degrees, format_vector functions remain the same) ---
def find_data_pairs(directory, img_ext, json_ext):
    """Finds pairs of image and json files with matching base names."""
    image_files = sorted([f for f in os.listdir(directory) if f.endswith(img_ext)])
    base_names = [os.path.splitext(f)[0] for f in image_files]

    file_pairs = []
    for base in base_names:
        img_path = os.path.join(directory, base + img_ext)
        json_path = os.path.join(directory, base + json_ext)
        if os.path.exists(json_path):
            file_pairs.append({
                "base_name": base,
                "image_path": img_path,
                "json_path": json_path
            })
        else:
            print(f"Warning: JSON file not found for image {img_path}")
    return file_pairs

def load_data(pair):
    """Loads image and JSON data for a given pair."""
    try:
        image = cv2.imread(pair["image_path"])
        if image is None:
            print(f"Error: Could not load image {pair['image_path']}")
            return None, None

        with open(pair["json_path"], 'r') as f:
            data = json.load(f)
        return image, data
    except FileNotFoundError:
        print(f"Error: File not found for pair {pair['base_name']}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file {pair['json_path']}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading data for {pair['base_name']}: {e}")
        return None, None

def quaternion_to_euler_degrees(quat_xyzw):
    """
    Converts a quaternion [x, y, z, w] to Euler angles (intrinsic XYZ order) in degrees.
    """
    if quat_xyzw is None or len(quat_xyzw) != 4:
        return "Invalid Quat"
    try:
        r = R.from_quat(quat_xyzw)
        euler_deg = r.as_euler('xyz', degrees=True)
        return euler_deg
    except Exception as e:
        print(f"Error converting quaternion {quat_xyzw}: {e}")
        return "Conv Error"

def format_vector(vec, precision=2):
    """Formats a list/tuple of numbers into a string."""
    if vec is None:
        return "N/A"
    try:
        return f"[{', '.join([f'{v:.{precision}f}' for v in vec])}]"
    except (TypeError, ValueError):
        return str(vec)

# --- Helper function for 3D to 2D Projection ---
def project_point(point_3d_cam, fx, fy, cx, cy):
    """
    Projects a 3D point in camera coordinates (-Z forward) onto the 2D image plane.
    Returns (u, v) pixel coordinates or None if behind the camera.
    Assumes standard pixel coordinates (+Y down from top-left).
    """
    x_cam, y_cam, z_cam = point_3d_cam

    # Points must be in front of the camera in the (-Z forward) system, i.e., Z < 0
    if z_cam >= 0:
        return None

    # Convert to standard CV (+Z forward) for projection formula
    # Z_cv = -z_cam
    # x_proj = x_cam / Z_cv
    # y_proj = y_cam / Z_cv
    # Simplified:
    x_proj = x_cam / (-z_cam)
    y_proj = y_cam / (-z_cam)


    # Apply intrinsics
    u_px = fx * x_proj + cx
    v_px = fy * y_proj + cy # OpenCV uses +Y down convention for pixel coords

    return int(u_px), int(v_px)

# --- Modified visualize_data function ---
def visualize_data(image, data, base_name):
    """Draws ground truth info, projected points, coordinate axes, and axis labels."""
    # --- (Keep initial checks and camera intrinsics loading) ---
    if image is None or data is None:
        # ... (error handling) ...
        if image is not None:
            vis_image = image.copy()
            cv2.putText(vis_image, f"File: {base_name} (JSON Error)", (10, TEXT_START_Y),
                        FONT, FONT_SCALE, (0, 0, 255), LINE_TYPE)
            return vis_image
        return None

    vis_image = image.copy()
    current_y = TEXT_START_Y

    intrinsics = data.get('camera_data', {}).get('intrinsics')
    if intrinsics is None:
        # print(f"Warning: Camera intrinsics not found...") # Optional warning
        fx, fy, cx, cy = None, None, None, None
    else:
        fx = intrinsics.get('fx')
        fy = intrinsics.get('fy')
        cx = intrinsics.get('cx')
        cy = intrinsics.get('cy')
        if None in [fx, fy, cx, cy]:
            # print(f"Warning: Incomplete camera intrinsics...") # Optional warning
            fx, fy, cx, cy = None, None, None, None

    # --- (Keep filename/coord system text) ---
    cv2.putText(vis_image, f"File: {base_name}", (10, current_y), FONT, FONT_SCALE, (255, 255, 255), LINE_TYPE)
    current_y += TEXT_LINE_HEIGHT
    cv2.putText(vis_image, "Coords: -Z fwd, X right, Y up", (10, current_y), FONT, FONT_SCALE * 0.9, (200, 200, 200), LINE_TYPE)
    current_y += TEXT_LINE_HEIGHT * 2

    if 'objects' not in data:
        # ... (no objects handling) ...
        cv2.putText(vis_image, "No 'objects' key in JSON", (10, current_y), FONT, FONT_SCALE, (0, 0, 255), LINE_TYPE)
        return vis_image

    # Iterate through objects
    for i, obj in enumerate(data['objects']):
        # --- (Keep object info retrieval: name, class, loc, quat, scale, euler) ---
        obj_name = obj.get('name', f"Object {i+1}")
        obj_class = obj.get('class', 'Unknown')
        location = obj.get('location')
        quaternion = obj.get('quaternion_xyzw')
        scale = obj.get('scale', [1.0, 1.0, 1.0])
        rotation_euler_deg = quaternion_to_euler_degrees(quaternion)

        has_pose = location is not None and quaternion is not None
        # --- (Keep pose validation and rot_matrix calculation) ---
        if has_pose and len(location) == 3 and len(quaternion) == 4 and len(scale) == 3:
            loc_vec = np.array(location)
            try:
                rot_matrix = R.from_quat(quaternion).as_matrix()
            except Exception as e:
                print(f"Error converting quaternion for object {i}: {e}")
                rot_matrix = None
                has_pose = False
        else:
            loc_vec = None
            rot_matrix = None
            has_pose = False

        # --- (Keep object info text drawing) ---
        header = f"- {obj_name} ({obj_class})"
        cv2.putText(vis_image, header, (10, current_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
        current_y += TEXT_LINE_HEIGHT
        loc_str = f"  Loc (X,Y,Z): {format_vector(location)}"
        cv2.putText(vis_image, loc_str, (10, current_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
        current_y += TEXT_LINE_HEIGHT
        rot_str = f"  Rot (Euler XYZ deg): {format_vector(rotation_euler_deg)}"
        cv2.putText(vis_image, rot_str, (10, current_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
        current_y += TEXT_LINE_HEIGHT
        scale_str = f"  Scale (X,Y,Z): {format_vector(scale)}"
        cv2.putText(vis_image, scale_str, (10, current_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
        current_y += TEXT_LINE_HEIGHT

        # --- Draw Coordinate Axes ---
        if has_pose and intrinsics and fx:
            # --- (Keep local axis endpoint definition and transformation to camera coords) ---
            len_x = AXIS_BASE_LENGTH * scale[0]
            len_y = AXIS_BASE_LENGTH * scale[1]
            len_z = AXIS_BASE_LENGTH * scale[2]
            origin_local = np.array([0, 0, 0])
            x_axis_local = np.array([len_x, 0, 0])
            y_axis_local = np.array([0, len_y, 0])
            z_axis_local = np.array([0, 0, -len_z])
            origin_cam = loc_vec
            x_axis_cam = rot_matrix @ x_axis_local + loc_vec
            y_axis_cam = rot_matrix @ y_axis_local + loc_vec
            z_axis_cam = rot_matrix @ z_axis_local + loc_vec

            # --- (Keep projection to 2D) ---
            origin_px = project_point(origin_cam, fx, fy, cx, cy)
            x_axis_px = project_point(x_axis_cam, fx, fy, cx, cy)
            y_axis_px = project_point(y_axis_cam, fx, fy, cx, cy)
            z_axis_px = project_point(z_axis_cam, fx, fy, cx, cy)

            # --- Draw lines AND LABELS if projection is valid ---
            if origin_px:
                # X Axis (Red)
                if x_axis_px:
                    cv2.line(vis_image, origin_px, x_axis_px, AXIS_COLOR_X, AXIS_LINE_THICKNESS)
                    # Add Label "X"
                    label_pos_x = (x_axis_px[0] + AXIS_LABEL_OFFSET_X, x_axis_px[1] + AXIS_LABEL_OFFSET_Y)
                    cv2.putText(vis_image, "X", label_pos_x, FONT, AXIS_LABEL_FONT_SCALE, AXIS_COLOR_X, AXIS_LABEL_LINE_TYPE)

                # Y Axis (Green)
                if y_axis_px:
                    cv2.line(vis_image, origin_px, y_axis_px, AXIS_COLOR_Y, AXIS_LINE_THICKNESS)
                    # Add Label "Y"
                    label_pos_y = (y_axis_px[0] + AXIS_LABEL_OFFSET_X, y_axis_px[1] + AXIS_LABEL_OFFSET_Y)
                    cv2.putText(vis_image, "Y", label_pos_y, FONT, AXIS_LABEL_FONT_SCALE, AXIS_COLOR_Y, AXIS_LABEL_LINE_TYPE)

                # Z Axis (Blue, points -Z)
                if z_axis_px:
                    cv2.line(vis_image, origin_px, z_axis_px, AXIS_COLOR_Z, AXIS_LINE_THICKNESS)
                    # Add Label "Z"
                    label_pos_z = (z_axis_px[0] + AXIS_LABEL_OFFSET_X, z_axis_px[1] + AXIS_LABEL_OFFSET_Y)
                    cv2.putText(vis_image, "Z", label_pos_z, FONT, AXIS_LABEL_FONT_SCALE, AXIS_COLOR_Z, AXIS_LABEL_LINE_TYPE)


        # --- (Keep projected cuboid points drawing) ---
        projected_points_pixels = obj.get('projected_cuboid')
        if projected_points_pixels is not None:
            # ... (validation and drawing loop for cuboid points/indices) ...
            is_valid_format = False
            if isinstance(projected_points_pixels, list) and len(projected_points_pixels) > 0:
                first_point = projected_points_pixels[0]
                if isinstance(first_point, list) and len(first_point) == 2:
                    if isinstance(first_point[0], (int, float)) and isinstance(first_point[1], (int, float)):
                        is_valid_format = True

            if is_valid_format:
                # if len(projected_points_pixels) != 9: pass # Optional warning removed

                for idx, point_coords in enumerate(projected_points_pixels):
                    if not (isinstance(point_coords, list) and len(point_coords) == 2 and
                            isinstance(point_coords[0], (int, float)) and isinstance(point_coords[1], (int, float))):
                        continue

                    px = int(point_coords[0])
                    py = int(point_coords[1])
                    cv2.circle(vis_image, (px, py), POINT_CIRCLE_RADIUS, POINT_CIRCLE_COLOR, -1)
                    index_text = str(idx)
                    text_pos = (px + INDEX_OFFSET_X, py + INDEX_OFFSET_Y)
                    cv2.putText(vis_image, index_text, text_pos, FONT, INDEX_FONT_SCALE, INDEX_FONT_COLOR, LINE_TYPE)
        # else: pass # Optional format warning removed
        # else: pass # Optional missing key warning removed


        current_y += OBJECT_INFO_OFFSET_Y # Add extra space before next object

    return vis_image

# --- (Keep main function) ---
# Remember to update AXIS_BASE_LENGTH and add the new AXIS_LABEL_* constants at the top
# Ensure the main() function is still present below.
def main():
    """Main function to run the inspector."""
    # ... (setup code, find_data_pairs) ...
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Directory not found: {DATA_DIR}")
        print("Please update the 'DATA_DIR' variable in the script.")
        return

    file_pairs = find_data_pairs(DATA_DIR, IMAGE_EXTENSION, JSON_EXTENSION)

    if not file_pairs:
        print(f"No matching image ({IMAGE_EXTENSION}) and json ({JSON_EXTENSION}) pairs found in {DATA_DIR}")
        return

    num_files = len(file_pairs)
    current_index = 0


    print("\n--- 6D Pose Ground Truth Inspector ---")
    print(f"Found {num_files} image/JSON pairs in '{DATA_DIR}'")
    print("Assuming Coordinate System: -Z Forward, X Right, Y Up")
    print("Using 'projected_cuboid' key for pixel coordinates.")
    print("Visualizing object coordinate axes (X:Red, Y:Green, Z:Blue[-Z fwd]) with Labels.") # Updated info
    print("Controls:")
    print("  Right Arrow / 'n' -> Next Image")
    print("  Left Arrow  / 'p' -> Previous Image")
    print("  'q' / Esc       -> Quit")
    print("-------------------------------------\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        # ... (main loop logic: load data, call visualize_data, imshow, waitKey) ...
        current_pair = file_pairs[current_index]

        image, data = load_data(current_pair)

        vis_image = visualize_data(image, data, current_pair['base_name'])

        if vis_image is None:
            height, width = (480, 640) # Default size
            vis_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(vis_image, f"Error loading image or JSON: {current_pair['base_name']}", (10, 30), FONT, 0.7, (0, 0, 255), 1)


        cv2.imshow(WINDOW_NAME, vis_image)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('n') or key == 83: # Right arrow might be 83
            current_index = (current_index + 1) % num_files
        elif key == ord('p') or key == 81: # Left arrow might be 81
            current_index = (current_index - 1 + num_files) % num_files


    cv2.destroyAllWindows()
    print("Inspector closed.")


if __name__ == "__main__":
    main()
