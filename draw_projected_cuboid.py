import cv2
import json
import os
from pathlib import Path

# Define connections between projected_cuboid points
CONNECTIONS = [
    (4, 8), (4, 2), (4, 3), (8, 6),
    (8, 7), (2, 6), (6, 5), (5, 7),
    (5, 1), (7, 3), (3, 1), (2, 1),
    (4, 6), (8, 2)
]

def load_images_and_jsons(directory):
    """Load all image files and corresponding JSON files from directory"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(directory).glob(ext))
    return sorted(image_files, key=lambda x: x.name)

def draw_cuboid(image, cuboid, color=(0, 255, 0), thickness=2):
    """Draw 3D cuboid on image using predefined connections"""
    for (i, j) in CONNECTIONS:
        try:
            x1, y1 = cuboid[i]
            x2, y2 = cuboid[j]
            cv2.line(image,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     color, thickness)
        except IndexError:
            continue  # Skip invalid connections

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw 2D bounding box on image"""
    x1, y1 = bbox['top_left']
    x2, y2 = bbox['bottom_right']
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def main(mode='3d'):
    image_dir = '/home/lore_be/data/deleteme2/'  # Current directory
    image_files = load_images_and_jsons(image_dir)
    current_idx = 0

    while True:
        if current_idx < 0:
            current_idx = 0
        elif current_idx >= len(image_files):
            current_idx = len(image_files) - 1

        image_path = image_files[current_idx]
        json_path = image_path.with_suffix('.json')

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error loading image: {image_path}")
            current_idx += 1
            continue

        # Load and draw annotations
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            for obj in data.get('objects', []):
                if mode == '3d':
                    cuboid = obj.get('projected_cuboid', [])
                    if cuboid:
                        draw_cuboid(img, cuboid)
                elif mode == '2d':
                    bbox = obj.get('bounding_box', {})
                    if bbox:
                        draw_bbox(img, bbox)
        else:
            print(f"No JSON found for {image_path.name}")

        # Display image
        cv2.imshow('Bounding Boxes', img)
        key = cv2.waitKeyEx(0)

        # Handle keyboard input
        if key in [27, ord('q')]:  # ESC or Q
            break
        elif key in [65361, 2424832]:  # Left arrow (Linux/Windows)
            current_idx -= 1
        elif key in [65363, 2555904]:  # Right arrow (Linux/Windows)
            current_idx += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # You can change the mode here: '2d' or '3d'
    main(mode='3d')