#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import yaml
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


class KITTI3DToFATConverter:
    """
    Converts [KITTI 3D Object Detection Evaluation 2017 Training Image Dataset]
    (https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
    to a format similar to Falling Things, but with a specific coordinate
    system: -z-front, x-right, -y-down (right-handed).
    """

    def __init__(self, args):
        self.args = args # Store args for later use
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            # Allow overwriting if specified, otherwise check if empty
            if not args.overwrite and len(os.listdir(args.output_dir)) > 0:
                raise AssertionError('Error: Output dir must be empty or --overwrite flag must be set.')
            elif args.overwrite:
                print(f"Warning: Overwriting content in {args.output_dir}")


        self.images = None
        self.camera_calibration = None
        self.objects = None

        # Transformation matrix from KITTI camera coordinates to Target coordinates
        # KITTI: x-right, y-down, z-front
        # TARGET: x-right, -y-down, -z-front
        # Target_x = Kitti_x
        # Target_y = -Kitti_y
        # Target_z = -Kitti_z
        self.T_target_from_kitti = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        # Inverse transform (Target to KITTI) is the same matrix
        self.T_kitti_from_target = self.T_target_from_kitti

    def run(self):
        self.images = self._get_files_by_format(self.args.kitti_dir, '.png')
        assert len(self.images) > 0, f"Error: No png files found recursively in {self.args.kitti_dir}."

        processed_count = 0
        for i, image_filename in enumerate(self.images):
            base_name = os.path.basename(image_filename)
            frame_id = os.path.splitext(base_name)[0]

            # --- Find corresponding calibration and label files ---
            # Assume standard KITTI structure: images in 'image_2', labels in 'label_2', calib in 'calib'
            # Construct paths relative to the image file found
            base_dir = os.path.dirname(os.path.dirname(image_filename)) # Go up two levels from image_2/xxxxxx.png
            calib_file = os.path.join(base_dir, 'calib', frame_id + '.txt')
            label_file = os.path.join(base_dir, 'label_2', frame_id + '.txt')

            if not os.path.exists(calib_file):
                print(f"Warning: Calibration file not found for {frame_id}, skipping. Searched at: {calib_file}")
                continue
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for {frame_id}, skipping. Searched at: {label_file}")
                continue
            # --- End Find ---

            self.camera_calibration = self._parse_calibration_file(calib_file)
            ground_truth_KITTI = self._parse_ground_truth_file(label_file)
            self.objects = self._convert_objects(ground_truth_KITTI,
                                                 self.camera_calibration)

            # filter for cars only (or other classes if needed)
            self.objects = [obj for obj in self.objects if
                            obj['class'] in ['car']] # Make this a list if more classes needed

            # store dataset
            if len(self.objects) == 0:
                continue

            output_image_path = os.path.join(self.args.output_dir, base_name)
            output_json_path = os.path.join(self.args.output_dir, frame_id + '.json')
            output_yaml_path = os.path.join(self.args.output_dir, frame_id + '.yml')

            shutil.copy(image_filename, output_image_path)
            camera_info_ros = self._convert_to_ros_camera_info(self.camera_calibration)
            centerpose_camera_data = self._convert_to_centerpose_camera_data(camera_info_ros)
            self._dump_annotation_file(self.objects, centerpose_camera_data, output_json_path)

            if self.args.save_camera_info:
                with open(output_yaml_path, "w") as write_file:
                    yaml.dump(camera_info_ros, write_file, default_flow_style=None,
                              sort_keys=False)

            if self.args.debug:
                # Load the *copied* image for drawing
                self._draw_cuboid_points(image_path=output_image_path)

            processed_count += 1
            sys.stdout.write(f"Converted {processed_count}/{len(self.images)} items (Frame: {frame_id}).\r")
            sys.stdout.flush()

        print(f"\nConversion finished. Processed {processed_count} images.")

    def _get_files_by_format(self, directory: str, ending: str) -> list:
        """
        Returns all files in given dir (recursively) with given ending.
        Specifically looks within subdirectories like 'image_2'.

        Parameters
        ----------
        directory : Root directory of KITTI data (e.g., 'training' or 'testing').
        ending : Ending of files to return (e.g., '.png').

        Returns
        -------
        list of strings containing full paths to all found files.
        """
        # Look for images specifically in common KITTI image folders like 'image_2'
        search_pattern = os.path.join(directory, '**', f'*{ending}')
        files = glob(search_pattern, recursive=True)
        # Filter to ensure we are likely getting dataset images (e.g. inside image_2 or image_3)
        files = [f for f in files if os.path.basename(os.path.dirname(f)).startswith('image_')]
        return sorted(files) # Sort for deterministic order

    def _convert_to_ros_camera_info(self, camera_calibration: dict) -> dict:
        """
        Converts given camera_calibration information to camera_info.yaml (ROS CameraInfo) format.
        Uses P2 for the left color camera projection.

        Parameters
        ----------
        camera_calibration : Camera calibration information in KITTI format.
        """
        # Ensure matrices are reshaped correctly if needed
        P2 = camera_calibration.get('P2')
        R0_rect = camera_calibration.get('R0_rect')

        if P2 is None:
            raise ValueError("Calibration data missing 'P2'")
        if R0_rect is None:
            # Provide a default identity if R0_rect is missing (though it shouldn't be for stereo setups)
            print("Warning: Calibration data missing 'R0_rect'. Using identity.")
            R0_rect = np.identity(3).flatten()
        else:
            R0_rect = R0_rect.flatten() # Flatten if it's 3x3

        # Reshape P2 into 3x4
        projection_matrix = P2.reshape(3, 4)

        # Extract 3x3 camera matrix (K) from P2
        camera_matrix = projection_matrix[:3, :3]

        # Extract rectification matrix R (use R0_rect)
        rectification_matrix = R0_rect.reshape(3, 3)

        # Assume image dimensions (these might need to be read from images if variable)
        # Typical KITTI dims, adjust if necessary
        img_width = 1242 # Example width
        img_height = 375 # Example height
        # If images are variable size, read them:
        # try:
        #     with Image.open(image_path) as img:
        #         img_width, img_height = img.size
        # except Exception as e:
        #     print(f"Warning: Could not read image dimensions, using defaults. Error: {e}")


        camera_info = {
            "image_width": img_width,
            "image_height": img_height,
            "camera_name": 'kitti_cam2', # Standard name for left color camera
            "distortion_model": "plumb_bob", # KITTI images are rectified, so distortion is ~0
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": camera_matrix.flatten().tolist()
            },
            "distortion_coefficients": {
                "rows": 1,
                "cols": 5,
                "data": [0.0, 0.0, 0.0, 0.0, 0.0] # Assuming rectified
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": rectification_matrix.flatten().tolist()
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": projection_matrix.flatten().tolist()
            },
        }
        return camera_info

    def _dump_annotation_file(self, objects: list, camera_data: dict, filename: str) -> None:
        """
        Stores objects ground truth information in the target format as json file.
        Parameters
        ----------
        objects : List containing object information in the target format.
        camera_data: Dict containing camera information in centerpose format.
        filename : Path to store json file.
        """

        output = {
            "AR_data": { # Placeholder, adjust if needed
                "plane_center": [ 0.0, 0.0, 0.0 ],
                "plane_normal": [ 0.0, 0.0, 0.0] # Often [0, 1, 0] for ground plane
            },
            'camera_data': camera_data,
            'objects': objects
        }

        with open(filename, "w") as write_file:
            json.dump(output, write_file, indent=4)

    def _draw_cuboid_points(self, image_path: str) -> None:
        """
        Draws projected cuboid of objects on given image.

        Parameters
        ----------
        image_path : Path to image.
        """
        try:
            fnt = ImageFont.truetype('DejaVuSansMono.ttf', 20) # Try a common default font
        except IOError:
            try:
                fnt = ImageFont.truetype('Arial.ttf', 20)
            except IOError:
                print("Warning: Cannot load font DejaVuSansMono or Arial. Using default.")
                fnt = ImageFont.load_default()

        with Image.open(image_path) as im:
            d = ImageDraw.Draw(im)
            # Ensure colors are distinct if possible
            colors = ['red', 'lime', 'blue', 'yellow', 'fuchsia', 'aqua']
            color_idx = 0

            for obj in self.objects:
                color = colors[color_idx % len(colors)]
                color_idx += 1

                # Use 'projected_cuboid' which should be the 9 points for centerpose
                projected_points = obj.get('projected_cuboid')
                if not projected_points or len(projected_points) != 9:
                    print(f"Warning: Skipping drawing object {obj.get('class')} due to missing/invalid 'projected_cuboid'.")
                    continue

                # projected_points format: [centroid, p1, p2, p3, p4, p5, p6, p7, p8]
                # Convert to integer tuples for drawing
                points = [(int(p[0]), int(p[1])) for p in projected_points]
                centroid = points[0]
                corners = points[1:] # 8 corner points

                # Draw the 8 corners based on CenterPose ordering (visual inspection needed)
                # Assuming a standard box connection pattern based on common 3D box definitions.
                # This might need adjustment based on how _reorder_cuboid_for_centerpose actually orders them.
                # Typical connections:
                # Bottom face: 0-1-2-3-0 (e.g., corners[0]-corners[1]-corners[2]-corners[3]-corners[0])
                # Top face: 4-5-6-7-4    (e.g., corners[4]-corners[5]-corners[6]-corners[7]-corners[4])
                # Vertical edges: 0-4, 1-5, 2-6, 3-7

                # Let's assume the FAT reordering maps to this standard ordering:
                # FAT indices: [7, 3, 4, 0, 6, 2, 5, 1] -> Corners indices: [0, 1, 2, 3, 4, 5, 6, 7]
                # corners[0] = fat_cuboid[7]
                # corners[1] = fat_cuboid[3]
                # corners[2] = fat_cuboid[4]
                # corners[3] = fat_cuboid[0]
                # corners[4] = fat_cuboid[6]
                # corners[5] = fat_cuboid[2]
                # corners[6] = fat_cuboid[5]
                # corners[7] = fat_cuboid[1]

                # Draw bottom face (indices 0, 1, 2, 3 in corners list)
                d.line([corners[0], corners[1]], fill=color, width=2)
                d.line([corners[1], corners[2]], fill=color, width=2)
                d.line([corners[2], corners[3]], fill=color, width=2)
                d.line([corners[3], corners[0]], fill=color, width=2)

                # Draw top face (indices 4, 5, 6, 7 in corners list)
                d.line([corners[4], corners[5]], fill=color, width=2)
                d.line([corners[5], corners[6]], fill=color, width=2)
                d.line([corners[6], corners[7]], fill=color, width=2)
                d.line([corners[7], corners[4]], fill=color, width=2)

                # Draw vertical edges (0-4, 1-5, 2-6, 3-7)
                d.line([corners[0], corners[4]], fill=color, width=2)
                d.line([corners[1], corners[5]], fill=color, width=2)
                d.line([corners[2], corners[6]], fill=color, width=2)
                d.line([corners[3], corners[7]], fill=color, width=2)

                # Draw centroid
                # d.point(centroid, fill=color) # Often too small
                d.ellipse((centroid[0]-3, centroid[1]-3, centroid[0]+3, centroid[1]+3), fill=color, outline=color)
                d.text((centroid[0]+5, centroid[1]+5), text=obj.get('class','?')[0].upper(), font=fnt, fill=color)

            im.save(image_path) # Overwrite the image with drawings

    def _parse_calibration_file(self, calibration_file: str) -> dict:
        """
        Parses sensor calibration data from a KITTI calib file into dict.

        Parameters
        ----------
        calibration_file : Path to the calibration file.

        Returns
        -------
        dict: Sensor calibration data.
        """
        calibration = {}
        with open(calibration_file, 'r') as file:
            for line in file.readlines():
                if ':' in line:
                    key, value_str = line.split(':', 1)
                    key = key.strip()
                    try:
                        values = np.array([float(x) for x in value_str.strip().split()])
                        calibration[key] = values
                    except ValueError:
                        print(f"Warning: Could not parse line in {calibration_file}: {line.strip()}")
        return calibration

    def _parse_ground_truth_file(self, ground_truth_file: str) -> dict:
        """
        Parses ground truth data in KITTI format from a label file into dict.

        Parameters
        ----------
        ground_truth_file : Path to the ground truth label file.

        Returns
        -------
        dict: Ground truth data grouped by class name. Keys are class names (str),
              values are numpy arrays where each row is an object instance.
        """
        ground_truth = {}
        column_names = [
            'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
            'bbox_right', 'bbox_bottom', 'dim_height', 'dim_width', 'dim_length',
            'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score' # Score only for results
        ] # Based on KITTI readme

        with open(ground_truth_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts: continue # Skip empty lines

            class_name = parts[0]
            values = []
            try:
                # Try converting remaining parts to float
                values = [float(x) for x in parts[1:]]
            except ValueError:
                print(f"Warning: Could not parse values in {ground_truth_file} for line: {line.strip()}")
                continue

            # Ensure we have the correct number of expected values (14 for GT: skip type)
            if len(values) < 14:
                print(f"Warning: Skipping line due to insufficient values ({len(values)} < 14) in {ground_truth_file}: {line.strip()}")
                continue
            elif len(values) > 14: # Handle potential score column if present
                values = values[:14] # Take only the first 14 values after class name


            if class_name not in ground_truth:
                ground_truth[class_name] = []

            ground_truth[class_name].append(values)

        # Convert lists to numpy arrays
        for key in ground_truth.keys():
            ground_truth[key] = np.array(ground_truth[key])

        return ground_truth

    def _compute_cuboid_corners(self, dimensions: np.array, location_kitti: np.array,
                                rotation_matrix_kitti: np.array) -> np.array:
        """
        Computes the 8 corner points of a 3D bounding box in KITTI camera coordinates.

        Parameters
        ----------
        dimensions : [height, width, length] (H, W, L) of the object in meters.
        location_kitti : [x, y, z] location of the object *bottom center* in KITTI camera space.
        rotation_matrix_kitti: 3x3 Rotation matrix around y-axis (in KITTI camera space).

        Returns
        -------
        np.array: 3x8 matrix where columns are the [x, y, z] coordinates
                  of the 8 cuboid corners in KITTI camera space.
        """
        h, w, l = dimensions[0], dimensions[1], dimensions[2]

        # 3D bounding box corners (object coordinate system, origin at bottom center)
        # x: length, y: height, z: width (following convention where -y is up)
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h] # KITTI y is down, so negative height is up
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_obj = np.array([x_corners, y_corners, z_corners]) # Shape (3, 8)

        # Rotate corners
        corners_rotated = rotation_matrix_kitti.dot(corners_obj)

        # Translate corners to the location in camera coordinates
        corners_cam_kitti = corners_rotated + location_kitti.reshape((3, 1))

        return corners_cam_kitti # Shape (3, 8)

    def _project_points_to_image(self, points_3d_kitti: np.array, projection_matrix_p2: np.array) -> np.array:
        """
        Projects 3D points (in KITTI camera coordinates) to 2D image coordinates using P2.

        Parameters
        ----------
        points_3d_kitti : np.array shape (3, N), points in KITTI camera coordinates.
        projection_matrix_p2 : np.array shape (3, 4), the P2 projection matrix.

        Returns
        -------
        np.array: shape (2, N), the projected 2D points [u, v].
        """
        num_points = points_3d_kitti.shape[1]
        # Convert to homogeneous coordinates (add row of 1s)
        points_3d_homo = np.vstack((points_3d_kitti, np.ones((1, num_points)))) # Shape (4, N)

        # Project using P2
        points_2d_homo = projection_matrix_p2.dot(points_3d_homo) # Shape (3, N)

        # Normalize (divide by third coordinate z/w)
        # Avoid division by zero or near-zero
        points_2d = np.zeros((2, num_points))
        valid_idx = points_2d_homo[2, :] > 0.01 # Check if points are in front of camera
        points_2d[0, valid_idx] = points_2d_homo[0, valid_idx] / points_2d_homo[2, valid_idx]
        points_2d[1, valid_idx] = points_2d_homo[1, valid_idx] / points_2d_homo[2, valid_idx]
        # Handle points behind or too close? For now, they result in [0, 0]

        return points_2d # Shape (2, N)


    def _convert_objects(self, ground_truth: dict,
                         sensor_calibration: dict) -> list:
        """
        Converts KITTI ground truth objects to the target format.

        Parameters
        ----------
        ground_truth : Dict containing ground truth parsed from KITTI labels file.
        sensor_calibration : Dict containing sensor calibration parsed from KITTI calib file.

        Returns
        -------
        List of object dictionaries containing object information in the target format.
        """
        # Get the P2 projection matrix (used for 2D projection)
        projection_matrix_p2 = sensor_calibration['P2'].reshape(3, 4)

        # KITTI occlusion levels to a continuous value (example mapping)
        kitti_occlusion_to_visibility_factor = {0: 0.0, 1: 0.25, 2: 0.75, 3: 1.0} # Higher number = more occluded

        objects = []

        for class_id, gt_data in ground_truth.items():
            # Skip classes we don't care about (e.g., 'DontCare', 'Misc')
            if class_id in ['DontCare', 'Misc']:
                continue

            for i in range(gt_data.shape[0]):
                gt_object = gt_data[i] # Row for this specific object

                # --- Extract KITTI Data ---
                # Indices based on _parse_ground_truth_file structure (after removing class name)
                truncation = float(gt_object[0])
                occlusion_level = int(gt_object[1]) # 0, 1, 2, 3
                alpha = float(gt_object[2]) # Observation angle
                bbox_2d = [float(gt_object[3]), float(gt_object[4]), float(gt_object[5]), float(gt_object[6])] # left, top, right, bottom
                dimensions = np.array([gt_object[7], gt_object[8], gt_object[9]]) # H, W, L in meters
                location_kitti = np.array([gt_object[10], gt_object[11], gt_object[12]]) # x, y, z in KITTI camera coords (bottom center)
                rotation_y_kitti = float(gt_object[13]) # Rotation around Y-axis in KITTI camera coords

                # --- Coordinate System Transformation ---
                # 1. Transform Location
                location_target = self.T_target_from_kitti.dot(location_kitti)

                # 2. Transform Rotation
                # KITTI rotation is around positive Y-axis ([0, 1, 0])
                # Target Y-axis is negative KITTI Y-axis ([0, -1, 0])
                # Rotation by theta around [0, 1, 0] is equivalent to rotation by -theta around [0, -1, 0]
                # We define the rotation in the *target* coordinate system.
                rotation_target = Rotation.from_rotvec([0, -rotation_y_kitti, 0]) # Axis is [0,1,0] in target frame, angle is negated
                quaternion_xyzw_target = rotation_target.as_quat() # Scipy default is [x, y, z, w]

                # 3. Compute 3D Cuboid Corners
                #    First compute in KITTI coordinates, then transform points.
                rotation_matrix_kitti = Rotation.from_rotvec([0, rotation_y_kitti, 0]).as_matrix()
                cuboid_corners_kitti = self._compute_cuboid_corners(dimensions, location_kitti, rotation_matrix_kitti) # Shape (3, 8)

                # Transform corner points to the target coordinate system
                cuboid_corners_target = self.T_target_from_kitti.dot(cuboid_corners_kitti) # Shape (3, 8)

                # --- Projections ---
                # Project the *original KITTI* 3D corners to 2D using P2
                projected_corners_2d = self._project_points_to_image(cuboid_corners_kitti, projection_matrix_p2) # Shape (2, 8)

                # --- Reorder & Prepare Output ---
                # Reorder 3D cuboid corners from calculation order to FAT order, then CenterPose order
                # Calculation order: Check _compute_cuboid_corners (depends on x,y,z corner definitions)
                # Assuming standard order from _compute_cuboid_corners corresponds to indices 0..7
                # KITTI order -> FAT order map (Indices based on FAT readme fig. compared to typical box):
                # FAT Point Index: [0,  1,  2,  3,  4,  5,  6,  7]
                # Typical Corner: [FLL,FRL,BRL,BLL,FUL,FRU,BRU,BUL] (Front/Back, Top/Bottom, Left/Right)
                # This mapping needs careful verification based on _compute_cuboid_corners output.
                # Let's assume _compute_cuboid_corners produces corners 0-3 bottom, 4-7 top in some order.
                # And let's reuse the original script's reordering logic for FAT and CenterPose

                # Reorder 3D points (Target Coordinates)
                #cuboid_fat_order_target = self._reorder_cuboid_points_kitti_to_fat(cuboid_corners_target) # Shape (3, 8)
                # Calculate centroid in target coords
                centroid_target = np.mean(cuboid_corners_target, axis=1) # Avg across the 8 points
                cuboid_centerpose_target = self._reorder_cuboid_for_centerpose(
                    cuboid_corners_target.T.tolist(), # Input needs list of lists [N, 3]
                    centroid_target.tolist()
                ) # 9 points: [centroid, p1..p8]

                # Reorder 2D projected points
                #projected_corners_fat_order = self._reorder_cuboid_points_kitti_to_fat(projected_corners_2d) # Shape (2, 8)
                # Calculate projected centroid
                projected_centroid = np.mean(projected_corners_2d, axis=1)
                projected_cuboid_centerpose = self._reorder_cuboid_for_centerpose(
                    projected_corners_2d.T.tolist(), # Input needs list of lists [N, 2]
                    projected_centroid.tolist()
                ) # 9 points: [centroid, p1..p8]

                # Apply scaling if requested (AFTER all coordinate transforms)
                if self.args.distance_in_cm:
                    location_target *= 100.0
                    # cuboid_centerpose_target is a list of lists, need to scale each point
                    cuboid_centerpose_target = [[coord * 100.0 for coord in point] for point in cuboid_centerpose_target]
                    # Scale dimensions H, W, L (but output format expects scale [W, H, L])
                    scale_target = [dimensions[1]*100.0, dimensions[0]*100.0, dimensions[2]*100.0]
                else:
                    # Scale is W, H, L
                    scale_target = [dimensions[1], dimensions[0], dimensions[2]]

                # --- Visibility & Occlusion (optional refinement) ---
                visibility = 1.0 - truncation # Basic visibility based on truncation
                # Refine based on occlusion level (example)
                occlusion_factor = kitti_occlusion_to_visibility_factor.get(occlusion_level, 0.0)
                visibility = max(0.0, visibility - occlusion_factor)


                # --- Build Final Object Dictionary ---
                object_dict = {
                    "class": class_id.lower(),
                    "name": f"{class_id.lower()}_{i}", # Unique name
                    "provenance": "kitti", # Indicate source
                    'visibility': visibility, # Estimated visibility
                    'truncation': truncation, # Raw truncation value
                    'occlusion': occlusion_level, # Raw occlusion level
                    'bounding_box_2d': { # Using KITTI's 2D bbox
                        'top_left': [bbox_2d[0], bbox_2d[1]],
                        'bottom_right': [bbox_2d[2], bbox_2d[3]]
                    },
                    # --- Core 3D Information in Target Coordinate System ---
                    'location': location_target.tolist(), # [x, y, z] in target coords
                    'quaternion_xyzw': quaternion_xyzw_target.tolist(), # [x, y, z, w] in target coords
                    'scale': scale_target, # [Width, Height, Length] in meters or cm
                    # --- Keypoints in Target Coordinate System ---
                    'keypoints_3d': cuboid_centerpose_target, # 9 points [centroid, p1..p8] in target coords
                    # --- Projected Keypoints (for visualization/2D tasks) ---
                    'projected_cuboid': projected_cuboid_centerpose, # 9 points [centroid, p1..p8] in 2D image coords
                }
                objects.append(object_dict)

        return objects

    def _reorder_cuboid_points_kitti_to_fat(self, cuboid_kitti_order: np.array) -> np.array:
        """
        Reorders points of a cuboid from a typical KITTI calculation order
        to the FAT format order.

        NOTE: This assumes the input `cuboid_kitti_order` (shape 3x8 or 2x8)
        has a consistent corner ordering produced by `_compute_cuboid_corners`.
        The exact mapping depends on how corners were defined there.
        This implementation uses the *same* re-indexing as the original script.
        Verify this mapping if `_compute_cuboid_corners` changes significantly.

        FAT order (indices 0-7) vs a possible calculation order (indices 0-7):
        Seems the original script maps Kitti indices [1, 4, 3, 2, 0, 5, 6, 7]
        to FAT indices [0, 1, 2, 3, 4, 5, 6, 7].

        Parameters
        ----------
        cuboid_kitti_order : np.array shape (D, 8) where D is dimension (2 or 3).
                            Corners assumed to be in the order from _compute_cuboid_corners.

        Returns
        -------
        np.array shape (D, 8): Cuboid corners reordered to FAT convention.
        """
        # Indices from original script's _reorder_cuboid_points function
        # FAT[0] = KITTI[1]
        # FAT[1] = KITTI[4]
        # FAT[2] = KITTI[3]
        # FAT[3] = KITTI[2]
        # FAT[4] = KITTI[0]
        # FAT[5] = KITTI[5]
        # FAT[6] = KITTI[6]
        # FAT[7] = KITTI[7]
        fat_order_indices = [1, 4, 3, 2, 0, 5, 6, 7]

        if cuboid_kitti_order.shape[1] != 8:
            raise ValueError(f"Input cuboid must have 8 points, got {cuboid_kitti_order.shape[1]}")

        cuboid_fat = cuboid_kitti_order[:, fat_order_indices]
        return cuboid_fat

    def _reorder_cuboid_for_centerpose(self, kitti_cuboid_points: list, centroid: list) -> list:
        """
        Reorders 8 cuboid points (already in KITTI order) and prepends the centroid
        to match the 9-point CenterPose format.

        Parameters
        ----------
        kitti_cuboid_points : List of 8 points [[x,y,z], ... or [x,y], ...] in kitti order.
        centroid : List [x,y,z] or [x,y] representing the centroid.

        Returns
        -------
        List of 9 points: [centroid, p1, p2, p3, p4, p5, p6, p7, p8]
                        where p1..p8 are corners reordered from FAT for CenterPose.
                        The specific order p1..p8 comes from the original script.
        """
        if len(kitti_cuboid_points) != 8:
            raise ValueError(f"Input list must have 8 points for FAT cuboid, got {len(kitti_cuboid_points)}")

        # CenterPose order relative to FAT order indices (from original script):
        # CenterPose[0] = centroid
        # CenterPose[1] = FAT[7]
        # CenterPose[2] = FAT[3]
        # CenterPose[3] = FAT[4]
        # CenterPose[4] = FAT[0]
        # CenterPose[5] = FAT[6]
        # CenterPose[6] = FAT[2]
        # CenterPose[7] = FAT[5]
        # CenterPose[8] = FAT[1]

        #centerpose_indices_from_kitti = [2,1,6,5,3,0,7,4]
        centerpose_indices_from_kitti = [3,0,7,4,2,1,6,5]
        reordered_corners = [kitti_cuboid_points[i] for i in centerpose_indices_from_kitti]
        return [centroid] + reordered_corners

    def _convert_to_centerpose_camera_data(self, camera_info_ros: dict) -> dict:
        """
        Converts ROS CameraInfo dict to the CenterPose camera_data format,
        adjusting the projection matrix and adding a view matrix for the
        target coordinate system (-z front, x right, -y down).
        """
        centerpose_camera_data = {}

        # --- Intrinsics ---
        cm = camera_info_ros['camera_matrix']['data'] # Flattened 3x3 K matrix
        intrinsics = {
            'fx': cm[0], 'fy': cm[4], 'cx': cm[2], 'cy': cm[5]
        }

        # --- Original Projection Matrix (P2) ---
        # This projects from KITTI's camera coordinates (z-front, x-right, y-down)
        pm_kitti = np.array(camera_info_ros['projection_matrix']['data']).reshape(3, 4)

        # --- Adjust Projection Matrix for Target Coordinates ---
        # Target system: x'=x, y'=-y, z'=-z  (where x,y,z are KITTI coords)
        # Original P2 maps [x, y, z, 1]_kitti -> [u, v, w]_img
        # We need P_target such that P_target * [x', y', z', 1]_target -> [u, v, w]_img
        # Since x=x', y=-y', z=-z', we substitute into the original mapping:
        # P2 * [x', -y', -z', 1] should give the same result.
        # P_target = P2 * T_kitti_from_target_homogeneous
        # T_kitti_from_target_homogeneous = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        # This means: P_target[:, 0] = P2[:, 0]
        #             P_target[:, 1] = -P2[:, 1]
        #             P_target[:, 2] = -P2[:, 2]
        #             P_target[:, 3] = P2[:, 3]
        pm_target = pm_kitti.copy()
        pm_target[:, 1] *= -1.0 # Negate second column
        pm_target[:, 2] *= -1.0 # Negate third column

        # Pad to 4x4 for the 'camera_projection_matrix' field
        camera_projection_matrix_4x4 = np.identity(4)
        camera_projection_matrix_4x4[:3, :] = pm_target

        # --- View Matrix ---
        # This transforms points from the *target* camera coordinates to the *world* coordinates.
        # If we assume the initial KITTI camera pose IS the world origin (identity transform),
        # then the view matrix should be the inverse of the transform from world(kitti) to target camera.
        # Transform from World(KITTI) to Target: T_target_from_kitti
        # Inverse transform (Target to World/KITTI): T_kitti_from_target
        # View Matrix = T_kitti_from_target (padded to 4x4)
        camera_view_matrix_4x4 = np.identity(4)
        camera_view_matrix_4x4[:3, :3] = self.T_kitti_from_target # T_kitti_from_target == T_target_from_kitti

        # --- Populate Dictionary ---
        centerpose_camera_data.update({
            'width': camera_info_ros['image_width'],
            'height': camera_info_ros['image_height'],
            # Assuming camera is at the origin of the world frame defined by KITTI's initial pose
            'location_world': [0.0, 0.0, 0.0],
            'quaternion_xyzw_worldframe': [0.0, 0.0, 0.0, 1.0], # No rotation relative to world origin
            'intrinsics': intrinsics,
            'camera_projection_matrix': camera_projection_matrix_4x4.tolist(),
            'camera_view_matrix': camera_view_matrix_4x4.tolist()
        })
        return centerpose_camera_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='kitti_to_centerpose_converter.py',
        description='Converts KITTI 3D Object Detection Training Dataset to a CenterPose-compatible format '
                    'with a specific target coordinate system (-z front, x right, -y down).')

    parser.add_argument('--kitti-dir',
                        required=True,
                        help='Path to KITTI dataset root directory (e.g., containing training/testing).',
                        metavar="DIR",
                        type=str)
    parser.add_argument('--output-dir',
                        required=True,
                        help='Path to store converted dataset.',
                        metavar="DIR",
                        type=str)
    parser.add_argument('--distance-in-cm',
                        action='store_true', # Use action='store_true' for boolean flags
                        help='If set, output distances (location, scale, keypoints_3d) in centimeters instead of meters.')
    parser.add_argument('--save-camera-info',
                        action='store_true',
                        help='If set, stores ROS CameraInfo yaml files for each frame.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='If set, allows overwriting files in the output directory.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='If set, draws projected 3D cuboids on the output images.')

    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.kitti_dir):
        print(f"Error: KITTI directory not found: {args.kitti_dir}")
        sys.exit(1)

    # Create output dir if it doesn't exist (init checks this too, but good practice)
    os.makedirs(args.output_dir, exist_ok=True)

    converter = KITTI3DToFATConverter(args)
    converter.run()