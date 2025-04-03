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
    to [Falling Things Format]
    (https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt)
    """

    def __init__(self, args):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            assert len(os.listdir(args.output_dir)) == 0, 'Error: Output dir must be empty.'

        self.images = None
        self.camera_calibration = None
        self.objects = None

    def run(self):
        self.images = self._get_files_by_format(args.kitti_dir, '.png')
        assert len(self.images) > 0, "Error: No png files in {}.".format(
            args.kitti_dir)

        for i, image_filename in enumerate(self.images):
            frame_id = image_filename.split('/')[-1][:-4]

            self.camera_calibration = self._parse_calibration_file(frame_id)
            ground_truth_KITTI = self._parse_ground_truth_file(frame_id)
            self.objects = self._convert_objects(ground_truth_KITTI,
                                                 self.camera_calibration)

            # filter for cars only
            self.objects = [obj for obj in self.objects if
                            obj['class'] == 'car']

            # store dataset
            if len(self.objects) == 0:
                continue
            shutil.copy(image_filename, args.output_dir)
            camera_info = self._convert_to_camera_info(self.camera_calibration)
            centerpose_camera_data = self._convert_to_centerpose_camera_data(camera_info)
            self._dump_annotation_file(self.objects, centerpose_camera_data, os.path.join(args.output_dir, frame_id + '.json'))

            if args.save_camera_info:
                with open(os.path.join(args.output_dir, frame_id + '.yml'), "w") as write_file:
                    yaml.dump(camera_info, write_file, default_flow_style=None,
                              sort_keys=False)

            if args.debug:
                self._draw_cuboid_points(
                    image_path=os.path.join(args.output_dir, frame_id + '.png'))

            sys.stdout.write("Converted {}/{} items.\r".format(i + 1, len(self.images)))
            sys.stdout.flush()

    def _get_files_by_format(self, directory: str, ending: str) -> list:
        """
        Returns all files in given dir with given ending.

        Parameters
        ----------
        directory : Directory to search for given files.
        ending : Ending of files to return.

        Returns
        -------
        list of strings containing all files with given ending in given dir.
        """

        dir_list = [root for root, dirs, _ in os.walk(directory)]
        files = []
        for annotation in dir_list:
            files.extend(glob(annotation + "/*" + ending))
        return files

    def _convert_to_camera_info(self, camera_calibration: dict) -> dict:
        """
        Converts given camera_calibration information to camera_info.yaml (ROS CameraInfo) format.

        Parameters
        ----------
        camera_calibration : Camera calibration information in KITTI format.
        """

        rectification_matrix = camera_calibration['R0_rect'].tolist()
        projection_matrix = camera_calibration['P2'].tolist()

        # get camera matrix from projection matrix (left 3x3 slice of matrix)
        camera_matrix = np.append(projection_matrix[0:3],
                                  np.append(projection_matrix[4:7],
                                            projection_matrix[8:11])).tolist()

        camera_info = {
            "image_width": 1238,
            "image_height": 370,
            "camera_name": 'front_left',
            "distortion_model": "plumb_bob",
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": camera_matrix
            },
            "distortion_coefficients": {
                "rows": 1,
                "cols": 5,
                "data": [0, 0, 0, 0, 0]
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": rectification_matrix
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": projection_matrix
            },
        }
        return camera_info

    def _dump_annotation_file(self, objects: list, camera_data: dict, filename: str) -> None:
        """
        Stores objects ground truth information in FAT format as json file.
        Parameters
        ----------
        objects : List containing object information in FAT format.
        camera_data: Dict containing camera information in centerpose format.
        filename : Path to store json file.
        """

        output = {    "AR_data": {
            "plane_center": [ 0, 0, 0 ],
            "plane_normal": [ 0, 0, 0]
        },
            'camera_data': camera_data,
                  'objects': objects}

        with open(filename, "w") as write_file:
            json.dump(output, write_file, indent=4)

    def _draw_cuboid_points(self, image_path: str) -> None:
        """
        Draws projected cuboid of objects on given image.

        Parameters
        ----------
        image_path : Path to image.
        """

        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        with Image.open(image_path) as im:
            d = ImageDraw.Draw(im)
            colors = ['red', 'green', 'yellow', 'blue']
            for obj in self.objects:
                color = random.choice(colors)
                projected_cuboid = obj.get('projected_cuboid')
                points = [(int(point[0]), int(point[1])) for point in
                          projected_cuboid]

                d.line([points[0], points[1]], fill=color, width=2)
                d.line([points[1], points[2]], fill=color, width=2)
                d.line([points[2], points[3]], fill=color, width=2)
                d.line([points[3], points[0]], fill=color, width=2)

                d.line([points[4], points[5]], fill=color, width=2)
                d.line([points[5], points[6]], fill=color, width=2)
                d.line([points[6], points[7]], fill=color, width=2)
                d.line([points[7], points[4]], fill=color, width=2)

                d.line([points[0], points[4]], fill=color, width=2)
                d.line([points[1], points[5]], fill=color, width=2)
                d.line([points[2], points[6]], fill=color, width=2)
                d.line([points[3], points[7]], fill=color, width=2)

                d.line([points[0], points[2]], fill=color, width=2)
                d.line([points[1], points[3]], fill=color, width=2)

                # centroid
                d.point(obj.get('projected_cuboid_centroid'), fill=color)
                d.text(obj.get('projected_cuboid_centroid'), text="C", font=fnt,
                       fill=color)

            im.save(image_path)

    def _parse_calibration_file(self, frame: str) -> dict:
        """
        Parses sensor calibration data of given frame into dict.

        Parameters
        ----------
        frame : str: ID of image associated with calibration info.

        Returns
        -------
        dict: Sensor calibration data.
        """

        calibration_file = os.path.join(args.kitti_dir, 'calib', frame + '.txt')
        calibration = {}
        with open(calibration_file, 'r') as file:
            for line in file.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calibration[key] = np.array(
                        [float(x) for x in value.split()])
        return calibration

    def _parse_ground_truth_file(self, frame: str) -> dict:
        """
        Parses ground truth data in KITTI format of given frame into dict.
        Details for ground truth format:
        https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb

        Parameters
        ----------
        frame : str: ID of image associated with ground truth.

        Returns
        -------
        dict: Ground truth data in KITTI format.
        """
        ground_truth_file = os.path.join(args.kitti_dir, 'label_2',
                                         frame + '.txt')
        ground_truth = {}
        with open(ground_truth_file, 'r') as file:
            for line in file.readlines():
                if len(line) > 3:
                    key, value = line.split(' ', 1)
                    if key in ground_truth.keys():
                        ground_truth[key].append(
                            [float(x) for x in value.split()])
                    else:
                        ground_truth[key] = [[float(x) for x in value.split()]]

        for key in ground_truth.keys():
            ground_truth[key] = np.array(ground_truth[key])

        return ground_truth

    def _compute_cuboid(self, location: np.array, dimensions: np.array,
                        rotation_matrix: np.array) -> np.array:
        """
        Parameters
        ----------
        location : [x, y, z] location in camera space.
        dimensions : [height, width, length] dimensions of object/cuboid.
        rotation_matrix: Rotation around y-axis as matrix.

        Returns
        -------
        """

        height = dimensions[0]
        width = dimensions[1]
        length = dimensions[2]

        # bounding box corner points in object coordinates
        x_corners = [0, length, length, length, length, 0, 0, 0]
        y_corners = [0, 0, height, height, 0, 0, height, height]
        z_corners = [0, 0, 0, width, width, width, width, 0]

        x_corners += -length / 2.0
        y_corners += -height
        z_corners += -width / 2.0

        cuboid = np.array([x_corners, y_corners, z_corners])

        # rotate
        cuboid = rotation_matrix.dot(cuboid)

        # translate
        cuboid += location.reshape((3, 1))

        return cuboid

    def _project_cuboid(self, cuboid: np.array, projection_matrix: np.array):
        """
        https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb

        Parameters
        ----------
        label :
        projection_matrix :

        Returns
        -------

        """

        cuboid_stacked = np.vstack((cuboid, np.ones((cuboid.shape[-1]))))
        cuboid_2D = projection_matrix.dot(cuboid_stacked)
        cuboid_2D = cuboid_2D / cuboid_2D[2]

        return cuboid_2D[:2]

    def _convert_objects(self, ground_truth: dict,
                         sensor_calibration: dict) -> list:
        """
        Converts given KITTI ground truth labels to FAT format.
        For details on FAT format see:
        https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt

        Parameters
        ----------
        ground_truth : Dict containing ground truth information parsed from
                        KITTI labels file.
        sensor_calibration : Dict containing sensor calibration parsed from
                            KITTI  calib file.

        Returns
        -------
        List of object dictionaries containing object information in FAT format.
        """

        #rectification_matrix = np.zeros((4, 4))
        #rectification_matrix[:3, :3] = sensor_calibration['R0_rect'].reshape(3,
        #                                                                     3)
        #rectification_matrix[3, 3] = 1
        projection_matrix = sensor_calibration['P2'].reshape(3, 4)

        kitti_occlusion_to_fat_occlusion = {0: 0.0, 1: 0.25, 2: 0.75, 3: 1.0}
        objects = []

        for class_id in ground_truth.keys():

            color = 'white'
            if class_id == 'Car':
                color = 'red'
            elif class_id == 'Pedestrian':
                color = 'pink'
            elif class_id == 'Cyclist':
                color = 'purple'
            elif class_id == 'DontCare':
                color = 'white'

            for i in range(ground_truth[class_id].shape[0]):
                object_dict = {"class": class_id.lower(), "name": class_id.lower(), "provenance": "objectron",}
                occlusion = max(0, int(ground_truth[class_id][i][1]))
                occlusion = kitti_occlusion_to_fat_occlusion.get(occlusion,0)
                truncation = float(ground_truth[class_id][i][0])
                visibility = 1.0 - truncation
                visibility = max(0, visibility - occlusion)

                # 2D Bounding Box
                left = ground_truth[class_id][i][3]
                top = ground_truth[class_id][i][4]
                right = ground_truth[class_id][i][5]
                bottom = ground_truth[class_id][i][6]

                object_dict.update({'visibility': visibility,
                                    'truncation': int(ground_truth[class_id][i][0]),
                                    'occlusion': float(ground_truth[class_id][i][1]),
                                    'bounding_box': {
                                        'top_left': [left, top],
                                        'bottom_right': [right, bottom]}})

                if class_id != 'DontCare':
                    gt_object = ground_truth[class_id][i]
                    dimensions = np.array(
                        [gt_object[7], gt_object[8], gt_object[9]])
                    location = np.array(
                        [gt_object[10], gt_object[11], gt_object[12]])

                    rotation_y = gt_object[13]

                    axis = np.array([0, 1, 0])  # y-axis
                    rotation_vector = axis * float(rotation_y)
                    rotation = Rotation.from_rotvec(rotation_vector)
                    #rotation = Rotation.from_euler("XYZ", [0, rotation_y, 0])

                    cuboid = self._compute_cuboid(location, dimensions,
                                                  rotation.as_matrix())
                    cuboid_2D = self._project_cuboid(cuboid, projection_matrix)
                    cuboid = self._reorder_cuboid_points(cuboid)

                    if args.distance_in_cm:
                        cuboid *= 100

                    projected_cuboid_dope = self._reorder_cuboid_points(
                        cuboid_2D).T.tolist()
                    centroid =  [np.average(cuboid_2D[0,]), np.average(cuboid_2D[1,])]
                    projected_cuboid_centerpose = self._reorder_cuboid_for_centerpose(projected_cuboid_dope,centroid)

                    cuboid_dope = cuboid.T.tolist()
                    location = [np.average(cuboid[0]), np.average(cuboid[1]),
                                 np.average(cuboid[2])]
                    cuboid_centerpose = self._reorder_cuboid_for_centerpose(cuboid_dope, location)


                    object_dict.update({
                                           'projected_cuboid': projected_cuboid_centerpose,
                                           'keypoints_3d': cuboid_centerpose,
                                           'location': location,
                                           'quaternion_xyzw': list(Rotation.as_quat(rotation)),
                                           'scale': [dimensions[1], dimensions[0], dimensions[2]]
                                           })
                objects.append(object_dict)

        return objects

    def _reorder_cuboid_for_centerpose(self, fat_cuboid: list, centroid: list) -> list:
        return [centroid,
                fat_cuboid[7],
                fat_cuboid[3],
                fat_cuboid[4],
                fat_cuboid[0],
                fat_cuboid[6],
                fat_cuboid[2],
                fat_cuboid[5],
                fat_cuboid[1]]

    def _calculate_quaternion(self, rotation: Rotation) -> list:
        """
        Calculates FAT rotation (FAT  coordinates are -y-up and z-front, right
        hand) as quaternion from given KITTI rotation (Kitti coordinates are
        -y-up and x-front, right hand).

        First apply rotation around y-axis (Kitti GT), second rotate 90° around
        y-axis to have FAT rotation.

        Parameters
        ----------
        rotation : Rotation according KITTI coordinate system: y-up and z-front, right
        hand.

        Returns
        -------
        Quaternion as list according to FAT coordinate system: y-up and x-front, right
        hand.
        """
        kitti_to_DOPE_rotation = Rotation.from_euler('xyz', [0, 90, 0],
                                                     degrees=True)
        return list((kitti_to_DOPE_rotation * rotation).as_quat())

    def _reorder_cuboid_points(self, cuboid_KITTI: np.array) -> np.array:
        """
        Reorders points of given cuboid in KITTI-order to cuboid in FAT-order.
        For details on FAT-order see:
        https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt

        Parameters
        ----------
        cuboid_KITTI : cuboid with points ordered in KITTI-order.

        Returns
        -------
        Cuboid in FAT-order.

        """
        cuboid_FAT = cuboid_KITTI.T
        return np.array([cuboid_FAT[1],
                         cuboid_FAT[4],
                         cuboid_FAT[3],
                         cuboid_FAT[2],
                         cuboid_FAT[0],
                         cuboid_FAT[5],
                         cuboid_FAT[6],
                         cuboid_FAT[7]]).T

    def _convert_to_centerpose_camera_data(self, camera_info):
        centerpose_camera_data = {}
        pm = camera_info['projection_matrix']['data']
        centerpose_camera_data.update({'width': camera_info['image_width'],
                                       'height': camera_info['image_height'],
                                       'location_world': [ 0, 0, 0],
                                       'quaternion_xyzw_worldframe': [0, 0, 0, 1],
                                       'intrinsics': {
                                           'cx': camera_info['camera_matrix']['data'][2],
                                           'cy': camera_info['camera_matrix']['data'][5],
                                           'fx': camera_info['camera_matrix']['data'][0],
                                           'fy': camera_info['camera_matrix']['data'][4]
                                       },
                                       'camera_projection_matrix': [
                                        [pm[0],pm[1],pm[2],pm[3]],
                                        [pm[4],pm[5],pm[6],pm[7]],
                                        [pm[8],pm[9],pm[10],pm[11]],
                                        [0, 0, 0, 1]
                                       ],
                                       'camera_view_matrix': [
                                       [ 1, 0, 0, 0 ],
                                       [ 0, 1, 0, 0 ],
                                       [ 0, 0, 1, 0 ],
                                       [ 0, 0, 0, 1 ]]

                                       })
        return centerpose_camera_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='KITTI_to_FAT_converter.py',
        description='Converts given KITTI 3D Object Detection Training Dataset to Falling Things Format.')

    parser.add_argument('--kitti-dir',
                        required=True,
                        help='Path to KITTI root directory.',
                        metavar="DIR",
                        type=str)
    parser.add_argument('--output-dir',
                        required=True,
                        help='Path to store converted dataset.',
                        metavar="DIR",
                        type=str)
    parser.add_argument('--distance-in-cm',
                        default=False,
                        help='Distance unit for output dataset is centimeters.',
                        metavar="",
                        type=bool)
    parser.add_argument('--save-camera-info',
                        default=False,
                        help='Stores CameraInfo yaml files for each frame.',
                        metavar="",
                        type=bool)
    parser.add_argument('--debug',
                        default=False,
                        metavar="",
                        help='Draws cuboids on the output image.',
                        type=bool)
    args = parser.parse_args()

    KITTI3DToFATConverter(args).run()
