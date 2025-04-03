#!/usr/bin/env python3
import os
import time
import shutil
from glob import glob
import random


class DatasetSplitter:

    def __init__(self):
        self.dataset_base_dir = '/home/lore_be/data/Kitti3D_FAT/kitti_centerpose_cars_only_truncate_occlusion_rotation_bbox/raw/'
        self.testset_dir = '/home/lore_be/data/Kitti3D_FAT/kitti_centerpose_cars_only_truncate_occlusion_rotation_bbox/test'
        self.trainset_dir = '/home/lore_be/data/Kitti3D_FAT/kitti_centerpose_cars_only_truncate_occlusion_rotation_bbox/train'
        self.valset_dir = '/home/lore_be/data/Kitti3D_FAT/kitti_centerpose_cars_only_truncate_occlusion_rotation_bbox/val'
        self.filename_list = self.create_filename_list()

        self.dataset_size = len(self.filename_list)
        self.testset_size = int(0.1 * self.dataset_size)
        self.valset_size = int(0.2 * self.dataset_size)
        self.trainset_size = int(0.7 * self.dataset_size)

        self.current_size_of_testset = 0
        self.current_size_of_valset = 0
        self.current_size_of_trainset = 0


    def run(self):
        # fill valset and trainset
        testset_list = []
        valset_list = []
        trainset_list = []

        print('Choose random files to create lists for training and validation set...')
        while self.current_size_of_testset < self.testset_size and len(self.filename_list) > 0:
            random_file = random.choice(self.filename_list)
            self.filename_list.remove(random_file)
            if os.path.exists(random_file[:-3] + "json"):
                testset_list.append(random_file[:-3])
                self.current_size_of_testset += 1

        while self.current_size_of_valset < self.valset_size and len(self.filename_list) > 0:
            random_file = random.choice(self.filename_list)
            self.filename_list.remove(random_file)
            if os.path.exists(random_file[:-3] + "json"):
                valset_list.append(random_file[:-3])
                self.current_size_of_valset += 1

        while self.current_size_of_trainset < self.trainset_size and len(self.filename_list) > 0:
            random_file = random.choice(self.filename_list)
            self.filename_list.remove(random_file)
            if os.path.exists(random_file[:-3] + "json"):
                trainset_list.append(random_file[:-3])
                self.current_size_of_trainset += 1

        # save files in lists
        for i,random_filename_without_ending in enumerate(testset_list):
            print_list = []
            if os.path.exists(random_filename_without_ending + 'png'):
                shutil.copyfile(random_filename_without_ending + 'png', '{}/{:06d}.png'.format(self.testset_dir, i))
            elif os.path.exists(random_filename_without_ending + 'jpg'):
                shutil.copyfile(random_filename_without_ending + 'jpg', '{}/{:06d}.jpg'.format(self.testset_dir, i))
            else:
                print(f"Warning: Image not found for {random_filename_without_ending}")
                continue
            shutil.copyfile(random_filename_without_ending + 'json', '{}/{:06d}.json'.format(self.testset_dir, i))
            print_list.append(random_filename_without_ending)
            if i % 100 == 0:
                print('{}/{}: saved {} to test_set. \n'.format(i, self.valset_size, print_list))

        for i,random_filename_without_ending in enumerate(valset_list):
            print_list = []
            if os.path.exists(random_filename_without_ending + 'png'):
                shutil.copyfile(random_filename_without_ending + 'png', '{}/{:06d}.png'.format(self.valset_dir, i))
            elif os.path.exists(random_filename_without_ending + 'jpg'):
                shutil.copyfile(random_filename_without_ending + 'jpg', '{}/{:06d}.jpg'.format(self.valset_dir, i))
            else:
                print(f"Warning: Image not found for {random_filename_without_ending}")
                continue
            shutil.copyfile(random_filename_without_ending + 'json', '{}/{:06d}.json'.format(self.valset_dir, i))
            print_list.append(random_filename_without_ending)
            if i % 100 == 0:
                print('{}/{}: saved {} to validation_set. \n'.format(i, self.valset_size, print_list))

        for i, random_filename_without_ending in enumerate(trainset_list):
            print_list = []
            if os.path.exists(random_filename_without_ending + 'png'):
                shutil.copyfile(random_filename_without_ending + 'png', '{}/{:06d}.png'.format(self.trainset_dir, i))
            elif os.path.exists(random_filename_without_ending + 'jpg'):
                shutil.copyfile(random_filename_without_ending + 'jpg', '{}/{:06d}.jpg'.format(self.trainset_dir, i))
            else:
                print(f"Warning: Image not found for {random_filename_without_ending}")
                continue
            shutil.copyfile(random_filename_without_ending + 'json', '{}/{:06d}.json'.format(self.trainset_dir, i))
            print_list.append(random_filename_without_ending)
            if i % 100 == 0:
                print('{}/{}: saved {} to training_set. \n'.format(i, self.trainset_size, print_list))


    def create_filename_list(self):
        dir_list = [root for root,dirs,_ in os.walk(self.dataset_base_dir)]
        filename_list = []
        for image_dir in dir_list:
            filename_list.extend(glob(image_dir + "/*.png"))
            filename_list.extend(glob(image_dir + "/*.jpg"))
        return filename_list

if __name__ == '__main__':
    DatasetSplitter().run()
    print('\ndone.')