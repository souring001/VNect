#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import cv2
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='pic/test_pic.jpg')
args = parser.parse_args()

in_dir = args.image.split('/')[0] + '/'
file_name = args.image.split('/')[-1]
out_dir = 'output2/'
out_name = file_name.split('.')[0] + '_vnect.' + file_name.split('.')[1]

box_size = 368
hm_factor = 8
joints_num = 21
scales = [1.0, 0.7]
joint_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8,
                 9, 14, 11, 12, 14, 14, 1, 4, 7, 10,
                 13]
estimator = VNectEstimator()
img = cv2.imread(in_dir + file_name)
x, y, w, h = (0, 0, img.shape[1], img.shape[0])

img_cropped = img
joints_2d, joints_3d = estimator(img_cropped)

# print('2D')
# print(joints_2d)
# print('3D')
# print(joints_3d)

print(joints_3d.shape)

# 2d plotting
joints_2d[:, 0] += y
joints_2d[:, 1] += x
img_draw = utils.draw_limbs_2d(img.copy(), joints_2d, joint_parents, [x, y, w, h])
cv2.imwrite(out_dir + out_name, img_draw)
