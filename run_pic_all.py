#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import os
import argparse
import cv2
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator

parser = argparse.ArgumentParser()
parser.add_argument('--imagedir', type=str, default='run_around_mask')
parser.add_argument('--outdir', type=str, default='output2')
args = parser.parse_args()

ROOT_DIR = os.path.abspath("./")
IMAGE_DIR = os.path.join(ROOT_DIR, args.imagedir)
OUT_DIR = os.path.join(ROOT_DIR, args.outdir)

file_names = next(os.walk(IMAGE_DIR))[2]
path_w = os.path.join(OUT_DIR, 'pelvis.txt')
path_p = os.path.join(OUT_DIR, 'pos_vnect.txt')
pelvis_text = ''
vnect_text = ''

box_size = 368
hm_factor = 8
joints_num = 21
scales = [1.0, 0.7]
joint_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8,
                 9, 14, 11, 12, 14, 14, 1, 4, 7, 10,
                 13]
estimator = VNectEstimator()

for file_name in sorted(file_names):

    img = cv2.imread(os.path.join(IMAGE_DIR, file_name))
    x, y, w, h = (0, 0, img.shape[1], img.shape[0])

    img_cropped = img
    joints_2d, joints_3d = estimator(img_cropped)
    
    py, px = joints_2d[14]
    
    rsx, rsy = joints_2d[2]
    lsx, lsy = joints_2d[5]
    
    shoulder_width = math.sqrt((rsx-lsx)**2 + (rsy-lsy)**2)

    pelvis_text += str(px) + ',' + str(py) + ',' + str(shoulder_width) + '\n'
    
    for i in range(joints_num):
        vnect_text += str(i) + ' ' + str(joints_3d[i][0]) + ' ' + str(joints_3d[i][1]) + ' ' + str(joints_3d[i][2])+', '

    vnect_text += '\n'

    print(file_name)

    # 2d plotting
    joints_2d[:, 0] += y
    joints_2d[:, 1] += x
    img_draw = utils.draw_limbs_2d(img.copy(), joints_2d, joint_parents, [x, y, w, h])
    cv2.imwrite(os.path.join(OUT_DIR, file_name), img_draw)

with open(path_w, mode='w') as f:
    f.write(pelvis_text)

with open(path_p, mode='w') as f:
    f.write(vnect_text)  
