
# -*- coding: UTF-8 -*-


import cv2
import numpy as np
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator

in_dir = 'pic/'
file_name = 'test_pic.jpg'
out_dir = 'output/'
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
#hog = HOGBox()
#hog.clicked = True
#choose, rect = hog(img.copy())
x, y, w, h = (0, 0, 368, 538)
#print(rect)
#img_cropped = img[y: y + h, x: x + w, :]
img_cropped = img
joints_2d, joints_3d = estimator(img_cropped)
print('2D')
print(joints_2d)
print('3D')
print(joints_3d)
# 3d plotting
#utils.draw_limbs_3d(joints_3d, joint_parents)
# 2d plotting
joints_2d[:, 0] += y
joints_2d[:, 1] += x
img_draw = utils.draw_limbs_2d(img.copy(), joints_2d, joint_parents, [x, y, w, h])
#cv2.imshow('2D Prediction', img_draw)
cv2.imwrite(out_dir + out_name, img_draw)


#cv2.waitKey()
#cv2.destroyAllWindows()
