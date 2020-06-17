import numpy as np
import cv2
import os
import os.path as osp

file_path = 'png_label/'
result_path = 'SEG/'

image_names = sorted([osp.join(file_path, image_name) for image_name in os.listdir(file_path)])

for img_name in image_names:
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    _, markers = cv2.connectedComponents(img, connectivity=4)
    cv2.imwrite(osp.join(result_path, 'man_seg'+img_name[-7:-4]+'.tif'), markers.astype(np.uint16))