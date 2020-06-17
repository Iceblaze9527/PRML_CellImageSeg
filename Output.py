import numpy as np
import cv2
import os
import os.path as osp

import torch
import torchvision.transforms.functional as trfunc
from UNet import UNet

from skimage.filters import threshold_multiotsu
from skimage.morphology import watershed

import matplotlib.pyplot as plt

def img_standardization(img):
    img = unit16b2uint8(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1, 1, 3))
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')

def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))
        
def load_images(test_path, result_path, pad):
    test_names = sorted([osp.join(test_path, image_name) for image_name in os.listdir(test_path)])
    test_images = {}

    for index in range(len(test_names)):
        test_image = img_standardization(cv2.imread(test_names[index],-1))
        test_image = trfunc.to_pil_image(test_image)
        test_image = trfunc.to_grayscale(test_image, 1)
        test_image = trfunc.pad(test_image, pad)
        test_image = trfunc.to_tensor(test_image)
        test_images[test_names[index]] = test_image
        
    return test_images
        
def post_processing(pred_act, connectivity=4):
    #Otsu三分类
    thresholds = threshold_multiotsu(pred_act, classes=3)
    pred_tri = np.digitize(pred_act, bins=thresholds)
    
    #watershed会把图像边界标为-1，因此需要扩展边界
    pred_canvas = np.zeros((pred_tri.shape[0] + 2, pred_tri.shape[1] + 2)).astype(np.uint8)
    pred_canvas[1:-1, 1:-1] = pred_tri
    
    pred_1 = np.where(pred_canvas==1,1,0).astype(np.int32) 
    pred_2 = np.where(pred_canvas==2,1,0).astype(np.uint8)
    pred_bin = np.where(pred_canvas>=1,1,0).astype(np.uint8)
    
    #获取连通域marker
    _, markers = cv2.connectedComponents(pred_2, connectivity=connectivity)
    markers = markers + 1
    markers[pred_1 == 1] = 0 
    
    #opencv的watershed只支持8位3通道输入
    pred_act_wt = (255 * cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
    markers = cv2.watershed(pred_act_wt, markers)
    
    #最终处理
    pred_final = markers[1:-1,1:-1] - 1
    pred_final[pred_final == -2] = 0
    pred_final = pred_final.astype(np.uint16)
    
    return pred_final

def test_out(best_model, test_path, result_path, pad):
    test_images = load_images(test_path, result_path, pad)

    best_model.eval()
    with torch.no_grad():
        for name, X in test_images.items():
            X = X.unsqueeze(0)# [N, 1, H, W]
            pred = best_model(X)# [N, 2, H, W]

            pred = pred[:, :, pad:-pad, pad:-pad] 
            pred_act = (torch.sigmoid(pred).detach().squeeze().numpy())[0]
            if (np.median(pred_act) > np.mean(pred_act)):
                pred_act = 1 - pred_act
            
            pred_final = post_processing(pred_act)

            print('X')
            plt.imshow(X.detach().squeeze().numpy())
            plt.show()
            print('pred_act')
            plt.imshow(pred_act)
            plt.show()
            print('pred_final')
            plt.imshow(pred_final)
            plt.show()
            plt.close()

            cv2.imwrite(osp.join(result_path, 'mask'+name[-7:]), pred_final.astype(np.uint16))