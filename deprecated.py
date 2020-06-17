"""
Credit to: https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
This is deprecated because of crashing

Generate the weight maps as specified in the UNet paper
for a set of binary masks.

Parameters
----------
masks: array-like
    A 3D array of shape (n_masks, image_height, image_width),
    where each slice of the matrix along the 0th axis represents one binary mask.

Returns
-------
array-like
    A 2D array of shape (image_height, image_width)

""" 
import torch
import numpy as np
from skimage.segmentation import find_boundaries
    
def weighted_map(y_num):
    w0=10
    sigma=5
    
    nrows, ncols = y_num.shape[1:]
    y_num = (y_num > 0).astype(int)
    distMap = np.zeros((nrows * ncols, y_num.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(y_num):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss_map = np.zeros((nrows, ncols))
    w_1 = 1 - y_num.sum() / loss_map.size
    w_0 = 1 - w_1
    loss_map[masks.sum(0) == 1] = w_1
    loss_map[masks.sum(0) == 0] = w_0
    y_w = xBLoss + loss_map
    ##
    
    y_new = y * y_w
    return torch.Tensor(y_new)

"""
This is the initial watershed algorithm, deprecated because of its poor performance
"""

def post_processing(pred_act, dkernel_size=1, mask_size=5, connectivity=4):
    #otsu 二值化
    threshold = threshold_otsu(pred_act)
    pred_bin = np.where(pred_act > threshold, 1, 0).astype(np.uint8)
    pred_canvas = np.zeros((pred_bin.shape[0] + 2, pred_bin.shape[1] + 2)).astype(np.uint8)
    pred_canvas[1:-1, 1:-1] = pred_bin
    
    #膨胀 = sure background
    kernel = np.ones((dkernel_size, dkernel_size),np.uint8)
    pred_dilated = cv2.dilate(pred_canvas, kernel, iterations = 1)
    
    #距离变换
    pred_dist = cv2.distanceTransform(pred_canvas, distanceType=cv2.DIST_L1, maskSize = mask_size).astype(np.uint8)
    #变换后二值化, 获得 sure foreground
    _, pred_dist_bin = cv2.threshold(pred_dist, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 获取未知部位
    pred_dis_bin = np.uint8(pred_dist_bin)
    unknown = cv2.subtract(pred_dilated, pred_dist_bin)
    
    #获取watershed marker
    _, markers = cv2.connectedComponents(pred_dist_bin, connectivity=connectivity)
    markers = markers + 1
    markers[unknown == 1] = 0
    
    #watershed
    pred_act_wt = (255 * cv2.cvtColor(pred_canvas, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
    markers = cv2.watershed(pred_act_wt, markers)
    
    # 最终处理
    pred_final = markers[1:-1, 1:-1] - 1
    pred_final[pred_final == -2] = 0
    pred_final = pred_final.astype(np.uint16)
 
    return pred_final