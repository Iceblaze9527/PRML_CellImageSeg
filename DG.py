import numpy as np
import cv2
import os
import os.path as osp
from torch.utils.data import Dataset

import elasticdeform as ED

class DatasetGen(Dataset):
    def __init__(self, train_path, gt_path, transform):
        super(DatasetGen, self).__init__()
        self.train_names = self.get_file_names(train_path)
        self.gt_names = self.get_file_names(gt_path) 
        self.transform = transform

    def __getitem__(self, index):     
        train_image = self.img_standardization(cv2.imread(self.train_names[index],-1))#转为8位3通道
        gt_image = self.unit16b2uint8(cv2.imread(self.gt_names[index],-1))#转为8位
        
        train_image = self.trans(train_image)
        gt_image = self.trans(gt_image)
        
        if self.transform is not None:
            train_image = self.transform(train_image)
        gt_image = (np.where(gt_image == 0,0,1)).astype(np.uint8)#二值化
        
        return train_image, gt_image
    
    def __len__(self):
        return len(self.train_names)

    def trans(self, img):
        return img
    
    def img_standardization(self, img):
        img = self.unit16b2uint8(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.tile(img, (1, 1, 3))
            return img
        elif len(img.shape) == 3:
            return img
        else:
            raise TypeError('The Depth of image large than 3 \n')
    
    def get_file_names(self, file_path):
        return sorted([osp.join(file_path, image_name) for image_name in os.listdir(file_path)])
    
    def unit16b2uint8(self, img):
        if img.dtype == 'uint8':
            return img
        elif img.dtype == 'uint16':
            return img.astype(np.uint8)
        else:
            raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))


class DatasetHGen(DatasetGen):#Horizontal
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        return cv2.flip(img, 1)
    
class DatasetVGen(DatasetGen):#Vertical
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        return cv2.flip(img, 0)

class DatasetHVGen(DatasetGen):#Horizontal + Vertical
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        return cv2.flip(img, -1)
    
class DatasetR90Gen(DatasetGen):
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        img = cv2.flip(cv2.transpose(img), 0)
        return img

class DatasetR270Gen(DatasetGen):
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        img = cv2.flip(cv2.transpose(img), 1)
        return img
    
class DatasetTPGen(DatasetGen):
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        img = cv2.transpose(img)
        return img

class DatasetSTPGen(DatasetGen):
    def __init__(self, train_path, gt_path, transform):
        super().__init__(train_path, gt_path, transform)
        
    def trans(self, img):
        img = cv2.transpose(cv2.flip(img, -1))
        return img

class DatasetEDGen(DatasetGen):
    def __init__(self, train_path, gt_path, transform, sigma, points, order):
        super().__init__(train_path, gt_path, transform)
        self.sigma = sigma
        self.points = points
        self.order = order
    
    def __getitem__(self, index):
        train_image = super().img_standardization(cv2.imread(self.train_names[index],-1))
        gt_image = super().unit16b2uint8(cv2.imread(self.gt_names[index],-1))
        gt_image = (np.where(gt_image == 0,0,1)).astype(np.uint8)

        #ElasticDeformation
        [train_image, gt_image] = ED.deform_random_grid([train_image, gt_image], 
                                                sigma=self.sigma, points=self.points, order=self.order, axis=[(0,1), (0,1)])

        if self.transform is not None:
            train_image = self.transform(train_image)

        return train_image, gt_image
