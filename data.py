# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:29:29 2019

@author: 97555
"""

import numpy as np
import pandas as pd
import os
import glob
import cv2
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    #图像预处理：resize to 128x128 通过crop,padding把字符放在图中央
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < W - 13) else W
    ymax = ymax + 10 if (ymax < H - 10) else H
    img = img0[ymin:ymax,xmin:xmax]
    
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))
    
def par2img(par_list,h=137,w=236,size=128):
    #读取Parquet文件，预处理图像，保存至本地
    if not os.path.isdir('../train_img'):
        os.makedirs('../train_img')
    count=0
    for par in par_list:
        df = pd.read_parquet(par)
        for index in range(len(df)):
            img = 255-df.iloc[index,1:].values.reshape(h,w).astype(np.uint8)
            img = (img*255.0/img.max()).astype(np.uint8)
            img = crop_resize(img)
            cv2.imwrite('../train_img/Train_'+str(count)+'.png',img)
            print(img.shape,count)
            count+=1
            
    print(count)
H=137
W=236
root_dir='../'

#par_list = glob.glob(root_dir+'train'+'*.parquet')
#train = pd.read_csv(root_dir+'train.csv')
#label = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
#par2img(par_list)

#############################下面是读取本地图片的dataset

#train_path = '../train_img/'
#train_images = sorted(glob.glob(train_path+'*.png'))
#train_labels_df = pd.read_csv(root_dir+'train.csv')
#train_labels = train_labels_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values


class BengaliDataset(Dataset):
    def __init__(self, data, transform=None):
        super(BengaliDataset, self).__init__
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
    
        x = cv2.imread(self.data[i][0],0)
        
        y = self.data[i][1]
        x = x.reshape(1,128,128)        
        x = torch.FloatTensor(x.astype(np.float32)/255.0)
        y = torch.FloatTensor(y)

        return x,y

#train_dataset = BengaliDataset(train_images,train_labels)
#image,label = train_dataset[0]
#print(image.shape,label)
