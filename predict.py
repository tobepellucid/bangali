# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:16:03 2019

@author: 97555
"""
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
from data import BengaliDataset

test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):

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

class GraphemeDataset(Dataset):#测试dataset
    def __init__(self, fname):
        print(fname)
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx,0]
        
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        img = img.astype(np.float32)/255.0
        return img, name

test_data = ['/kaggle/input/bengaliai-cv19/test_image_data_0.parquet','/kaggle/input/bengaliai-cv19/test_image_data_1.parquet','/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',
             '/kaggle/input/bengaliai-cv19/test_image_data_3.parquet']

%%time

row_id,target = [],[]
for fname in test_data:
    #data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')
    test_image = GraphemeDataset(fname)
    dl = torch.utils.data.DataLoader(test_image,batch_size=128,num_workers=4,shuffle=False)
    with torch.no_grad():
        for x,y in tqdm(dl):
            x = x.unsqueeze(1).float().cuda()
            p1,p2,p3 = model(x)
            p1 = p1.argmax(-1).view(-1).cpu()
            p2 = p2.argmax(-1).view(-1).cpu()
            p3 = p3.argmax(-1).view(-1).cpu()
            for idx,name in enumerate(y):
                row_id += [f'{name}_vowel_diacritic',f'{name}_grapheme_root',
                           f'{name}_consonant_diacritic']
                target += [p1[idx].item(),p2[idx].item(),p3[idx].item()]
                
sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
sub_df.to_csv('submission.csv', index=False)
sub_df.head(20)