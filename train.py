# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:16:03 2019

@author: 97555
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
import tqdm
import scipy.misc
import random
import cv2
import torchvision.transforms.functional as VF
import time
from data import BengaliDataset
from model import net
from model import Rnet_1ch
import pandas as pd
import torchvision.models as models



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

       
def accuracy(y, t):
    t = t.long()
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc

def split_data(images,labels):
    index = [i for i in range(len(images))]
    random.shuffle(index)
    
    train_data,val_data=[],[]
    for i,v in enumerate(index):
        if i<0.9*len(images):
            train_data.append((images[v],labels[v]))
        else:
            val_data.append((images[v],labels[v]))
    return train_data,val_data
    return train_images,train_labels,val_images,val_labels
    
class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        x1,x2,x3 = input
        y = target.long()
        return 2.0*F.cross_entropy(x1,y[:,0]) + F.cross_entropy(x2,y[:,1]) + \
          F.cross_entropy(x3,y[:,2])

def train(all_images,all_labels,batch_size,num_epochs,learning_rate):

    train_data,val_data = split_data(all_images,all_labels)
    
    n_grapheme=168
    n_vowel=11
    n_consonant=7
    n_total_class=186
    
    cuda = torch.cuda.is_available()
    model = net().cuda()
    

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)
    
    train_dataset = BengaliDataset(train_data)
    val_dataset = BengaliDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,shuffle=True)
    
    loss_minimal =100
    
    for epoch in range(0,num_epochs+1):
        
        model.train()
        tq = tqdm.tqdm(total = len(train_loader)*batch_size)
        tq.set_description('train_epoch %d, lr %f' % (epoch, learning_rate))
        
        loss_epoch=0
        acc_g=0
        acc_v=0
        acc_c=0
        for i, (img, label) in enumerate(train_loader):
            if cuda:
                img = img.cuda()
                label = label.cuda()
            
            optimizer.zero_grad()
            pred = model(img)
            
            loss = Loss_combine()(pred,label)
        
            
            tq.update(batch_size)
            tq.set_postfix(loss='%.3f'%(loss.item()))
            
            acc_grapheme = accuracy(pred[0],label[:,0])
            acc_vowel = accuracy(pred[1],label[:,1])
            acc_consonant = accuracy(pred[2],label[:,2])
            
            loss.backward()
            optimizer.step()
            loss_epoch+=loss.item()
            acc_g+=acc_grapheme.item()
            acc_v+=acc_vowel.item()
            acc_c+=acc_consonant.item()

        loss_epoch = loss_epoch/len(train_loader)
        
        acc_g = acc_g/len(train_loader)
        acc_v = acc_v/len(train_loader)
        acc_c = acc_c/len(train_loader)
        
        if loss_epoch<loss_minimal:
            loss_minimal = loss_epoch
            torch.save(model.state_dict(), 'model_resnet18_3head.pth')
        
        tq.close()
        print('epoch:%d ,loss for train: %f, acc_g: %f, acc_v: %f, acc_c: %f'%(epoch,loss_epoch,acc_g,acc_v,acc_c))
        #eval
        if epoch%5==0:
            model.eval()
            acc_epoch=0
            tq = tqdm.tqdm(total = len(val_loader)*batch_size)
            tq.set_description('val_epoch %d' % (epoch))
            for i, (img, label) in enumerate(val_loader):
                if cuda:
                    img = img.cuda()
                    label = label.cuda()
                
                pred = model(img)

                
                acc_grapheme = accuracy(pred[0],label[:,0])
                acc_vowel = accuracy(pred[1],label[:,1])
                acc_consonant = accuracy(pred[2],label[:,2])
                acc = acc_grapheme+acc_vowel+acc_consonant
                
                tq.update(batch_size)
                tq.set_postfix(acc='g:%.3f,v:%.3f,c:%.3f'%(acc_grapheme.item(),acc_vowel.item(),acc_consonant.item()))
                

                acc_epoch+=acc.item()
        
            acc_epoch = acc_epoch/len(train_loader)
            tq.close()
            print('epoch:%d, acc for val: %f'%(epoch,acc_epoch))
        
        
if __name__=='__main__':
    
    
    SEED = 2019
    
    seed_everything(SEED) 
    
    train_path = '../train_img/'
    all_images = sorted(glob.glob(train_path+'*.png'))
    train_labels_df = pd.read_csv('../train.csv')
    all_labels = train_labels_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    train(all_images,all_labels,batch_size=64,num_epochs=150,learning_rate=0.0001)
    