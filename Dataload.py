# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:18:00 2019

@author: 15216
"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.misc import imread,imresize,imshow

#全局变量
train_raw_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/raw_image_data/seg_images/train/"
train_label_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/label_image_data/Mask_label/seg_mask/train/"
train_map_file_path="D:/my_course/graduation_project/AerialImageSolarArray/3385804/train_map_file.txt"

val_raw_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/raw_image_data/seg_images/validate/"
val_label_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/label_image_data/Mask_label/seg_mask/validate/"
val_map_file_path="D:/my_course/graduation_project/AerialImageSolarArray/3385804/validate_map_file.txt"
#生成原始图片路径与标签/掩模图片路径的配对.txt文件,以便后续数据读取
#train_raw_content=os.listdir(train_raw_dir)
#train_raw_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
#train_label_content=os.listdir(train_label_dir)
#train_label_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
#num=len(train_label_content)

def generate_raw_label_map_file(raw_dir,label_dir,map_file_path):
    raw_content=os.listdir(train_raw_dir)
    raw_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
    label_content=os.listdir(train_label_dir)
    label_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
    num=len(label_content)
    try:
        with open(map_file_path,'w') as f:
            for index in range(0,num):
                raw_path=os.path.join(raw_dir,raw_content[index])
                label_path=os.path.join(label_dir,label_content[index])
                f.write(raw_path)
                f.write(' ')
                f.write(label_path)
                f.write(' ')
                f.write('\n')
    except:
        print('error occurs when write raw_label_map_file!')
                
    

#generate_raw_label_map_file(raw_dir,label_dir,map_file_path)
#从已生成的.txt文件读取原始图片路径与标签/掩模图片路径配对
def read_raw_label_paths(map_file_path,index):
    try:
        with open(map_file_path,'r') as f:
            lines=f.readlines()
            path=lines[index]
            return path
    except:
        print('error occurs when read path from raw_label_map_file!')
    #return path
#p=read_raw_label_paths(map_file_path,0)
def my_loader(path):
    return imread(path)

class ImageLoader(data.Dataset):
    def __init__(self,raw_dir,label_dir,map_file_path,norm,loader=my_loader):
        self.raw_dir=raw_dir
        self.label_dir=label_dir
        self.map_file_path=map_file_path
        self.norm=norm
        self.loader=loader
        
    def __getitem__(self,index):
        raw_label_path=read_raw_label_paths(self.map_file_path,index)
        raw_path=raw_label_path.split(' ')[0]
        label_path=raw_label_path.split(' ')[1]
        raw_image=self.loader(raw_path)
        label_image=self.loader(label_path)
        raw_image=raw_image.astype('int16')
        label_image=label_image.astype('int16')
        
        if self.norm:
            raw_image=raw_image.astype(float)
            raw_image=(raw_image-128)/128
            
        raw_image=transforms.ToTensor()(raw_image)
        label_image=transforms.ToTensor()(label_image)
        
        return index,raw_image,label_image
        
    def __len__(self):
        return len(os.listdir(self.raw_dir))

def test_imageloader():
    try:
       generate_raw_label_map_file(train_raw_dir,train_label_dir,train_map_file_path)
       imageloader=ImageLoader(train_raw_dir,train_label_dir,train_map_file_path,norm=False)
       raw=imageloader[1][1].numpy()
       raw=np.transpose(raw,(1,2,0))
       plt.imshow(raw)
       plt.show()
       print('test pass!')
    except:
       print('test failed!')
#    print(np.shape(raw))
        
if __name__=="__main__":
    test_imageloader()
    
    
    