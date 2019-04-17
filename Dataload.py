# -*- coding: utf-8 -*-
"""
@author: qing
usage:define Dataloader and test
"""
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
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

test_raw_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/raw_image_data/seg_images/test/"
test_label_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/label_image_data/Mask_label/seg_mask/test/"
test_map_file_path="D:/my_course/graduation_project/AerialImageSolarArray/3385804/test_map_file.txt"

def generate_raw_label_map_file(storage_raw_dir,storage_label_dir,map_file_path):
    raw_content=os.listdir(storage_raw_dir)
    raw_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
    label_content=os.listdir(storage_label_dir)
    label_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
    num=len(label_content)
    try:
        with open(map_file_path,'w') as f:
            for index in range(0,num):
                raw_path=os.path.join(storage_raw_dir,raw_content[index])
                label_path=os.path.join(storage_label_dir,label_content[index])
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
    batch_size=2
    workers=4
    generate_raw_label_map_file(test_raw_dir,test_label_dir,test_map_file_path)
    #imageloader=ImageLoader(train_raw_dir,train_label_dir,train_map_file_path,norm=False)
#       raw=imageloader[1][1].numpy()
#       raw=np.transpose(raw,(1,2,0))
#       plt.imshow(raw)
#       plt.show()
    test_dataloader=data.DataLoader(
    ImageLoader(test_raw_dir,test_label_dir,test_map_file_path,norm=False),
    batch_size=batch_size,shuffle=False,
    num_workers=workers)
    return test_dataloader

        
if __name__=="__main__":
    test_dataloader=test_imageloader()
   
    for Id,(index,input_image,label_image) in enumerate(test_dataloader):
        d=input_image[0].permute(1,2,0).numpy()
        
        if Id==0:
            plt.imshow(d)
            plt.show()
            break
    print('test pass!')
       
    
    
    