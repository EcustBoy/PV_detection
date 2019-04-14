# -*- coding: utf-8 -*-
"""
@author: qing
usage:train model 
"""
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import numpy as np
import time
import os

from Unet_model import *
from function import *
from Dataload import *

train_raw_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/raw_image_data/seg_images/"
train_label_dir="D:/my_course/graduation_project/AerialImageSolarArray/3385804/label_image_data/Mask_label/seg_mask/"

map_file_path="D:/my_course/graduation_project/AerialImageSolarArray/3385804/map_file.txt"
#生成原始图片路径与标签/掩模图片路径的配对.txt文件,以便后续数据读取
raw_content=os.listdir(raw_dir)
raw_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
label_content=os.listdir(label_dir)
label_content.sort(key=lambda x: (int((x.split('-')[0]).split('_')[0])-1)*50 +int((x.split('-')[0]).split('_')[1]))
num=len(label_content)

def train(model,dataloader,criterion,optimizer,epoch):
    model.train()
    for Id,(index,input_image,label_image) in enumerate(dataloader):
        s_time=time.time()
        #output shape:(batch_size,channel,h,w)
        input_image=torch.autograd.Variable(input_image)
        output_image=model(input_image)
        batch_size=output_image.size()[0]
        channel=output_image.size()[1]
        output_image=output_image.contiguous().view(batch_size,channel,-1)
        label_image=label_image.contiguous().view(batch_size,1,-1)
        output_image=output_image.transpose(1,2)
        label_image=label_image.transpose(1,2)
        output_image=output_image.view(-1,channel)
        label_image=label_image.contiguous().view(-1,1)
        #转为Variable
        
        label_image=torch.autograd.Variable(label_image)
        #定义损失函数和优化器类型
        loss=criterion(output_image,label_image)
        #计算KPI
        precision,recall,accuracy,F1_score=calc_KPI(output_image,label_image)
        #优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e_time=time.time()
        time_=e_time-s_time
        #print
        print('epoch:{}-batch_id:{}-precision:{}-recall:{}-accuracy:{}-calc_time:{}'\
              .format(epoch,Id,precision,recall,accuracy,time_))
        
def validate(model,dataloader,criterion):
    model.eval()
    for Id,(index,input_image,label_image) in enumerate(dataloader):
        s_time=time.time()
        #output shape:(batch_size,channel,h,w)
        input_image=torch.autograd.Variable(input_image)
        output_image=model(input_image)
        batch_size=output_image.size()[0]
        channel=output_image.size()[1]
        output_image=output_image.contiguous().view(batch_size,channel,-1)
        label_image=label_image.contiguous().view(batch_size,1,-1)
        output_image=output_image.transpose(1,2)
        label_image=label_image.transpose(1,2)
        output_image=output_image.view(-1,channel)
        label_image=label_image.contiguous().view(-1,1)
        #转为Variable
        
        label_image=torch.autograd.Variable(label_image)
        #定义损失函数和优化器类型
        loss=criterion(output_image,label_image)
        #计算KPI
        precision,recall,accuracy,F1_score=calc_KPI(output_image,label_image)
        #优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e_time=time.time()
        time_=e_time-s_time
        #print
        print('epoch:{}-batch_id:{}-precision:{}-recall:{}-accuracy:{}-calc_time:{}'\
              .format(epoch,Id,precision,recall,accuracy,time_))
    
    

def main():
    #记录每一个batch的损失数据
    losses=[]
    batch_size=2
    workers=4
    model=Unet()
    #载入训练与验证数据集
    train_dataloader=torch.utils.data.DataLoader(
            ImageLoader(raw_dir,label_dir,map_file_path,norm=False),
            batch_size=batch_size,shuffle=False,
            num_workers=workers)
    
    val_dataloader=torch.utils.data.DataLoader(
            ImageLoader(raw_dir,label_dir,map_file_path,norm=False),
            batch_size=batch_size,shuffle=False,
            num_workers=workers)
    
    
    
    
    
    
    
    