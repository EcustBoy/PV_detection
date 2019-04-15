# -*- coding: utf-8 -*-
"""
@author: qing
usage:train model 
"""
import torch
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.misc import imread,imresize,imshow

from Dataload import *
from Unet_model import *
from function import *


def train(model,dataloader,criterion,optimizer,epoch,losses):
    model.train()
    for Id,(index,input_image,label_image) in enumerate(dataloader):
        s_time=time.time()
        #output shape:(batch_size,channel,h,w)
        #input_image=torch.autograd.Variable(input_image)
        #print('feed data...')
        input_image=input_image.float()
        label_image=label_image.float()
        
        output_image=model(input_image)
        #print('feed over')
        batch_size=output_image.size()[0]
        channel=output_image.size()[1]
        
        output_image=output_image.contiguous().view(batch_size,channel,-1)
        label_image=label_image.contiguous().view(batch_size,1,-1)
        output_image=output_image.transpose(1,2).contiguous()
        label_image=label_image.transpose(1,2).contiguous()
        output_image=output_image.view(-1,channel)
        label_image=label_image.contiguous().view(-1,1)/255
        label_image=label_image.long().squeeze_()
#        print('output_image shape:{}-type:{}'.format(output_image.size(),output_image.type()))
#        print('label_image shape:{}-type:{}'.format(label_image.size(),label_image.type()))
#        print(output_image)
#        print(label_image.sum())
        
        #label_image=torch.autograd.Variable(label_image)
        #定义损失函数和优化器类型
        loss=criterion(output_image,label_image)
        loss_=loss.detach().numpy()
        losses.append(loss_)
#        try:
#            loss=criterion(output_image,label_image)
#        except:
#            print('loss function call failed!')
#            print('error occurs when training!')
#            print('-------')
#            break
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
        print('===============================')
    return losses
        
#def validate(model,dataloader,criterion):
#    model.eval()
#    for Id,(index,input_image,label_image) in enumerate(dataloader):
#        s_time=time.time()
#        #output shape:(batch_size,channel,h,w)
#        input_image=torch.autograd.Variable(input_image)
#        output_image=model(input_image)
#        batch_size=output_image.size()[0]
#        channel=output_image.size()[1]
#        output_image=output_image.contiguous().view(batch_size,channel,-1)
#        label_image=label_image.contiguous().view(batch_size,1,-1)
#        output_image=output_image.transpose(1,2)
#        label_image=label_image.transpose(1,2)
#        output_image=output_image.view(-1,channel)
#        label_image=label_image.contiguous().view(-1,1)
#        #转为Variable
#        
#        label_image=torch.autograd.Variable(label_image)
#        #定义损失函数和优化器类型
#        loss=criterion(output_image,label_image)
#        #计算KPI
#        precision,recall,accuracy,F1_score=calc_KPI(output_image,label_image)
#        #优化器更新
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        e_time=time.time()
#        time_=e_time-s_time
#        #print
#        print('epoch:{}-batch_id:{}-precision:{}-recall:{}-accuracy:{}-calc_time:{}'\
#              .format(epoch,Id,precision,recall,accuracy,time_))


def test(model,test_dataloader,criterion):
    model.eval()
    for Id,(index,input_image,label_image) in enumerate(test_dataloader):
        input_image=input_image.float()
        label_image=label_image.float()
        
        output_image=model(input_image)
        #print('feed over')
        batch_size=output_image.size()[0]
        channel=output_image.size()[1]
        
        output_image=output_image.contiguous().view(batch_size,channel,-1)
        label_image=label_image.contiguous().view(batch_size,1,-1)
        output_image=output_image.transpose(1,2).contiguous()
        label_image=label_image.transpose(1,2).contiguous()
        output_image=output_image.view(-1,channel)
        label_image=label_image.contiguous().view(-1,1)/255
        label_image=label_image.long().squeeze_()
        loss=criterion(output_image,label_image)
        
        precision,recall,accuracy,F1_score=calc_KPI(output_image,label_image)
        
        input_image=input_image.permute(1,2,0).numpy()
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(input_image)
        plt.title('raw image')
        plt.subplot(132)
        plt.subplot(133)
        

        
if __name__=="__main__":
    epochs=500
    batch_size=5
    workers=2
    weight_decay=5e-4
    momentum=0.9
    losses=[]
    
    model=Unet()
    train_dataloader=data.DataLoader(
    ImageLoader(train_raw_dir,train_label_dir,train_map_file_path,norm=True),
    batch_size=batch_size,shuffle=True,
    num_workers=workers)

    val_dataloader=data.DataLoader(
    ImageLoader(val_raw_dir,val_label_dir,val_map_file_path,norm=True),
    batch_size=batch_size,shuffle=True,
    num_workers=workers)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05,
                                momentum=momentum,
                                weight_decay=weight_decay)
   
    for i in range(0,epochs):
        print('epoch:{}'.format(i))
        losses=train(model,train_dataloader,criterion,optimizer,i,losses)
    print('train finised')
    plt.plot(losses)
        

    


#-----------------------------------------------------





    
    