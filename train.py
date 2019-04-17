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

def train(model,dataloader,criterion,optimizer,epoch,losses,precision,recall,accuracy,F1_score):
    model.train()
    for Id,(index,input_image,label_image) in enumerate(dataloader):
        s_time=time.time()
        
        input_image=input_image.float()
        label_image=label_image.float()
        
        output_image=model(input_image)

        batch_size=output_image.size()[0]
        channel=output_image.size()[1]
        
        output_image=output_image.contiguous().view(batch_size,channel,-1)
        label_image=label_image.contiguous().view(batch_size,1,-1)
        output_image=output_image.transpose(1,2).contiguous()
        label_image=label_image.transpose(1,2).contiguous()
        output_image=output_image.view(-1,channel)
        label_image=label_image.contiguous().view(-1,1)/255
        label_image=label_image.long().squeeze_()
        
        #label_image=torch.autograd.Variable(label_image)
        #定义损失函数和优化器类型
        loss=criterion(output_image,label_image)
        loss_=loss.detach().numpy()
        losses.append(loss_)
        #计算KPI
        Precision,Recall,Accuracy,F1=calc_KPI(output_image,label_image)
        precision.append(Precision)
        recall.append(Recall)
        accuracy.append(Accuracy)
        F1_score.append(F1)
        #优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e_time=time.time()
        time_=e_time-s_time
        print('train\n')
        print('epoch:{}-batch_id:{}-precision:{}-recall:{}-accuracy:{}-calc_time:{}'\
              .format(epoch,Id,Precision,Recall,Accuracy,time_))
        print('===============================')
    return losses,precision,recall,accuracy,F1_score
        
def validate(model,dataloader,criterion,epoch_l,losses_l,precision_l,recall_l,accuracy_l,F1_score_l):
    model.eval()
    precision=[]
    recall=[]
    accuracy=[]
    F1_score=[]
    losses=[]
    for Id,(index,input_image,label_image) in enumerate(dataloader):
        s_time=time.time()
        
        input_image=input_image.float()
        label_image=label_image.float()
        
        output_image=model(input_image)
        
        batch_size=output_image.size()[0]
        channel=output_image.size()[1]
        
        output_image=output_image.contiguous().view(batch_size,channel,-1)
        label_image=label_image.contiguous().view(batch_size,1,-1)
        output_image=output_image.transpose(1,2).contiguous()
        label_image=label_image.transpose(1,2).contiguous()
        output_image=output_image.view(-1,channel)
        label_image=label_image.contiguous().view(-1,1)/255
        label_image=label_image.long().squeeze_()
        #label_image=torch.autograd.Variable(label_image)
        #定义损失函数和优化器类型
        loss=criterion(output_image,label_image)
        loss_=loss.detach().numpy()
        losses.append(loss_)
        #计算KPI
        Precision,Recall,Accuracy,F1=calc_KPI(output_image,label_image)
        precision.append(Precision)
        recall.append(Recall)
        accuracy.append(Accuracy)
        F1_score.append(F1)
 
        e_time=time.time()
        time_=e_time-s_time
        
    losses=sum(losses)/len(losses)
    losses_l.append(losses)
    precision=sum(precision)/len(precision)
    precision_l.append(precision)
    recall=sum(recall)/len(recall)
    recall_l.append(recall)
    accuracy=sum(accuracy)/len(accuracy)
    accuracy_l.append(accuracy)
    F1_score=sum(F1_score)/len(F1_score)
    F1_score_l.append(F1_score)
    
    print('validate\n')
    print('epoch:{}-losses:{}-precision:{}-recall:{}-accuracy:{}-calc_time:{}'\
              .format(epoch,losses,precision,recall,accuracy,time_))
    print('---------------------')
    return losses_l,precision_l,recall_l,accuracy_l,F1_score_l
    


def test(model,test_dataloader):
    model.eval()
    for Id,(index,input_image,label_image) in enumerate(test_dataloader):
        input_image=input_image.float()
        label_image=label_image.float()
        #output_image shape:(batch_size,2,100,100)
        output_image=model(input_image)
        batch_size=output_image.size()[0]
        #channel=output_image.size()[1]
        #segmask is visualization of predicted segmentation results
        segmask=torch.zeros(batch_size,100,100)
        for b in range(0,batch_size):
            for row in range(0,100):
                for col in range(0,100):
                    if output_image[b,0,row,col]>=0.5:
                        segmask[b,row,col]=0
                    else:
                        segmask[b,row,col]=255
        if Id==0:
            break
    #将input_image,output_image,segmask由tensor类型转为numpy类型
    i=input_image[0].permute(1,2,0).numpy()
    l=label_image[0].permute(1,2,0).numpy()[:,:,0]
    s=segmask[0].numpy()
    plt.figure();
    plt.imshow(i)
    return i,l,s
                        

if __name__=="__main__":
    epochs=100
    batch_size=5
    workers=2
    weight_decay=5e-4
    momentum=0.9
    train_losses=[]
    val_losses=[]
    precision=[]
    precision_=[]
    recall=[]
    recall_=[]
    accuracy=[]
    accuracy_=[]
    F1_score=[]
    F1_score_=[]
    
    model=Unet()
    
    train_dataloader=data.DataLoader(
    ImageLoader(train_raw_dir,train_label_dir,train_map_file_path,norm=True),
    batch_size=batch_size,shuffle=True,
    num_workers=workers)

    val_dataloader=data.DataLoader(
    ImageLoader(val_raw_dir,val_label_dir,val_map_file_path,norm=True),
    batch_size=batch_size,shuffle=True,
    num_workers=workers)
    
    test_dataloader=data.DataLoader(
    ImageLoader(test_raw_dir,test_label_dir,test_map_file_path,norm=False),
    batch_size=1,shuffle=False,
    num_workers=workers)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05,
                                momentum=momentum,
                                weight_decay=weight_decay)
   
    for epoch in range(0,epochs):
        print('epoch:{}'.format(epoch))
        train_losses,precision,recall,accuracy,F1_score=\
        train(model,train_dataloader,criterion,optimizer,epoch,train_losses,precision,recall,accuracy,F1_score)
        #每一代训练完毕后调用验证函数，计算全体验证集上的KPI平均值，并存入列表中
        val_losses,precision_,recall_,accuracy_,F1_score_=\
        validate(model,val_dataloader,criterion,epoch,val_losses,precision_,recall_,accuracy_,F1_score_)
    #全部训练结束后调用测试函数，可视化对比输入RGB图，真实标签图与网络预测图
    input_image,label_image,segmask=test(model,test_dataloader)
    print('test:input_image size:{}'.format(np.shape(input_image)))
    print('test:label_image size:{}'.format(np.shape(label_image)))
    print('test:segmask size:{}'.format(np.shape(segmask)))
    print('train finised')
    #可视化部分
    plt.figure(1)
    plt.title('losses')
    plt.subplot(121)
    plt.plot(train_losses,'b')
    plt.title('train losses')
    plt.subplot(122)
    plt.plot(val_losses,'r')
    plt.title('val losses')
    
    plt.figure(2)
    plt.subplot(421)
    plt.plot(precision,'b')
    plt.title('train precision')
    plt.subplot(422)
    plt.plot(precision_,'r')
    plt.title('val precision')
    plt.subplot(423)
    plt.plot(recall,'b')
    plt.title('train recall')
    plt.subplot(424)
    plt.plot(recall_,'r')
    plt.title('val recall')
    plt.subplot(425)
    plt.plot(accuracy,'b')
    plt.title('train accuracy')
    plt.subplot(426)
    plt.plot(accuracy_,'r')
    plt.title('val accuracy')
    plt.subplot(427)
    plt.plot(F1_score,'b')
    plt.title('train F1_score')
    plt.subplot(428)
    plt.plot(F1_score_,'r')
    plt.title('val F1_score')
    
    plt.figure(3)
    plt.subplot(131)
    plt.imshow(input_image)
    plt.title('input')
    plt.subplot(132)
    plt.imshow(label_image,cmap='gray')
    plt.title('label')
    plt.subplot(133)
    plt.imshow(segmask,cmap='gray')
    plt.title('prediction')
    
    
        

    








    
    