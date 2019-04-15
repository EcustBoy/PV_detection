# -*- coding: utf-8 -*-
"""
@author: qing
usage:define U-Net model and test
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

class Unet(nn.Module):
    def convlayer(self,in_channel,out_channel):
        #两侧中的同级内卷积运算模块
        #in_channel:输入图片通道数;out_channel:输出图片通道数
        module=nn.Sequential(
                #conv:o=(i-k+2*p)/s+1
                nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
                )
        return module
    
    def maxpool(self,x,kernel_size,stride,padding):
        #左半侧编码器的(下采样)最大池化操作
        return nn.MaxPool2d(kernel_size,stride,padding)(x)
    
    def deconvlayer(self,in_channel,out_channel,k,s,p):
        #右半侧解码器的上采样(反卷积)操作
        module=nn.Sequential(
                #deconv:o=(i-1)s+k-2*p
                nn.ConvTranspose2d(in_channel,out_channel,kernel_size=k,stride=s,padding=p),
                nn.ReLU(inplace=True)
                )
        return module
        
    def __init__(self):
        super(Unet,self).__init__()
        #左半侧编码器结构的卷积运算
        self.D_conv1=self.convlayer(3,8)
        self.D_conv2=self.convlayer(8,16)
        self.D_conv3=self.convlayer(16,32)
        self.D_conv4=self.convlayer(32,64)
        self.B_conv=self.convlayer(64,128)
        #右半侧解码器结构的卷积运算
        self.U_conv4=self.convlayer(128,64)
        self.U_conv3=self.convlayer(64,32)
        self.U_conv2=self.convlayer(32,16)
        self.U_conv1=self.convlayer(16,8)
        self.last=nn.Conv2d(8,2,kernel_size=1,stride=1)
        #右半侧解码器结构的上采样运算
        self.U4=self.deconvlayer(128,64,2,2,1)
        self.U3=self.deconvlayer(64,32,2,2,1)
        self.U2=self.deconvlayer(32,16,2,2,1)
        self.U1=self.deconvlayer(16,8,2,2,0)
        
    def forward(self,x):
        #编码器部分
        d1=self.D_conv1(x)
        #print('d1 shape:{}'.format(d1.size()))
        d1_pool=self.maxpool(d1,2,2,0)
        #print('d1_pool shape:{}'.format(d1_pool.size()))
        d2=self.D_conv2(d1_pool)
        #print('d2 shape:{}'.format(d2.size()))
        d2_pool=self.maxpool(d2,2,2,1)
        #print('d2_pool shape:{}'.format(d2_pool.size()))
        d3=self.D_conv3(d2_pool)
        #print('d3 shape:{}'.format(d3.size()))
        d3_pool=self.maxpool(d3,2,2,1)
        #print('d3_pool shape:{}'.format(d3_pool.size()))
        d4=self.D_conv4(d3_pool)
        #print('d4 shape:{}'.format(d4.size()))
        d4_pool=self.maxpool(d4,2,2,1)
        #print('d4_pool shape:{}'.format(d4_pool.size()))
        d5=self.B_conv(d4_pool)
        #print('d5 shape:{}'.format(d5.size()))
        #解码器部分
        u4=self.U4(d5)
        u4=u4.contiguous()
        #print('u4 shape:{}'.format(u4.size()))
         #两个不同的张量按高度维度堆叠
        u4_=self.U_conv4(torch.cat((d4,u4),1))
        #print('u4_ shape:{}'.format(u4_.size()))
        #print('u4_ size:{}'.format(u4_.size()))
        u3=self.U3(u4_)
        u3=u3.contiguous()
        #print('u3 shape:{}'.format(u3.size()))
        u3_=self.U_conv3(torch.cat((d3,u3),1))
        #print('u3_ shape:{}'.format(u3_.size()))
        u2=self.U2(u3_)
        u2=u2.contiguous()
        #print('u2 shape:{}'.format(u2.size()))
        u2_=self.U_conv2(torch.cat((d2,u2),1))
        #print('u2_ shape:{}'.format(u2_.size()))
        u1=self.U1(u2_)
        u1=u1.contiguous()
        #print('u1 shape:{}'.format(u1.size()))
        u1_=self.U_conv1(torch.cat((d1,u1),1))
        #print('u1_ shape:{}'.format(u1_.size()))
        y=self.last(u1_)
        #print('y shape:{}'.format(y.size()))
        #y shape:(batch_size,channel,h,w)
        return y
    
def Unet_test():
    #测试程序
    x=torch.randn(1,3,100,100)
    model=Unet()
    #print(model)
    y=model(x)
    original_size=list(x.size())
    output_size=list(y.size())
    print('the size of Unet model output is {}'.format(output_size))
    assert (original_size[2]==output_size[2])&(original_size[3]==output_size[3]),\
    "the size of output tensor:'{}' is wrong".format(original_size)
    print('test pass!')
    
if __name__=='__main__':
    Unet_test()


        
        
        
        
        
        
        
        
        
        
        
        

    
