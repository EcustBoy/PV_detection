# -*- coding: utf-8 -*-
"""
@author: qing
usage:define some utils function if needed
"""
import torch
import numpy as np


def calc_KPI(output,label):
    #output shape:(batch_size,channel,h,w)
    """
    calculate: precision recall accuracy F-score
    output shape: (batch_size,channel,h,w)
    """
    channel=output.size(1)
    batch_size=output.size(0)
    #shape:(batch_size,channel,h*w)
    output=output.contiguous().view(batch_size,channel,-1)
    label=label.contiguous().view(batch_size,1,-1)
    #print('shape of output and label:{}{}'.format(output.size(),label.size()))
    #shape:(batch_size,h*w,channel)
    output=output.transpose(1,2)
    label=label.transpose(1,2)
    #shape:(batch_size*h*w,channel)
    output=output.contiguous().view(-1,channel)
    label=label.contiguous().view(-1,1)
    #print('shape of output and label:{}{}'.format(output.size(),label.size()))
#    print('-----label:{}'.format(label))
    _,output_class=output.topk(1,1,True,True)
#    _,label_class=label.topk(1,1,True,True)
    #eq()运算输出值为torch.uint8类型
    output_class=output_class.eq(1)
    label_class=label.eq(1)
    #计算本batch各项指标
    P=output_class.sum()
    num=output_class.size()[0]
    N=num-P
    T=label_class.sum()
    TP=output_class.add(label_class).eq(2).sum()
    FP=P-TP
    FN=T-TP
    TN=N-FN
#    print(output_class)
#    print(label_class)
#    print(P)
#    print(N)
#    print(T)
#    print(TP)
#    print(FP)
#    print(FN)
#    print(TN)
    try:
        precision=TP.numpy()/P.numpy()
    except:
        precision=np.nan
    try:
        recall=TP.numpy()/T.numpy()
    except:
        recall=np.nan
    try:
        accuracy=(TP.numpy()+TN.numpy())/(P.numpy()+N.numpy())
    except:
        accuracy=np.nan
    try:
        F1_score=2/(1/precision+1/recall)
    except:
        #说明上一步溢出，precision or recall接近或等于0,此时直接赋予F1_score为0
        F1_score=0
    #print('presicion:{} recall:{} accuracy:{} F1_score:{:f}'.format(precision,recall,accuracy,F1_score))
    return (precision,recall,accuracy,F1_score)
    

#测试程序
def test_calc_KPI():
    output=torch.zeros(2,2,10,10)
    output[:,0,:,0:5]=1
    output[:,1,:,5:10]=1
    label=torch.zeros(1,1,10,10)
    label[:,0,:,3:7]=1
    try :
        KPI=calc_KPI(output,label)
        print('test pass!')
        return KPI
    except:
        print('error occurs when call calc_KPI function!')
   

if __name__=="__main__":
    KPI=test_calc_KPI()
    print(KPI)
    

