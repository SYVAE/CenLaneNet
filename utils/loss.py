import torch
import numpy as np
import torch.nn as nn

def CalCosinLoss(input,instancegt,margin,alpha,step_zero=True):
    in_loss=0
    ex_loss=0
    for i in range(0,input.shape[0]):
        total_center=[]
        tempgt=instancegt[i,:,:]
        uniquenum=torch.unique(tempgt)
        tempinput=input[i,:,:,:]

        for j in uniquenum:
            if j==0 and step_zero:
                continue
            mask=tempgt==j
            masktensor_=tempinput[:,mask]
            tempcenter=masktensor_.mean(1)

            center=tempcenter/tempcenter.norm()
            total_center.append(center)

            a=center.view(center.shape[0],1,1)
            center_map=a.repeat(1,tempinput.shape[1],tempinput.shape[2])
            c=center_map.mul(tempinput/tempinput.norm(dim=0))
            c=(1-c.sum(0)).mul(mask.float())

            e=c.sum(0).sum(0)
            in_loss=in_loss+e

        for jj in range(0,len(total_center)):
            for jjj in range(0,len(total_center)):
                if jj==jjj:
                    continue
                center1=total_center[jj]
                center2=total_center[jjj]
                a=center1.mul(center2)
                b=a.sum(0)
                c=torch.clamp(b-margin,min=0)
                ex_loss=ex_loss+c

    loss = in_loss + alpha * ex_loss
    return loss