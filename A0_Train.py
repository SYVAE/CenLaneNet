import torch
from torch.autograd import Variable
import numpy as np
import copy

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import cv2
import random
import time
import utils.centernet as cennet
import model_zoo.LaneNet as LaneNet
import os
import Configs.config as cfg
from torch.utils.data import DataLoader
from utils.color import *
from utils.loss import *
from Data.New_DataLoader import *
from tqdm import tqdm
import argparse

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        print('lr{0}'.format(param_group['lr']))
    global cur_lr
    cur_lr = new_lr


show =0
def TraintheNetwork(config):
    if not os.path.exists(config.savepath):
        os.mkdir(config.savepath)

    if config.Continue and os.path.exists(config.pretrained_model+'tempmodel.pkl'):
        model = LaneNet.LaneNet(config.The_selected_backbone,config)
        try:
            model.load_state_dict(torch.load(config.pretrained_model+'tempmodel.pkl'))
        except:
            temp=torch.load(config.pretrained_model+'tempmodel.pkl',map_location="cuda:0")
            from collections import OrderedDict
            new_state_dict=OrderedDict()
            for k,v in temp.items():
                name=k[7:]
                new_state_dict[name]=v

            model.load_state_dict(new_state_dict)
        print("load...")
    else:
         print("This will Train from scratch")
         model = LaneNet.LaneNet(config.The_selected_backbone,config)

    model.cuda()
    model=torch.nn.DataParallel(model,device_ids=[config.device])
    model.train()

    #define the optimizer :SGD
    optimizer=optim.Adam(model.parameters(),lr=cfg.training_cfg.base_lr)
    lr_sheduler=optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95,last_epoch=-1)
    segcriterion = nn.CrossEntropyLoss(weight=Variable(torch.tensor(np.array([0.5, 1]))).float().cuda())
###########Begin the traning process###########
    Dataset=LaneDataset(config.GtDataroot,mode='train')
    VDataset=LaneDataset(config.GtDataroot,mode='test')
    MyDataloader=DataLoader(Dataset,batch_size=config.batchsize,shuffle=True,num_workers=4,pin_memory=True)
    MyVDataloader=DataLoader(VDataset,batch_size=config.batchsize,shuffle=True,num_workers=4,pin_memory=True)

    minloss=1000000
    LOSS=[]
    validation_LOSS=[]
    for iteration in range(0,config.max_iter):
        total_loss=[]
        model.train()
        for data in tqdm(MyDataloader):
            input=data[0].float().cuda()
            instancegt=data[1].long().cuda()
            label = data[2].long().cuda()
            cenlabel=data[3].float().cuda()
###################forward###############################
            output = model.forward(input,label)
###################forward###############################
            segres = output[1]
            cen_seg = output[2]
            insres = output[0]
            insres=insres/insres.norm(dim=1).unsqueeze(1)

            loss1 = segcriterion(segres, label)
            loss2 = cennet._neg_loss(cen_seg, cenlabel)
            loss3 = CalCosinLoss(insres, instancegt, margin=config.instance_Margin, alpha=5000)
            loss =config.seg_cen_instance[0] * loss1 + config.seg_cen_instance[1] * loss2 + config.seg_cen_instance[2] * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(float(loss.detach()))

        lr_sheduler.step()
        vtotal_loss = []
        with torch.no_grad():
            model.eval()
            for data in tqdm(MyVDataloader):
                input = data[0].float().cuda()
                instancegt = data[1].long().cuda()
                label = data[2].long().cuda()
                cenlabel = data[3].float().cuda()
                ###################forward###############################
                output = model.forward(input, label)
                ###################forward###############################
                segres = output[1]
                cen_seg = output[2]
                insres = output[0]
                insres = insres / insres.norm(dim=1).unsqueeze(1)

                loss1 = segcriterion(segres, label)
                loss2 = cennet._neg_loss(cen_seg, cenlabel)
                loss3 = CalCosinLoss(insres, instancegt, margin=config.instance_Margin, alpha=5000)
                loss = config.seg_cen_instance[0] * loss1 + config.seg_cen_instance[1] * loss2 + config.seg_cen_instance[2] * loss3
                vtotal_loss.append(float(loss.detach()))

        '''show'''
        if iteration>5:
            '''1.seg prediction'''
            seg_res = output[1]
            seg_res = seg_res[0, :, :, :].cpu().squeeze(0).detach().numpy().transpose((1, 2, 0))
            out = seg_res
            out = np.argmax(out, axis=2)
            '''2.cen prediction'''
            cen_seg_res = output[2]
            cen_seg_res = cen_seg_res[0, :, :, :].cpu().squeeze(0).detach().numpy()
            '''3.instance prediciton'''
            ins_res = insres[0, :, :, :].cpu().squeeze(0).detach().numpy().transpose((1, 2, 0))
            ins_res[out == 0, :] = 0
            plt.figure(1)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(out)
            plt.subplot(1, 3, 2)
            plt.imshow(cen_seg_res)
            plt.subplot(1, 3, 3)
            plt.imshow(ins_res[:,:,0:3])
            plt.pause(0.01)

            LOSS.append(np.stack(total_loss).mean())
            validation_LOSS.append(np.stack(vtotal_loss).mean())
            if np.stack(vtotal_loss).mean()<minloss:
                minloss=np.stack(vtotal_loss).mean()
                torch.save(model.state_dict(), config.savepath + 'besttempmodel.pkl')
                print("bestmodel_selected")


            plt.figure("loss")
            plt.clf()
            plt.plot(np.array(range(0,len(LOSS))),np.stack(LOSS),'-',c='b',label='Trainloss')
            plt.plot(np.array(range(0, len(validation_LOSS))), np.stack(validation_LOSS), '-', c='r', label='validationloss')
            plt.legend()
            plt.title("loss")
            plt.pause(0.01)
            plt.savefig('loss.png')
        torch.save(model.state_dict(),config.savepath+'tempmodel.pkl')
        if iteration%config.saveinterval==0:
            torch.save(model.state_dict(), config.savepath + 'iter_'+str(iteration)+'_tempmodel.pkl')
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print("here")
    parser=argparse.ArgumentParser()
    parser.add_argument("--softmax_flag",default=0,type=int)
    parser.add_argument("--max_iter",default=100,type=int)
    parser.add_argument("--batchsize",default=5,type=int)
    parser.add_argument("--Continue",default=1,type=int)
    parser.add_argument("--pretrained_model",default='./tempmodel/')
    parser.add_argument("--savepath",default='./tempmodel/')

    parser.add_argument("--scale_factor",default=30)
    parser.add_argument("--margin",default=0.6,type=float)
    parser.add_argument("--instance_Margin",default=0,type=float)
    parser.add_argument("--saveinterval",default=10,type=int)

    ######loss_ratio####
    parser.add_argument("--seg_cen_instance",default=[1,0.05,0.0001])
    parser.add_argument("--The_selected_backbone",default='efficientnet-b6',choices=['resnet18','efficientnet-b6','ERF'])
    '''validation'''
    parser.add_argument("--GtDataroot",default="./",type=str)
    parser.add_argument('--device',default=0,type=int)
    config=parser.parse_args()
    torch.cuda.set_device(config.device)
    print(torch.cuda.get_device_name(0))
    TraintheNetwork(config)
