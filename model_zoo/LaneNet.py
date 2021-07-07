from model_zoo import Backbone
from model_zoo import PostBranch
from Configs import config as cfg
import torch
import torch.nn as nn
class LaneNet(nn.Module):
    def __init__(self,backbone,config=None):
        super(LaneNet,self).__init__()
        if backbone=='resnet34' or backbone=='resnet18':
            if  cfg.Model_cfg.DualAttention==False:
                self.backbone=Backbone.BaseResNet(backbone=backbone)
            else:
                self.backbone=Backbone.BaseResNetDA(backbone=backbone)
        elif backbone=='ERF':
            if cfg.Model_cfg.DualAttention == False:
                self.backbone = Backbone.ERFnet()
            else:
                self.backbone = Backbone.ERFnetDA()
        else:
            if cfg.Model_cfg.DualAttention==False:
                 self.backbone = Backbone.BaseEfficientNet(backbone=backbone)
            else:
                self.backbone = Backbone.BaseEfficientNetDA(backbone=backbone)

        self.postbranch=PostBranch.PostBranch(config)

    def forward(self, x,gt=None,savepath=None,count=None):
        x=self.backbone(x)
        ins,seg,cen=self.postbranch(x,gt,savepath,count)
        return ins,seg,cen

if __name__=='__main__':
    import time
    import numpy as np
    print(torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax_flag", default=0, type=int)
    parser.add_argument("--scale_factor", default=40)
    parser.add_argument("--margin", default=0.6, type=float)
    config=parser.parse_args()
    net=LaneNet('efficientnet-b6',config)
    net.cuda()
    x = torch.ones(1, 3, cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1]).float().cuda()
    timelist=[]
    for i in range(0,100):
        t=time.time()
        ins,seg,cen=net(x)
        timelist.append(time.time()-t)
    print(np.mean(np.stack(timelist)))
    print(cen.shape)