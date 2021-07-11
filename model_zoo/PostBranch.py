###########################################################################
# Created by: YI SUN
# Copyright (c) 2020
###########################################################################
from __future__ import division

from .Backbone import *
import torch.nn.functional as F
from Configs import config as cfg
import numpy as np
__all__ = ['PostBranch']

##fortraining###
class PostBranch(nn.Module):
    def __init__(self, configs):
        super(PostBranch, self).__init__()

        self.configs=configs
        #for instance
        self.fc1 = C(cfg.Model_cfg.DAhead_outputchannel, 32, 1, 0, 1)
        self.layers1 = C(32, 16, 3, 1, 1)
        self.layers2 = C(16, 6, 3, 1, 1)

        # for centermap
        self.fc2 = C(cfg.Model_cfg.DAhead_outputchannel, 32, 1, 0, 1)
        self.layers4 = C(32, 16, 3, 1, 1)
        self.layers5 = C(16, 1, 3, 1, 1)

        # for segmentation
        self.fc3 = C(cfg.Model_cfg.DAhead_outputchannel, 32, 1, 0, 1)
        self.layers6 = C(32, 16, 3, 1, 1)

        self.weights = nn.Parameter(
                nn.init.kaiming_normal_(torch.rand(2, 16, 1, 1, requires_grad=True)))

        self.s = configs.scale_factor
        self.m = configs.margin
    def forward(self, x,gt=None,savepath=None,count=None):
        if gt is not None:
            # imsize = cfg.Dataprocess_cfg.imgSize
            imsize = (gt.shape[1],gt.shape[2])

            up=F.interpolate(x, imsize, mode='bilinear', align_corners=True)

            ins_x1 = self.fc1(up)
            ins_x2=self.layers1(ins_x1)
            ins_x3=self.layers2(ins_x2)
            insres=ins_x3

            #center
            cen_x1=self.fc2(up)
            cen_x2=self.layers4(cen_x1)
            cen_x3=self.layers5(cen_x2)
            cen_x4 = torch.clamp(cen_x3.sigmoid_(), min=1e-4, max=1 - 1e-4)
            cenres=cen_x4

            #segmentation
            seg_x1 = self.fc3(up)
            seg_x2 = self.layers6(seg_x1)

            cosine = torch.nn.functional.conv2d(F.normalize(seg_x2), F.normalize(self.weights))
            phi = cosine - self.m
            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot = one_hot.view(cosine.shape[0], cosine.shape[1], -1)
            one_hot.scatter_(1, gt.view(gt.shape[0], 1, -1), 1)

            one_hot = one_hot.view(cosine.shape[0], cosine.shape[1], cosine.shape[2], cosine.shape[3])
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            imsize = cfg.Dataprocess_cfg.gtSize
            # instance
            up = F.interpolate(x, imsize, mode='bilinear', align_corners=True)

            ins_x1 = self.fc1(up)
            ins_x2 = self.layers1(ins_x1)
            ins_x3 = self.layers2(ins_x2)
            insres = ins_x3

            # center
            cen_x1 = self.fc2(up)
            cen_x2 = self.layers4(cen_x1)
            cen_x3 = self.layers5(cen_x2)
            cen_x4 = torch.clamp(cen_x3.sigmoid_(), min=1e-4, max=1 - 1e-4)
            cenres = cen_x4

            # segmentation
            seg_x1 = self.fc3(up)
            seg_x2 = self.layers6(seg_x1)


            cosine = torch.nn.functional.conv2d(F.normalize(seg_x2), F.normalize(self.weights))
            cosine[0, 1, :, :] = cosine[0, 1, :, :] - self.m
            output = cosine * self.s
            # plt.show()
        return insres, output,cenres
