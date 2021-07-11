###########################################################################
# Created by: Hang Zhang
# Modified by: SUNYI
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from model_zoo.EfficientNet import EfficientNet
from Configs import config as cfg
from model_zoo.attention import *
from model_zoo.Netblock import *
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BaseEfficientNetDA(nn.Module):
    def __init__(self, backbone, modelpath='./pretrained', norm_layer=nn.BatchNorm2d,pretrained=False):
        super(BaseEfficientNetDA, self).__init__()
        # copying modules from pretrained models
        self.pretrained = EfficientNet.from_pretrained(backbone) if pretrained else EfficientNet.from_name(backbone)
        self.blocknum=cfg.Model_cfg.EffcientBlocks[1]
        self.head = DANetHead(cfg.Model_cfg.efficientb6[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)

    def forward(self, x):
        """ This is modified by Lanenet, 2020/5/12. """
        x = self.pretrained._swish(self.pretrained._bn0(self.pretrained._conv_stem(x)))

        for idx in range(0,self.blocknum):
            drop_connect_rate = self.pretrained._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.pretrained._blocks)
            x = self.pretrained._blocks[idx](x, drop_connect_rate=drop_connect_rate)


        # Head
        x = self.head(x)
        x = list(x)
        x = x[0]
        return x


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)
