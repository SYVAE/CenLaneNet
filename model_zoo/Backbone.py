###########################################################################
# Created by: Hang Zhang
# Modified by: SUNYI
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

from model_zoo.ERFnet import *
from model_zoo.ResNet import *
from model_zoo.EfficientNet import EfficientNet
from Configs import config as cfg
from model_zoo.attention import *
from model_zoo.Netblock import *
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class BaseResNetDA(nn.Module):
    def __init__(self, backbone,modelpath='./pretrained', dilated=True, norm_layer=nn.BatchNorm2d, pretrained=False,
                 multi_grid=False, multi_dilation=None):
        super(BaseResNetDA, self).__init__()
        # copying modules from pretrained models
        if backbone == 'resnet34':
            self.pretrained = resnet34(pretrained=False, dilated=dilated,
                                              norm_layer=norm_layer, root=modelpath,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone=='resnet18':
            self.pretrained = resnet18(pretrained=False, dilated=dilated,
                                       norm_layer=norm_layer, root=modelpath,
                                       multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.head = DANetHead(cfg.Model_cfg.resnet_channel[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)
    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        x = self.head(c4)
        x = list(x)
        x=x[0]
        # print("c4shape",c4.shape)
        return x

class BaseResNet(nn.Module):
    def __init__(self, backbone,modelpath='./pretrained', dilated=True, norm_layer=nn.BatchNorm2d, pretrained=False,
                 multi_grid=False, multi_dilation=None):
        super(BaseResNet, self).__init__()
        # copying modules from pretrained models
        if backbone == 'resnet34':
            self.pretrained = resnet34(pretrained=False, dilated=dilated,
                                              norm_layer=norm_layer, root=modelpath,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone=='resnet18':
            self.pretrained = resnet18(pretrained=False, dilated=dilated,
                                       norm_layer=norm_layer, root=modelpath,
                                       multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.fc=C(cfg.Model_cfg.resnet_channel[3],cfg.Model_cfg.DAhead_outputchannel,1,0,1)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        c=self.fc(c4)
        # print("c4shape",c4.shape)
        return c

class BaseEfficientNet(nn.Module):
    def __init__(self, backbone, modelpath='./pretrained', pretrained=False):
        super(BaseEfficientNet, self).__init__()
        # copying modules from pretrained models
        self.pretrained = EfficientNet.from_pretrained(backbone) if pretrained else EfficientNet.from_name(backbone)
        if backbone=='efficientnet-b5':
            self.blocknum=cfg.Model_cfg.EffcientBlocks[0]
            self.fc = C(cfg.Model_cfg.efficientb5[3], cfg.Model_cfg.DAhead_outputchannel, 1, 0, 1)
        elif backbone=='efficientnet-b6':
            self.blocknum=cfg.Model_cfg.EffcientBlocks[1]
            self.fc = C(cfg.Model_cfg.efficientb6[3], cfg.Model_cfg.DAhead_outputchannel, 1, 0, 1)
        elif backbone == 'efficientnet-b4':
            self.blocknum = cfg.Model_cfg.EffcientBlocks[2]
            self.fc = C(cfg.Model_cfg.efficientb4[3], cfg.Model_cfg.DAhead_outputchannel, 1, 0, 1)
        elif backbone == 'efficientnet-b3':
            self.blocknum = cfg.Model_cfg.EffcientBlocks[3]
            self.fc = C(cfg.Model_cfg.efficientb3[3], cfg.Model_cfg.DAhead_outputchannel, 1, 0, 1)

    def forward(self, x):
        """ This is modified by Lanenet, 2020/5/12. """
        # Convolution layers
        # Stem
        x = self.pretrained._swish(self.pretrained._bn0(self.pretrained._conv_stem(x)))

        # Blocks
        for idx in range(0,self.blocknum):
            drop_connect_rate = self.pretrained._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.pretrained._blocks)
            x = self.pretrained._blocks[idx](x, drop_connect_rate=drop_connect_rate)
            # print('idx:{0},shape{1}'.format(idx+1,x.shape))

        # Head
        x=self.fc(x)
        return x

class BaseEfficientNetDA(nn.Module):
    def __init__(self, backbone, modelpath='./pretrained', norm_layer=nn.BatchNorm2d,pretrained=False):
        super(BaseEfficientNetDA, self).__init__()
        # copying modules from pretrained models
        self.pretrained = EfficientNet.from_pretrained(backbone) if pretrained else EfficientNet.from_name(backbone)
        if backbone=='efficientnet-b5':
            self.blocknum=cfg.Model_cfg.EffcientBlocks[0]
            # self.fc = C(cfg.Model_cfg.efficientb5[3], cfg.Model_cfg.DAhead_outputchannel, 1, 0, 1)
            self.head = DANetHead(cfg.Model_cfg.efficientb5[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)
        elif backbone=='efficientnet-b6':
            self.blocknum=cfg.Model_cfg.EffcientBlocks[1]
            self.head = DANetHead(cfg.Model_cfg.efficientb6[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)
        elif backbone=='efficientnet-b4':
            self.blocknum=cfg.Model_cfg.EffcientBlocks[2]
            self.head = DANetHead(cfg.Model_cfg.efficientb4[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)
        elif backbone == 'efficientnet-b3':
            self.blocknum = cfg.Model_cfg.EffcientBlocks[3]
            self.head = DANetHead(cfg.Model_cfg.efficientb3[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)
        elif backbone == 'efficientnet-b7':
            self.blocknum = cfg.Model_cfg.EffcientBlocks[4]
            self.head = DANetHead(cfg.Model_cfg.efficientb7[3], cfg.Model_cfg.DAhead_outputchannel, norm_layer)

    def forward(self, x):
        """ This is modified by Lanenet, 2020/5/12. """
        # Convolution layers
        # Stem
        x = self.pretrained._swish(self.pretrained._bn0(self.pretrained._conv_stem(x)))

        # Blocks
        for idx in range(0,self.blocknum):
            drop_connect_rate = self.pretrained._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.pretrained._blocks)
            x = self.pretrained._blocks[idx](x, drop_connect_rate=drop_connect_rate)
            # print('idx:{0},shape{1}'.format(idx+1,x.shape))

        # Head
        x = self.head(x)
        x = list(x)
        x = x[0]
        return x


class ERFnet(nn.Module):
    def __init__(self,norm_layer=nn.BatchNorm2d):
        super(ERFnet,self).__init__()
        self.encoder = Encoder()
        ##decoder
        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.head = DANetHead(128, 128, norm_layer)

    def forward(self,x):
        output=self.encoder(x)
        x = self.head(output)
        x = list(x)
        output = x[0]
        # for layers in self.layers:
        #     # print(output.shape)
        #     output=layers(output)
        return output

class ERFnetDA(nn.Module):
    def __init__(self,norm_layer=nn.BatchNorm2d):
        super(ERFnetDA,self).__init__()
        self.encoder = Encoder()
        ##decoder
        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.head = DANetHead(128, 128, norm_layer)

    def forward(self,x):
        output=self.encoder(x)
        x = self.head(output)
        x = list(x)
        output = x[0]
        # for layers in self.layers:
        #     # print(output.shape)
        #     output=layers(output)
        return output


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

if __name__ == '__main__':
    print('here')
    device=0
    print(torch.cuda.get_device_name(device))
    torch.cuda.set_device(device)
    net=ERFnet()
    net.cuda()
    x=torch.ones(1,3,cfg.Dataprocess_cfg.imgSize[0],cfg.Dataprocess_cfg.imgSize[1]).float().cuda()
    net(x)