import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class CBR(nn.Module):
    """Conv2d-batchNormalization-Relu"""
    def __init__(self,in_channel,out_channel,ksize,padding,stride,bias=True):
        super(CBR,self).__init__()
        self.layers=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=(ksize,ksize),padding=padding,
                                  stride=stride,bias=bias),
                                      nn.BatchNorm2d(out_channel),
                                      nn.ReLU(inplace=True))
    def forward(self, input):
        outputs=self.layers(input)
        return outputs

class CR(nn.Module):
    """Conv2d-Relu"""
    def __init__(self,in_channel,out_channel,ksize,padding,stride,bias=True):
        super(CR,self).__init__()
        self.layers=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=(ksize,ksize),padding=padding,
                                  stride=stride,bias=bias),
                                      nn.ReLU(inplace=True))
    def forward(self, input):
        outputs=self.layers(input)
        return outputs

class C(nn.Module):
    """conv2d"""
    def __init__(self,in_channel,out_channel,ksize,padding,stride,bias=True):
        super(C,self).__init__()
        self.layers=nn.Conv2d(in_channel,out_channel,kernel_size=(ksize,ksize),padding=padding,
                                  stride=stride,bias=bias)
    def forward(self, input):
        outputs=self.layers(input)
        return outputs

class CB(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, padding, stride, bias=True):
        super(CB, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(ksize, ksize), padding=padding,
                                              stride=stride, bias=bias),
                                    nn.BatchNorm2d(out_channel))
    def forward(self, input):
        outputs = self.layers(input)
        return outputs

class ResNetblock_downsampling(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel):
        super(ResNetblock_downsampling,self).__init__()
        self.branchleft1=nn.Sequential(nn.Conv2d(in_channel,mid_channel,kernel_size=(1,1),padding=0,stride=2,bias=True),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),

                                       nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=1, stride=1,
                                                 bias=True),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),

                                       nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1,
                                                 bias=True),
                                       nn.BatchNorm2d(out_channel)
                                       )
        self.branchright1=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),padding=0,stride=2,bias=True),
                                       nn.BatchNorm2d(out_channel))
    def forward(self, input):

        res1=self.branchleft1(input)
        res2=self.branchright1(input)
        output=res1+res2

        return output


class ResNetblock_nodownsampling(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel):
        super(ResNetblock_nodownsampling, self).__init__()
        self.branchleft1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=(1, 1), padding=0, stride=1, bias=True),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=1, stride=1,
                      bias=True),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1,
                      bias=True),
            nn.BatchNorm2d(out_channel)
            )
    def forward(self, input):
        res1 = self.branchleft1(input)
        output = input + res1
        return output

class PsPblock(nn.Module):
    '''construct the PspBlock introduced in "Pyramid Scene Parsing Network"'''
    def __init__(self,in_channel,out_channel):
        super(PsPblock,self).__init__()
        #downsampling 2x
        self.p2x=nn.Sequential(nn.AvgPool2d(2,2,0),
                               nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                               nn.BatchNorm2d(out_channel),
                               nn.ReLU(inplace=True),
                               nn.UpsamplingBilinear2d(None,2)
                               )
        # downsampling 4x
        self.p4x=nn.Sequential(nn.AvgPool2d(4,4,0),
                               nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                               nn.BatchNorm2d(out_channel),
                               nn.ReLU(inplace=True),
                               nn.UpsamplingBilinear2d(None, 4)
                               )
        # downsampling 8x
        self.p8x=nn.Sequential(nn.AvgPool2d(8,8,0),
                               nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1, bias=True),
                               nn.BatchNorm2d(out_channel),
                               nn.ReLU(inplace=True),
                               nn.UpsamplingBilinear2d(None, 8)
                               )
        # # downsampling 16x
        # self.p16x = nn.Sequential(nn.AvgPool2d(16, 16, 0),
        #                          nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        #                          nn.BatchNorm2d(out_channel),
        #                          nn.ReLU(inplace=True),
        #                           nn.UpsamplingBilinear2d(None, 16)
        #                          )
    def forward(self, input):
        x1=self.p2x(input)
        x2=self.p4x(input)
        x3=self.p8x(input)
        # x4=self.p16x(input)
        outputs=torch.cat([input,x1,x2,x3],1)
        return outputs



##################DaNet copyfrom https://github.com/Andy-zhujunwen/danet-pytorch##########


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        # feat_b =x.view(batch_size, -1, height * width).permute(0, 2, 1)
        # # feat_b = feat_b / (feat_b.norm(dim=2, keepdim=True)+1e-7)
        # feat_c =x.view(batch_size, -1, height * width)
        # feat_c = feat_c / (feat_c.norm(dim=1, keepdim=True)+1e-7)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, out_channel, norm_layer=nn.BatchNorm2d):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels)
        self.cam = _ChannelAttentionModule()
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        self.out =nn.Conv2d(inter_channels, out_channel, 1)


    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        fusion_out = self.out(feat_fusion)
        return fusion_out






if __name__=="__main__":
    pass