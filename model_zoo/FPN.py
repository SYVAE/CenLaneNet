import torch
import torch.nn as nn
import Configs.config as cfg
import torch.nn.functional as F
from typing import List
class FPN(nn.Module):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf
    """
    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, cfg.Model_cfg.FPN_outputchannel, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.Model_cfg.FPN_pad else 0
        self.pred_layers = nn.Conv2d(cfg.Model_cfg.FPN_outputchannel, cfg.Model_cfg.FPN_outputchannel, kernel_size=3, padding=padding)
        self.interpolation_mode = cfg.Model_cfg.FPN_interpolation_mode

    def forward(self, x8_,x4_):
        # upsampe(x4_)+x8_
        _, _, h, w = x4_.size()
        x = F.interpolate(x8_, size=(h, w), mode=self.interpolation_mode, align_corners=False)

        x = self.lat_layers[0](x4_) + self.lat_layers[1](x)


        x=F.relu(x, inplace=True)


        return x
