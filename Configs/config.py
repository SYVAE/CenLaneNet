'''Configuration settings'''
import numpy as np
__all__={'training_cfg','Dataprocess_cfg','Model_cfg'}
class training_cfg():
    optimization=['Adam','SGD']
    base_lr = 1e-3

class Dataprocess_cfg():
     flip=True
     rotation=False
     transition=True
     # Randomize hue, vibrance, etc.
     augment_photometric_distort=True
     prob=0.5
     dataMean = np.array([[[123.6800, 116.7790, 103.9390]]])
     imgSize = [256, 512]
     gtSize=[256,512]
     CUlane_im_width=1640
     CUlane_im_height=590

class Model_cfg():
     DualAttention=1
     DAhead_outputchannel=128
     resnet_channel=[64,128,256,512]
     EffcientBlocks = [13, 15, 10, 8, 18]  # b5 13, b6 15 b4 10, b3 8,b7 18
     efficientb6=[8,40,14,72]# 4x layeridx channel 8x layeridx channel

