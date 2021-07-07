from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
from Configs import config as cfg
import random
import matplotlib.pyplot as plt
class LaneDataset(Dataset):
    def __init__(self,path,mode='train'):
        list = []
        list_dir = next(os.walk(path + 'img/'))
        imagename = list_dir[2]
        for i in range(0, len(imagename)):
            name = imagename[i].split('.')[0]
            # print(name)
            list.append(name)

        self.instancelist = []
        self.imagelist = []
        self.seglist = []
        self.cenlist = []
        random.shuffle(list)

        if mode=='train':
            for j in range(0, len(list)):
                if j%50==0:
                    continue
                self.instancelist.append(path+ 'instance_label/' + list[j] + '.png')
                self.imagelist.append(path + 'img/' + list[j] + '.jpg')
                self.seglist.append(path + 'seglabel/' + list[j] + '.png')
                self.cenlist.append(path + 'centerpoint/' + list[j] + '.png')
        else:
            for j in range(0, len(list)):
                if j%50!=0:
                    continue
                self.instancelist.append(path+ 'instance_label/' + list[j] + '.png')
                self.imagelist.append(path + 'img/' + list[j] + '.jpg')
                self.seglist.append(path + 'seglabel/' + list[j] + '.png')
                self.cenlist.append(path + 'centerpoint/' + list[j] + '.png')
        print("We have %d annotated image" % (len(self.imagelist)))

    def __len__(self):
        return len(self.instancelist)

    def __getitem__(self, idx):
        tempim = cv2.imread(self.imagelist[idx])
        tempim = cv2.cvtColor(tempim, cv2.COLOR_BGR2RGB)
        tempinstance = cv2.imread(self.instancelist[idx])
        tempseg = cv2.imread(self.seglist[idx])
        tempcen = cv2.imread(self.cenlist[idx])
        tempseg = tempseg[:, :, 0]
        tempinstance = tempinstance[:, :, 0]
        tempcen = tempcen[:, :, 0].astype(np.float32)
        #########gaussian blur#########
        pos = np.where(tempcen == 1)
        pos = np.stack(pos)
        for idx in range(0, pos.shape[1]):
            tempcen = draw_gaussian(tempcen, (pos[1, idx], pos[0, idx]), 10)

        if np.random.rand() < cfg.Dataprocess_cfg.prob and cfg.Dataprocess_cfg.flip:
            tempim, tempseg, tempinstance, tempcen = Flip(tempim, tempseg, tempinstance, tempcen)
        if np.random.rand() < cfg.Dataprocess_cfg.prob and cfg.Dataprocess_cfg.transition:
            tempim, tempseg, tempinstance, tempcen = Translation(tempim, tempseg, tempinstance, tempcen)
        if np.random.rand() < cfg.Dataprocess_cfg.prob and cfg.Dataprocess_cfg.augment_photometric_distort:
            tempim = Change_intensity(tempim)

        tempim = (tempim - cfg.Dataprocess_cfg.dataMean)

        # pos=positionalencoding2d(4, 256, 512)
        input = torch.from_numpy(np.ascontiguousarray(tempim.transpose(2, 0, 1))).float()
        # input=torch.cat([input,pos],dim=0)
        instancegt = torch.from_numpy(np.ascontiguousarray(tempinstance))
        label = torch.from_numpy(np.ascontiguousarray(tempseg))
        cenlabel = torch.from_numpy(np.ascontiguousarray(tempcen))

        # tempim = cv2.resize(tempim, (tempcen.shape[1], tempcen.shape[0]))
        # plt.figure("im")
        # plt.imshow(tempim / 255)
        #
        # plt.figure("ins")
        # plt.imshow(tempim[:, :, 0] / 255 + tempinstance)
        #
        # plt.figure("seg")
        # plt.imshow(tempseg)
        #
        # plt.figure("cen")
        # plt.imshow(tempim[:, :, 0] / 255 + tempcen)
        # #
        # plt.show()
        return input,instancegt,label,cenlabel





#################################################################################################################
## Add Gaussian noise
#################################################################################################################
def Gaussian(im):
    m = (0, 0, 0)
    s = (20, 20, 20)
    img=np.zeros(im.shape)
    cv2.randn(img, m, s)
    test_image = im + img
    return test_image


#################################################################################################################
## Change intensity
#################################################################################################################
def Change_intensity(im):

    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    value = int(random.uniform(-60.0, 60.0))
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -1 * value
        v[v < lim] = 0
        v[v >= lim] -= lim
    final_hsv = cv2.merge((h, s, v))
    test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return test_image


## Flip
#################################################################################################################
def Flip(im,seg,instance,cen):
    temp_image = cv2.flip(im, 1)
    temp_seg=cv2.flip(seg,1)
    temp_instance=cv2.flip(instance,1)
    temp_cen=cv2.flip(cen,1)
    return temp_image,temp_seg,temp_instance,temp_cen

#################################################################################################################
## Translation
#################################################################################################################
def Translation(im,seg,instance,cen):
        tx = np.random.randint(-50, 50)
        ty = np.random.randint(-30, 30)

        temp_image = cv2.warpAffine(im, np.float32([[1, 0, tx], [0, 1, ty]]),
                                    (im.shape[1], im.shape[0]))

        temp_seg = cv2.warpAffine(seg, np.float32([[1, 0, tx], [0, 1, ty]]),
                                    (seg.shape[1], seg.shape[0]))
        temp_instance = cv2.warpAffine(instance, np.float32([[1, 0, tx], [0, 1, ty]]),
                                    (instance.shape[1], instance.shape[0]))

        temp_cen = cv2.warpAffine(cen, np.float32([[1, 0, tx], [0, 1, ty]]),
                                       (cen.shape[1], cen.shape[0]))
        return temp_image,temp_seg,temp_instance,temp_cen

#################################################################################################################
## Rotate
#################################################################################################################
def Rotate(im,seg,instance,cen):

        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((im.shape[0], im.shape[1]), angle, 1)

        temp_image = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
        temp_seg = cv2.warpAffine(seg, M, (seg.shape[1], seg.shape[0]))
        temp_instance = cv2.warpAffine(instance, M, (instance.shape[1], instance.shape[0]))
        temp_cen = cv2.warpAffine(cen, M, (cen.shape[1], cen.shape[0]))

        return temp_image, temp_seg, temp_instance,temp_cen


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    heatmap[y - top:y + bottom, x - left:x + right]=masked_heatmap
    return heatmap
