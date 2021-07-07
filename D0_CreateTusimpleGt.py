'''Tusimple Datasets loader'''
import json
import numpy as np
import torch
import cv2
import random
import matplotlib as plt
import matplotlib.pyplot
import os
import Configs.config as cfg
import argparse

class Dataloader():
    '''__Format__
    {
      'raw_file': str. Clip file path.
      'lanes': list. A list of lanes. For each list of one lane, the elements are width values on image.
      'h_samples': list. A list of height values corresponding to the 'lanes', which means len(h_samples) == len(lanes[i])
    }
'''
    def __init__(self,tusimpleData_root,processedroot,ifinitialization=True):
        self.tusimpleData_root=tusimpleData_root
        self.gtroot=processedroot
        if ifinitialization:
            self.raw_data=[]#tusimple format json file
            with open(self.tusimpleData_root+'label_data_0313.json')as f:
                while True:
                    line=f.readline()
                    if not line:
                        break
                    temp=json.loads(line)
                    self.raw_data.append(temp)

            with open(self.tusimpleData_root + 'label_data_0531.json')as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    temp = json.loads(line)
                    self.raw_data.append(temp)

            with open(self.tusimpleData_root + 'label_data_0601.json')as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    temp = json.loads(line)
                    self.raw_data.append(temp)
            print("total number of the annotation:%d"%(len(self.raw_data)))
        if not os.path.exists(self.gtroot + 'img/'):
            os.mkdir(self.gtroot + 'img/')
        if not os.path.exists(self.gtroot + 'seglabel/'):
            os.mkdir(self.gtroot + 'seglabel/')
        if not os.path.exists(self.gtroot + 'instance_label/'):
            os.mkdir(self.gtroot + 'instance_label/')
        if not os.path.exists(self.gtroot + 'centerpoint/'):
            os.mkdir(self.gtroot + 'centerpoint/')
    def gaussian2D(self,shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian(self,heatmap, center, radius, k=1, delte=6):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / delte)

        x, y = center

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        heatmap[y - top:y + bottom, x - left:x + right]=masked_heatmap
        return heatmap

    def create_mydataset(self,inputsize,gtsize,thickness,debug=False):
        if debug:
            cv2.namedWindow("im",cv2.WINDOW_NORMAL)
        for i in range(0,len(self.raw_data)):
            print(">>>processing the %d(%d)th image"%(i,len(self.raw_data)))
            temp_image=cv2.imread(self.tusimpleData_root+self.raw_data[i]["raw_file"])
            ratiow=gtsize[1]*1.0/temp_image.shape[1]
            ratioh = gtsize[0] * 1.0 / temp_image.shape[0]

            temp_image=cv2.resize(temp_image,(inputsize[1],inputsize[0]))
            temp_seg=np.zeros((gtsize[0],gtsize[1]))
            temp_instance=np.zeros((gtsize[0],gtsize[1]))
            temp_center=np.zeros((gtsize[0],gtsize[1]))
            temp_point=self.raw_data[i]['lanes']

            laneidx=0
            for j in temp_point:
                laneidx=laneidx+1
                cols=(np.array(j).astype(np.float))
                rows=(np.array(self.raw_data[i]['h_samples']).astype(np.float))
                mask=cols>=0
                cols=cols[mask]
                rows=rows[mask]
                if rows.shape[0]==0:
                    continue
                maxidx=np.argmax(rows)
                minidx=np.argmin(rows)
                center_row=0.5*(rows[minidx]+rows[maxidx])
                p1=np.polyfit(rows,cols,3)
                center_col=np.polyval(p1,center_row)

                cols = (cols * ratiow).astype('int16')
                rows = (rows * ratioh).astype('int16')
                center_col=(center_col* ratiow).astype('int16')
                center_row=(center_row*ratioh).astype('int16')
                cv2.circle(temp_center, (center_col.astype(np.int16), center_row.astype(np.int16)), color=[1, 1, 1], radius=1, thickness=-1)


                for num in range(0,len(cols)-1):
                    if  cols[num]>0 and cols[num+1]>0:
                        cv2.line(temp_seg, (cols[num], rows[num]), (cols[num + 1], rows[num + 1]), [1, 1, 1], thickness)
                        cv2.line(temp_instance, (cols[num], rows[num]), (cols[num + 1], rows[num + 1]), [laneidx, laneidx, laneidx],thickness,cv2.LINE_AA)

            cv2.imwrite(self.gtroot+'img/'+str(i)+'.jpg',temp_image)
            cv2.imwrite(self.gtroot + 'instance_label/' + str(i) + '.png', temp_instance)
            cv2.imwrite(self.gtroot + 'seglabel/' + str(i) + '.png', temp_seg)
            cv2.imwrite(self.gtroot + 'centerpoint/' + str(i) + '.png',temp_center)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--GtDataroot",default="./",type=str)
    parser.add_argument("--tusimple_root",default="/home/sunyi/sy/data/LaneDetection/tusimple_raw/tosunyi/train_set/",type=str)
    config=parser.parse_args()
    a=Dataloader(tusimpleData_root=config.tusimple_root,processedroot=config.GtDataroot)
    a.create_mydataset(cfg.Dataprocess_cfg.imgSize,cfg.Dataprocess_cfg.gtSize,2,debug=False)
