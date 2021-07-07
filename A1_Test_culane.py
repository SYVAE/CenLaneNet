from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils.centernet as cennet
from evaluation.evaluation_culane import eval_lane
import cv2
import time
import model_zoo.LaneNet as LaneNet
from Configs import config as cfg
import  os
from utils.color import *
import torch

dataMean = cfg.Dataprocess_cfg.dataMean
Debug=0

interval=1

def forevaluation(config):
    model = LaneNet.LaneNet(config.The_selected_backbone, config)
    try:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        tempmodel = torch.load(config.testmodel, map_location="cuda:0")
        for k, v in tempmodel.items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("loaded")
    except:
        model.load_state_dict(torch.load(config.testmodel, map_location="cuda:0"))
        print("loaded")

    model.cuda()
    model.eval()
    count = 0
    #################################################
    PX = Variable(torch.zeros(cfg.Dataprocess_cfg.gtSize[0], cfg.Dataprocess_cfg.gtSize[1], 1)).cuda()
    PY = Variable(torch.zeros(cfg.Dataprocess_cfg.gtSize[0], cfg.Dataprocess_cfg.gtSize[1], 1)).cuda()
    for i in range(0, cfg.Dataprocess_cfg.gtSize[1]):
        PX[:, i, 0] = i
    for i in range(0, cfg.Dataprocess_cfg.gtSize[0]):
        PY[i, :, 0] = i
    PP = torch.cat([PY, PX], dim=2)
    print(PP.shape)
    ################################################
    torch.cuda.empty_cache()
    cv2.namedWindow("final", cv2.WINDOW_NORMAL)

    with torch.no_grad():
        with open(config.CUlane_dataroot + config.CUlane_validation, 'r') as f:
            while (True):
                line = f.readline()
                if not line:
                    break
                count = count + 1

                temp = line.split('\n')
                a = temp[0]
                imname = a.strip(' ')
                subname = a.split('.')
                dirname = a.split('/')
                dirname = '.'+dirname[0] + '/' + dirname[1] + '/' + dirname[2] + '/'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                resultname = '.'+subname[0] + '.' + subname[1] + '.lines.txt'

                if not os.path.exists(config.CUlane_dataroot +'test/'+ imname):
                    continue

                image = cv2.imread(config.CUlane_dataroot + 'test/'+ imname)
                image = cv2.resize(image, (cfg.Dataprocess_cfg.imgSize[1], cfg.Dataprocess_cfg.imgSize[0]))
                tempim = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (cfg.Dataprocess_cfg.gtSize[1], cfg.Dataprocess_cfg.gtSize[0]))
                tempim = (tempim - cfg.Dataprocess_cfg.dataMean)

                input = torch.from_numpy(tempim.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
                #################################forward##################################
                t = time.clock()
                output = model.forward(input)
                torch.cuda.synchronize(0)
                # print("time:%f(s)" % (time.clock() - t))
                #################################forward##################################
                ###################show seg#############################
                t = time.clock()
                segres = output[1]
                seg_res = segres
                out = torch.argmax(seg_res.squeeze(0), dim=0)
                segmask = out
                ####################show instance##########################
                insres = output[0]
                insres = insres / insres.norm(dim=1).unsqueeze(1)
                ins_res = insres[0, :, :, :]
                ##################show center###########################
                cen_seg_res = output[2]
                cen_seg_res = cennet._nms(cen_seg_res, 5)
                scores, inds, clses, ys, xs, total_center = cennet._topk2(cen_seg_res, ins_res, K=40)
                ####################################initialize the weight and conv#######################################
                weights = Variable(torch.empty(len(total_center), ins_res.shape[0], 1, 1)).cuda()
                for m in range(0, len(total_center)):
                    weights[m, :, 0, 0] = torch.from_numpy(total_center[m].transpose()).cuda()

                t = time.clock()
                classificationres = torch.nn.functional.conv2d(insres, weights)
                torch.cuda.synchronize()

                tempa = classificationres[0, :, :, :]
                tempc = torch.argmax(tempa, dim=0)
                tempb = torch.max(tempa, dim=0)

                t = time.clock()
                with open(resultname, 'w') as result_f:
                    ratioh = cfg.Dataprocess_cfg.CUlane_im_height / cfg.Dataprocess_cfg.gtSize[0]
                    ratiow = cfg.Dataprocess_cfg.CUlane_im_width / cfg.Dataprocess_cfg.gtSize[1]
                    for m in range(0, weights.shape[0]):
                        mask1 = tempc == m
                        mask2 = segmask == 1
                        mask3 = tempb[0] > config.CosThresh2
                        mask = mask1 & mask2 & mask3
                        mask = mask.squeeze(0).squeeze(0)
                        mask = mask > 0
                        pos = PP[mask]
                        pos = pos.cpu().numpy().transpose()
                        if pos.size == 0:
                            continue
                        if pos.shape[1] < config.minNum:
                            continue
                        # 1. no fit
                        newposx = []
                        newposy = []
                        for row in range(cfg.Dataprocess_cfg.gtSize[0], int(
                                cfg.Dataprocess_cfg.gtSize[0] - 17 * 20 / 350 * cfg.Dataprocess_cfg.gtSize[0]),
                                         -3):
                            idx = np.where(pos[0, :] == int(row))
                            if np.size(idx):
                                newposy.append(row)
                                a = np.sort(pos[1, idx[0]])
                                newposx.append(np.median(a))
                                avex = np.mean(pos[1, idx[0]])
                                raw_size_h = int(row * ratioh)
                                raw_size_r = avex * ratiow
                                result_f.write(str(raw_size_r) + ' ' + str(raw_size_h) + ' ')
                                cv2.line(image, (int(avex), int(row)), (int(avex), int(row)), [255, 255, 1],
                                         thickness=3)
                        result_f.write('\n')
                torch.cuda.synchronize(0)
                cv2.imshow("final", image)
                cv2.waitKey(1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax_flag", default=0, type=int)
    parser.add_argument("--scale_factor", default=30)
    parser.add_argument("--margin", default=0.57, type=float)
    parser.add_argument("--CosThresh2", default=0.8, type=float)
    parser.add_argument("--minNum", default=600, type=int)
    ######loss_ratio####
    parser.add_argument("--The_selected_backbone", default='efficientnet-b6',
                        choices=['resnet18', 'efficientnet-b6', 'ERF'])
    '''validation'''
    parser.add_argument("--CUlane_dataroot", default='/home/sunyi/sy/data/LaneDetection/CULane/',type=str)
    parser.add_argument("--CUlane_validation", default="list/test_split/test.txt")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--testmodel', default="./model/efficient-b6_culane/model.pkl")
    config=parser.parse_args()
    forevaluation(config)
    eval_lane(config.CUlane_dataroot, "./",ifvalidation=False)

