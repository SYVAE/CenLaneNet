from torch.autograd import Variable
import utils.centernet as cennet
from evaluation.evaluation_culane import eval_lane
import cv2
from Configs import config as cfg
import  os
from utils.utils import *
import Configs.config as Cfg
import torch
#best

def validation(configs,model):
    data_root = configs.CUlane_dataroot
    test_file = configs.CUlane_dataroot+configs.CUlane_validation
    dataMean=Cfg.Dataprocess_cfg.dataMean

    count = 0
    #################################################
    PX = Variable(torch.zeros(cfg.Dataprocess_cfg.gtSize[0], cfg.Dataprocess_cfg.gtSize[1], 1)).cuda()
    PY = Variable(torch.zeros(cfg.Dataprocess_cfg.gtSize[0], cfg.Dataprocess_cfg.gtSize[1], 1)).cuda()
    for i in range(0, cfg.Dataprocess_cfg.gtSize[1]):
        PX[:, i, 0] = i
    for i in range(0, cfg.Dataprocess_cfg.gtSize[0]):
        PY[i, :, 0] = i
    PP = torch.cat([PY, PX], dim=2)
    ################################################
    torch.cuda.empty_cache()
    with torch.no_grad():
        with open(test_file,'r') as f:
                while(True):
                    line = f.readline()
                    if not line:
                        break
                    count= count+1

                    temp=line.split('\n')
                    a=temp[0]
                    a=a[1:]
                    imname=a.strip(' ')
                    subname=a.split('.')
                    dirname=a.split('/')
                    dirname=dirname[0]+'/'+dirname[1]+'/'+dirname[2]+'/'
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    resultname=subname[0]+'.'+subname[1]+'.lines.txt'

                    if not os.path.exists(data_root+imname):
                         continue

                    image = cv2.imread(data_root+'/'+imname)
                    image=cv2.resize(image,(cfg.Dataprocess_cfg.imgSize[1], cfg.Dataprocess_cfg.imgSize[0]))
                    tempim = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image=cv2.resize(image,(cfg.Dataprocess_cfg.gtSize[1], cfg.Dataprocess_cfg.gtSize[0]))
                    tempim = (tempim - dataMean)/255

                    input = torch.from_numpy(tempim.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
             #################################forward##################################

                    output = model.forward(input)

                    #################################forward##################################
                    ###################show seg#############################

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
                    scores, inds, clses, ys, xs,total_center = cennet._topk2(cen_seg_res,ins_res, K=40)
                    ####################################initialize the weight and conv#######################################
                    weights = Variable(torch.empty(len(total_center), ins_res.shape[0], 1, 1)).cuda()
                    # print('weights:', weights.shape)
                    for m in range(0, len(total_center)):
                        weights[m, :, 0, 0] = torch.from_numpy(total_center[m].transpose()).cuda()


                    classificationres = torch.nn.functional.conv2d(insres, weights)

                    tempa = classificationres[0, :, :, :]
                    tempc = torch.argmax(tempa, dim=0)
                    tempb = torch.max(tempa, dim=0)

                    with open(resultname, 'w') as result_f:
                        ratioh=cfg.Dataprocess_cfg.CUlane_im_height/cfg.Dataprocess_cfg.gtSize[0]
                        ratiow = cfg.Dataprocess_cfg.CUlane_im_width / cfg.Dataprocess_cfg.gtSize[1]
                        for m in range(0, weights.shape[0]):
                            mask1 = tempc == m
                            mask2 = segmask == 1
                            mask3 = tempb[0] > configs.CosThresh2
                            mask = mask1 & mask2 & mask3
                            mask = mask.squeeze(0).squeeze(0)
                            mask = mask > 0
                            pos = PP[mask]

                            pos=pos.cpu().numpy().transpose()
                            if pos.size==0:
                                continue
                            if pos.shape[1] < configs.minNum:
                                continue
                            # 1. no fit
                            newposx=[]
                            newposy=[]
                            for row in range(cfg.Dataprocess_cfg.gtSize[0], int(cfg.Dataprocess_cfg.gtSize[0]-17 * 20 / 350 * cfg.Dataprocess_cfg.gtSize[0]), -3):
                                idx = np.where(pos[0, :] == int(row))
                                if np.size(idx):
                                    newposy.append(row)
                                    a=np.sort(pos[1, idx[0]])
                                    newposx.append(np.median(a))
                                    avex=np.mean(pos[1, idx[0]])
                                    raw_size_h = int(row * ratioh)
                                    raw_size_r = avex * ratiow
                                    result_f.write(str(raw_size_r) + ' ' + str(raw_size_h) + ' ')
                                    cv2.line(image, (avex, int(row)), (avex, int(row)), [255, 255, 1], thickness=3)
                            result_f.write('\n')
                    torch.cuda.synchronize(0)
                    cv2.imshow("final", image)
                    cv2.waitKey(1)

        F=eval_lane(data_root, "./",ifvalidation=True)
        return F
