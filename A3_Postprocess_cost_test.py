import torch
from torch.autograd.variable import Variable
import utils.centernet as cennet
import json
import time
import model_zoo.LaneNet as LaneNet
from Configs import config as cfg
from evaluation.evaluation_Tusimple import LaneEval
from utils.postprocess import *
import random
from tqdm import tqdm
from utils.color import *
def forevaluation(config):
    inputW=cfg.Dataprocess_cfg.imgSize[1]
    inputH=cfg.Dataprocess_cfg.imgSize[0]
    dataMean = cfg.Dataprocess_cfg.dataMean
    model = LaneNet.LaneNet(config.The_selected_backbone, config)
    # F = eval_lane(config.CUlane_dataroot, "./", ifvalidation=ifvalidation)
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
    #################################################
    PX = Variable(torch.zeros(cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1], 1)).cuda()
    PY = Variable(torch.zeros(cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1], 1)).cuda()
    for i in range(0, cfg.Dataprocess_cfg.imgSize[1]):
        PX[:, i, 0] = i
    for i in range(0, cfg.Dataprocess_cfg.imgSize[0]):
        PY[i, :, 0] = i
    PP = torch.cat([PY, PX], dim=2)
    ################################################
    torch.cuda.empty_cache()
    print("load...")
    model.cuda()
    model.eval()
    torch.cuda.empty_cache()

    costtime=[]
    total_costtime=[]
    with torch.no_grad():
    ########load data################
        raw_data=[]
        with open(config.TusimpleTesting_root + 'test_tasks_0627.json')as f:
            while True:
                line = f.readline()
                if not line:
                    break
                temp = json.loads(line)
                raw_data.append(temp)
        ################################
        ###my testres
        for i in tqdm(range(0, len(raw_data))):
            image = cv2.imread(config.TusimpleTesting_root + raw_data[i]["raw_file"])
            temp_image = cv2.resize(image, (inputW, inputH))
            tempim = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
            tempim = (tempim - dataMean)
            input = torch.from_numpy(tempim.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
            t0 = time.time()
            #################################forward##################################
            output = model.forward(input)
            #################################forward##################################
            ###################show seg#############################
            t = time.time()
            seg_res = (output[1])[0, :, :, :]
            segmask = torch.argmax(seg_res.squeeze(0), dim=0)
            ####################show instance##########################
            insres = output[0] / output[0].norm(dim=1).unsqueeze(1)
            ins_res = insres[0, :, :, :]
            ##################show center###########################
            cen_seg_res = cennet._nms(output[2], 11)
            scores, inds, clses, ys, xs, total_center = cennet._topk2(cen_seg_res, ins_res, K=20)
            ###################post  process##############################
            weights = Variable(torch.empty(len(total_center), ins_res.shape[0], 1, 1)).cuda()
            for m in range(0, len(total_center)):
                weights[m, :, 0, 0] = torch.from_numpy(total_center[m].transpose()).cuda()

            classificationres = torch.nn.functional.conv2d(insres, weights)
            costtime.append(time.time()-t)
            total_costtime.append(time.time()-t0)

            "show instance map"
            if config.show:
                tempa = classificationres[0, :, :, :]
                tempc = torch.argmax(tempa, dim=0)
                tempb = torch.max(tempa, dim=0)
                t = time.clock()
                label_colours = get_rgbforlabels()
                rgb = np.zeros((image.shape[0], image.shape[1], 3))
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
                    mask = mask.cpu().numpy().astype(np.float)
                    mask = np.ascontiguousarray(np.array(mask))
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
                    mask = mask > 0
                    r = (label_colours[m, 0])
                    g = (label_colours[m, 1])
                    b = (label_colours[m, 2])

                    rgb[mask, 0] = r
                    rgb[mask, 1] = g
                    rgb[mask, 2] = b

                showim2 = ((image + rgb) // 2).astype(np.uint8)
                cv2.namedWindow("instancemap",cv2.WINDOW_NORMAL)
                cv2.imshow("instancemap", showim2)
                cv2.waitKey(1)

        total_cost=np.stack(costtime)
        average=total_cost.mean()
        print("------center-based------")
        print("average_postprocessing_time:{0}".format(average))
        total_cost = np.stack(total_costtime)
        average = total_cost.mean()
        print("total_inference_time:{0}".format(average))

def forevaluation_nocen(config):
    inputW = cfg.Dataprocess_cfg.imgSize[1]
    inputH = cfg.Dataprocess_cfg.imgSize[0]
    dataMean = cfg.Dataprocess_cfg.dataMean
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
    #################################################
    PX = Variable(torch.zeros(cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1], 1)).cuda()
    PY = Variable(torch.zeros(cfg.Dataprocess_cfg.imgSize[0], cfg.Dataprocess_cfg.imgSize[1], 1)).cuda()
    for i in range(0, cfg.Dataprocess_cfg.imgSize[1]):
        PX[:, i, 0] = i
    for i in range(0, cfg.Dataprocess_cfg.imgSize[0]):
        PY[i, :, 0] = i
    PP = torch.cat([PY, PX], dim=2)
    ################################################
    torch.cuda.empty_cache()
    print("load...")
    model.cuda()
    model.eval()
    torch.cuda.empty_cache()

    costtime = []
    total_costtime = []
    # torch.backends.cudnn.benchmark = True
    with torch.no_grad():
    ########load data################
        raw_data=[]
        with open(config.TusimpleTesting_root + 'test_tasks_0627.json')as f:
            while True:
                line = f.readline()
                if not line:
                    break
                temp = json.loads(line)
                raw_data.append(temp)
        for i in tqdm(range(0, len(raw_data))):
            image = cv2.imread(config.TusimpleTesting_root + raw_data[i]["raw_file"])
            temp_image = cv2.resize(image, (inputW, inputH))
            tempim = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
            tempim = tempim - dataMean
            input = torch.from_numpy(tempim.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
            #################################forward##################################
            t0 = time.time()
            output = model.forward(input)
            #################################forward##################################

            '''--------------------------------instacne clustering--------------------'''
            t = time.time()
            seg_res = (output[1])[0, :, :, :]
            segmask = torch.argmax(seg_res.squeeze(0), dim=0)
            insres = output[0] / output[0].norm(dim=1).unsqueeze(1)
            embedding = insres[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
            lane_seg_img = embedding_post_process(embedding, segmask.cpu().numpy(),0.1, 4)
            costtime.append(time.time()-t)
            total_costtime.append(time.time()-t0)
            if config.show:
                label_colours = get_rgbforlabels()
                rgb = np.zeros((image.shape[0], image.shape[1], 3))
                for id, lane_idx in enumerate(np.unique(lane_seg_img)):
                    if lane_idx == 0:
                        continue
                    mask = lane_seg_img == lane_idx
                    mask=cv2.resize(mask.astype(np.uint8),dsize=(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
                    mask = np.ascontiguousarray(mask)
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
                    mask = mask > 0
                    r = (label_colours[id, 0])
                    g = (label_colours[id, 1])
                    b = (label_colours[id, 2])

                    rgb[mask, 0] = r
                    rgb[mask, 1] = g
                    rgb[mask, 2] = b

                showim2 = ((image + rgb) // 2).astype(np.uint8)
                cv2.namedWindow("instancemap", cv2.WINDOW_NORMAL)
                cv2.imshow("instancemap", showim2)
                cv2.waitKey(1)


        total_cost=np.stack(costtime)
        average=total_cost.mean()
        print("------mean-shift------")
        print("average_postprocessing_time:{0}".format(average))
        total_cost = np.stack(total_costtime)
        average = total_cost.mean()
        print("total_inference_time:{0}".format(average))

def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax_flag", default=0, type=int)
    parser.add_argument("--scale_factor", default=30)
    parser.add_argument("--margin", default=0.6, type=float)
    parser.add_argument("--CosThresh2", default=0.8, type=float)
    parser.add_argument("--minNum", default=1000, type=int)
    parser.add_argument("--show",default=0,type=int)
    parser.add_argument("--The_selected_backbone", default='efficientnet-b6',
                        choices=['resnet18', 'efficientnet-b6', 'ERF'])
    '''validation'''
    parser.add_argument("--TusimpleTesting_root", default="/home/sunyi/sy/data/LaneDetection/tusimple_raw/tosunyi/test_set/",type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--testmodel', default="./model/efficient-b6_tusimple/model.pkl")
    config = parser.parse_args()
    torch.cuda.set_device(config.device)
    forevaluation(config)
    forevaluation_nocen(config)