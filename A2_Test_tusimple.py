from torch.autograd import Variable
import utils.centernet as cennet
import json
import torch
import time
import model_zoo.LaneNet as LaneNet
from Configs import config as cfg
from evaluation.evaluation_Tusimple import LaneEval
from utils.postprocess import *

dataMean = cfg.Dataprocess_cfg.dataMean
def forevaluation(config):
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
        test_json=raw_data
        for i in range(0, len(raw_data)):
            print(">>>processing the %d(%d)th image" % (i, len(raw_data)))
            image = cv2.imread(config.TusimpleTesting_root + raw_data[i]["raw_file"])

            temp_image = cv2.resize(image, (inputW, inputH))
            tempim = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
            tempim = (tempim - dataMean)

            input = torch.from_numpy(tempim.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
            #################################forward##################################
            torch.cuda.synchronize(0)
            t = time.clock()
            output = model.forward(input)
            torch.cuda.synchronize(0)
            print("time:%f(s)" % (time.clock() - t))
            #################################forward##################################
            ###################show seg#############################
            segres = output[1]
            seg_res = segres[0, :, :, :]
            seg_res = seg_res.cpu().squeeze(0)
            seg_res = seg_res.detach().numpy().transpose((1, 2, 0))
            out = seg_res
            out = np.argmax(out, axis=2)
            step = np.arange(0, out.shape[0] - 1, 4).astype(int)
            selectionmask = np.zeros(out.shape)
            selectionmask[step, :] = 1
            segmask=out
            ###################show seg#############################

            ####################show instance##########################
            insres = output[0]
            insres = insres / insres.norm(dim=1).unsqueeze(1)
            ins_res = insres[0, :, :, :]
            ins_res = ins_res.cpu().squeeze(0)
            ins_res = ins_res.detach().numpy().transpose((1, 2, 0))
            cen_seg_res = output[2]

            cen_seg_res = cennet._nms(cen_seg_res, 11)
            scores, inds, clses, ys, xs, total_center = cennet._topk(cen_seg_res, ins_res, K=20)

            ###################post  process##############################
            ins_res[out == 0, :] = 0
            ###############initialize the weight and conv#######################################
            weights = Variable(torch.empty(len(total_center), ins_res.shape[2], 1, 1)).cuda()
            print('weights:', weights.shape)
            for m in range(0, len(total_center)):
                weights[m, :, 0, 0] = torch.from_numpy(total_center[m].transpose()).cuda()

            classificationres = torch.nn.functional.conv2d(insres, weights)
            # debug
            tempa = classificationres[0, :, :, :]
            tempb = tempa.cpu().detach().numpy()
            tempc = np.argmax(tempb, axis=0)
            tempv = np.max(tempb, axis=0)

            cv2.namedWindow("final", cv2.WINDOW_NORMAL)
            for m in range(0, weights.shape[0]):
                mask1 = tempc == m
                mask2 = out == 1
                mask3 = tempv > config.CosThresh2
                mask = mask1 & mask2 & mask3
                mask = cv2.resize(mask.astype(np.uint8), dsize=(image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

                pos = np.stack(np.where(mask == 1))
                if pos.shape[1] < config.minNum:
                    continue

                pos[0, :] = pos[0, :]
                pos[1, :] = pos[1, :]
                idx = np.argsort(pos[0, :], axis=0)
                pos[0, :] = pos[0, idx]
                pos[1, :] = pos[1, idx]

                x = raw_data[i]['h_samples']
                showy = pos[0, :]
                maxyidx = np.argmax(showy)
                minidx = np.argmin(showy)

                showx = pos[1, :]
                minpos = showy[minidx]

                prex = []
                angles = LaneEval.get_angle(showx, showy)
                print(angles)
                for n in range(0, len(x)):
                    idx = np.where(showy == x[n])
                    if x[n] <= minpos:
                        prex.append(-2)
                    elif np.size(idx):
                        posx = np.mean(showx[idx[0]])
                        posy = x[n]
                        cv2.line(image, (int(posx), posy), (int(posx), posy), [0, 0, 255], thickness=5)

                        if posx < 0 or posx > image.shape[1]:
                            prex.append(-2)
                        else:
                            prex.append(posx.astype(np.float))
                    else:
                        higheridx = np.where(showy < x[n])
                        loweridx = np.where(showy > x[n])
                        if np.size(higheridx) and np.size(loweridx):
                            hidx = len(higheridx) - 1
                            lidx = len(loweridx) - 1

                            hx = showx[higheridx[0][hidx]]
                            hy = showy[higheridx[0][hidx]]
                            lx = showx[loweridx[0][lidx]]
                            ly = showy[loweridx[0][lidx]]
                            k = (hx - lx) * 1.0 / (hy - ly)
                            posx = int(k * (x[n] - hy) + hx)
                            posy = x[n]
                            cv2.line(image, (int(posx), posy), (int(posx), posy), [0, 0, 255], thickness=5)
                            if posx < 0 or posx > image.shape[1]:
                                prex.append(-2)
                            else:
                                prex.append(posx)
                        elif np.size(higheridx):

                            f2 = np.polyfit(pos[0, pos.shape[1] // 2:pos.shape[1]],
                                            pos[1, pos.shape[1] // 2:pos.shape[1]], 1)

                            posx = np.polyval(f2, x[n])
                            posy = x[n]
                            cv2.line(image, (int(posx), posy), (int(posx), posy), [0, 0, 255], thickness=5)
                            if posx < 0 or posx > image.shape[1]:
                                prex.append(-2)
                            else:
                                prex.append(posx)
                        elif np.size(loweridx) and x[n] >= 260:

                            f2 = np.polyfit(pos[0, 0:pos.shape[1] // 5], pos[1, 0:pos.shape[1] // 5], 1)

                            posx = np.polyval(f2, x[n])
                            posy = x[n]
                            cv2.line(image, (int(posx), posy), (int(posx), posy), [0, 0, 255], thickness=5)
                            if posx < 0 or posx > image.shape[1]:
                                prex.append(-2)
                            else:
                                prex.append(posx)
                        else:
                            prex.append(-2)

                test_json[i]["lanes"].append(prex)
            test_json[i]['run_time'] = 1

            cv2.imshow("final", image)
            cv2.waitKey(1)

        save_result(test_json, "test_result.json")
    print(LaneEval.bench_one_submit("test_result.json", "test_label.json"))

def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", default=30)
    parser.add_argument("--margin", default=0.6, type=float)
    parser.add_argument("--CosThresh2", default=0.8, type=float)
    parser.add_argument("--minNum", default=1000, type=int)
    ######loss_ratio####
    parser.add_argument("--The_selected_backbone", default='efficientnet-b6',
                        choices=['resnet18', 'efficientnet-b6', 'ERF'])
    '''validation'''
    parser.add_argument("--TusimpleTesting_root", default="./test_set/",type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--testmodel', default="./model/efficient-b6_tusimple/model.pkl")
    config = parser.parse_args()
    torch.cuda.set_device(config.device)
    forevaluation(config)
