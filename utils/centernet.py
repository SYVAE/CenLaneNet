import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


# copy from centernet

def _neg_loss(preds, cengt):
    loss = 0
    # for pred,gt in zip(preds,cengt.unsqueeze(0)):
    # print(cengt.shape)
    for i in range(preds.shape[0]):
        gt = cengt[i, :, :]
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        pred = preds[i, :, :, :].squeeze()

        # plt.figure("one hot")
        # plt.imshow(gt.squeeze().detach().cpu().numpy())
        # plt.pause(0)
        # print(neg_inds.shape)

        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep



def _topk2(scores, ins_res, K=20, CosThresh=0.5):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds // (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).long()
    topk_xs = (topk_inds % width).long()

    topk_feature = np.zeros([K, ins_res.shape[0]])
    tmpins_res=ins_res.cpu().numpy()
    for m in range(0, topk_ys.shape[1]):
        topk_feature[m, :] = tmpins_res[:, topk_ys[0, m], topk_xs[0, m]]

    ####Proposal point######
    matrix=np.matmul(topk_feature,topk_feature.transpose())


    proposal_center = []
    proposal_x = []
    proposal_y = []
    stop = matrix.shape[0]
    mask=matrix>CosThresh
    pos=np.arange(0,matrix.shape[1])
    for mm in range(0, stop):
        # a = mask[mm, :]  # obtain the row vec
        b = pos[mask[mm, :] ]
        maxidx = np.argmax(Variable(topk_scores[0, b]).cpu().detach().numpy())
        propsal = b[maxidx]
        if propsal == mm:
            proposal_center.append(topk_feature[mm, :])
            proposal_x.append(topk_xs[0, mm])
            proposal_y.append(topk_ys[0, mm])

    proposal_x = Variable(torch.stack(proposal_x)).cpu().detach().numpy().astype(np.int)
    proposal_y = Variable(torch.stack(proposal_y)).cpu().detach().numpy().astype(np.int)

    return topk_scores, topk_inds, topk_clses, proposal_y, proposal_x, proposal_center






def _topk(scores, ins_res, K=20, CosThresh=0.5):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds // (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_feature = np.zeros([K, ins_res.shape[2]])

    for m in range(topk_ys.shape[1]):
        tempfeature = ins_res[topk_ys[0, m].int(), topk_xs[0, m].int(), :]
        topk_feature[m, :] = tempfeature

    ####Proposal point######
    matrix = np.matmul(topk_feature, topk_feature.transpose())

    # plt.figure("m")
    # plt.imshow(matrix)
    # plt.show()
    # plt.pause(0)

    proposal_center = []
    proposal_x = []
    proposal_y = []
    stop = matrix.shape[0]
    for mm in range(0, stop):
        # print(mm)
        if matrix[mm, mm] != 0:
            a = matrix[mm, :]  # obtain the row vec
            b = np.where(a >= CosThresh)
            maxidx = np.argmax(Variable(topk_scores[0, b]).cpu().detach().numpy())
            propsal = b[0][maxidx]
            if propsal == mm:
                proposal_center.append(topk_feature[mm, :])
                proposal_x.append(topk_xs[0, mm])
                proposal_y.append(topk_ys[0, mm])
    proposal_x = Variable(torch.stack(proposal_x)).cpu().detach().numpy().astype(np.int)
    proposal_y = Variable(torch.stack(proposal_y)).cpu().detach().numpy().astype(np.int)

    return topk_scores, topk_inds, topk_clses, proposal_y, proposal_x, proposal_center