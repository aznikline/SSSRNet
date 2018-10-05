from __future__ import print_function
import torch
import argparse
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from os import listdir
from os.path import join
import os
import time, math
from torchvision import transforms
import dataload
import numpy as np
from math import log10
from deeplab.model import Res_Deeplab
from srresnet_semantic1 import Net, mid_layer, CrossEntropy_Probability, CrossEntropy2d, KL_Loss
from deeplab.datasets import VOCDataSSSet
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
import visualization as vl
import matplotlib.pyplot as plt
from collections import OrderedDict

##########################
# Author : Hang
#
# Date: Jan, 16, 2018
#
# Version: 1.0
#
# Description: Test SSSRNet with  class-specific PSNR
#
# Details: In Python3.6 the dict is ordered.
##########################

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/SSSRNet5_baseline2_concate_lr_epoch_50.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]
DATA_LIST_PATH = '/tmp4/hang_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
NUM_CLASSES = 21
gpuid = 1
classes = {'background': [0, 0, 0],
           'aeroplane': [0, 0, 0],
           'bicycle': [0, 0, 0],
           'bird': [0, 0, 0],
           'boat': [0, 0, 0],
           'bottle': [0, 0, 0],
           'bus': [0, 0, 0],
           'car': [0, 0, 0],
           'cat': [0, 0, 0],
           'chair': [0, 0, 0],
           'cow': [0, 0, 0],
           'diningtable': [0, 0, 0],
           'dog': [0, 0, 0],
           'horse': [0, 0, 0],
           'motorbike': [0, 0, 0],
           'person': [0, 0, 0],
           'pottedplant': [0, 0, 0],
           'sheep': [0, 0, 0],
           'sofa': [0, 0, 0],
           'train': [0, 0, 0],
           'tvmonitor': [0, 0, 0],
           'all': [0, 0, 0]}


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')



def test(testloader, deeplab, mid, criterion, criterion1, criterion_kl):

    data_lr_list = []
    data_sr_list = []
    data_hr_list = []
    is_show = False
    avgloss_lr = 0
    avgloss_sr = 0
    avgloss_hr = 0
    avgpsnr = 0
    for iteration, batch in enumerate(testloader, 1):
        input_ss, input = Variable(batch[0], volatile=True), Variable(batch[1],volatile=True)
        target_ss, target = Variable(batch[2], volatile=True), Variable(batch[3], volatile=True)
        pred_ss, pred = Variable(batch[4], volatile=True), Variable(batch[5], volatile=True)
        label_lr, label_hr = Variable(batch[6]), Variable(batch[7])
        name = batch[9]
        print(iteration)
        mse = torch.nn.MSELoss(size_average=True)(pred, target)
        print('PSNR:{}'.format(10 * log10(1 / mse.data[0])))
        avgpsnr += 10 * log10(1 / mse.data[0])
        ###############################LR Scale#########################################
        input_ss = input_ss.cuda(gpuid)
        seg_lr = deeplab(input_ss)
        ###########Cal LR CrossEntropy Loss############
        Label_patch = label_lr.cpu().data[:].numpy()
        label_pro = np.repeat(Label_patch, 21, axis=1)
        for i in range(21):
            tmp = label_pro[:, i - 1:i]
            if i == 0:
                tmp[tmp == 255] = 0
            tmp[tmp != i] = -1
            tmp[tmp == i] = 1
            tmp[tmp == -1] = 0
        label_lr_gt = Variable(torch.from_numpy(label_pro[:, :, :, :]).float()).cuda(gpuid)
        label_lr_gt1 = Variable(torch.from_numpy(np.argmax(label_pro[:, :, :, :], axis=1))).cuda(gpuid)
        size = (input_ss.size()[2], input_ss.size()[3])
        #avgloss_lr += criterion(seg_lr, label_lr_gt, size)
        #avgloss_lr += criterion1(seg_lr, label_lr_gt1, size)
        avgloss_lr += criterion_kl(seg_lr, label_lr_gt, size)
        avgloss_lr += loss_calc(pred, labels, args.gpu)
        ##############################################

        #print('loss: {}'.format(loss))
        seg_lr = mid(seg_lr, size)
        seg_lr = seg_lr.cpu().data[0].numpy()
        seg_lr = seg_lr.transpose(1,2,0)
        seg_lr = np.asarray(np.argmax(seg_lr, axis=2), dtype=np.int)
        gt_lr = np.asarray(label_lr.cpu().data[0,0].numpy(), dtype=np.int)
        #result = transforms.ToPILImage()(SR.cpu().data[0])
        #path = join(SR_dir, name+'.png')
        #result.save(path)
        if is_show:
            vl.show_label(gt_lr, seg_lr)
        data_lr_list.append([gt_lr.flatten(), seg_lr.flatten()])
        ###############################LR Scale#########################################

        ###############################SR Scale#########################################
        pred_ss = pred_ss.cuda(gpuid)
        seg_sr = deeplab(pred_ss)
        ###########Cal LR CrossEntropy Loss############
        Label_patch = label_hr.cpu().data[:].numpy()
        label_pro = np.repeat(Label_patch, 21, axis=1)
        for i in range(21):
            tmp = label_pro[:, i - 1:i]
            if i == 0:
                tmp[tmp == 255] = 0
            tmp[tmp != i] = -1
            tmp[tmp == i] = 1
            tmp[tmp == -1] = 0
        label_hr_gt = Variable(torch.from_numpy(label_pro[:, :, :, :]).float()).cuda(gpuid)
        label_hr_gt1 = Variable(torch.from_numpy(np.argmax(label_pro[:, :, :, :], axis=1))).cuda(gpuid)
        size = (pred_ss.size()[2], pred_ss.size()[3])
        #avgloss_sr += criterion(seg_sr, label_hr_gt, size)
        #avgloss_sr += criterion1(seg_sr, label_hr_gt1, size)
        avgloss_sr += criterion_kl(seg_sr, label_hr_gt, size)
        ##############################################

        #print('loss: {}'.format(loss))
        seg_sr = mid(seg_sr, size)
        seg_sr = seg_sr.cpu().data[0].numpy()
        seg_sr = seg_sr.transpose(1,2,0)
        seg_sr = np.asarray(np.argmax(seg_sr, axis=2), dtype=np.int)
        gt_hr = np.asarray(label_hr.cpu().data[0,0].numpy(), dtype=np.int)
        #result = transforms.ToPILImage()(SR.cpu().data[0])
        #path = join(SR_dir, name+'.png')
        #result.save(path)
        if is_show:
            vl.show_label(gt_hr, seg_sr)
        data_sr_list.append([gt_hr.flatten(), seg_sr.flatten()])
        ###############################SR Scale#########################################

        ###############################HR Scale#########################################
        target_ss = target_ss.cuda(gpuid)
        seg_hr = deeplab(target_ss)
        ###########Cal LR CrossEntropy Loss############
        size = (target_ss.size()[2], target_ss.size()[3])
        #avgloss_hr += criterion(seg_hr, label_hr_gt, size)
        #avgloss_hr += criterion1(seg_hr, label_hr_gt1, size)
        avgloss_hr += criterion_kl(seg_hr, label_hr_gt, size)
        ##############################################

        #print('loss: {}'.format(loss))
        seg_hr = mid(seg_hr, size)
        seg_hr = seg_hr.cpu().data[0].numpy()
        seg_hr = seg_hr.transpose(1,2,0)
        seg_hr = np.asarray(np.argmax(seg_hr, axis=2), dtype=np.int)
        #result = transforms.ToPILImage()(SR.cpu().data[0])
        #path = join(SR_dir, name+'.png')
        #result.save(path)
        if is_show:
            vl.show_label(gt_hr, seg_hr)
        data_hr_list.append([gt_hr.flatten(), seg_hr.flatten()])
        ###############################HR Scale#########################################
        image = target.cpu().data[0].numpy().transpose((1, 2, 0))
        #vl.show_ssloss_seg1(image, gt_hr, seg_hr, seg_sr)


    print('LR CrossEntropy Loss: {}'.format(avgloss_lr.data[0]/iteration))
    get_iou(data_lr_list, NUM_CLASSES)

    print('SR CrossEntropy Loss: {}'.format(avgloss_sr.data[0]/iteration))
    get_iou(data_sr_list, NUM_CLASSES)

    print('HR CrossEntropy Loss: {}'.format(avgloss_hr.data[0]/iteration))
    get_iou(data_hr_list, NUM_CLASSES)

    print('SR PSNR: {}'.format(avgpsnr/iteration))

opt = parser.parse_args()
global LR_dir, SR_dir, HR_dir
print(opt)

print("===> Loading datasets")
# root_dir = '/tmp4/hang_data/DIV2K'
root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012'
LR_dir = join(root_dir, 'SS_Output/LR')
if os.path.isdir(LR_dir):
    pass
else:
    os.mkdir(LR_dir)

SR_dir = join(root_dir, 'SS_Output/SR')
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

HR_dir = join(root_dir, 'SS_Output/HR')
if os.path.isdir(HR_dir):
    pass
else:
    os.mkdir(HR_dir)

deeplab_res = Res_Deeplab(num_classes=21)
saved_state_dict = torch.load('model/VOC12_scenes_20000.pth')
deeplab_res.load_state_dict(saved_state_dict)
deeplab_res = deeplab_res.eval()
mid = mid_layer()
criterion = CrossEntropy_Probability()
criterion1 = CrossEntropy2d()
criterion_kl = KL_Loss()

mid = mid.cuda(gpuid)
deeplab_res = deeplab_res.cuda(gpuid)
criterion = criterion.cuda(gpuid)
criterion1 = criterion1.cuda(gpuid)
criterion_kl = criterion_kl.cuda(gpuid)

testloader = data.DataLoader(
    VOCDataSSSet(root_dir, DATA_LIST_PATH, crop_size=(321, 321), mean=IMG_MEAN, scale=False, mirror=False),
    batch_size=1, shuffle=False, pin_memory=True)

test(testloader, deeplab_res, mid, criterion, criterion1, criterion_kl)


