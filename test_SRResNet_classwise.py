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
from SSSRNet5_finetune import Net, mid_layer
from deeplab.datasets import VOCDataValSet
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
from visualization import show_seg
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
parser.add_argument("--model", default="model/SRResNet_VOC160_lr_finetune42.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

IMG_MEAN = [104.00698793,116.66876762,122.67891434]
DATA_LIST_PATH = '/tmp4/hang_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
NUM_CLASSES = 21
gpuid = 1
classes = {'background': [0,0],
           'aeroplane': [0,0],
           'bicycle': [0,0],
           'bird': [0,0],
           'boat': [0,0],
           'bottle': [0,0],
           'bus': [0,0],
           'car': [0,0],
           'cat': [0,0],
           'chair': [0,0],
           'cow': [0,0],
           'diningtable': [0,0],
           'dog': [0,0],
           'horse': [0,0],
           'motorbike': [0,0],
           'person': [0,0],
           'pottedplant': [0,0],
           'sheep': [0,0],
           'sofa': [0,0],
           'train': [0,0],
           'tvmonitor': [0,0],
           'all': [0,0]}

def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric import ConfusionMatrix    #混淆矩阵/误差矩阵 检测分类的正确率

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM    #f是生成的误差矩阵？
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()    #M为jaccard相似系数 越大样本相似度越高
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')

def classwise_evaluate(SR, target, label_hr, class_num, classes):
    Label_patch = label_hr.numpy()
    label_pro = np.repeat(Label_patch, class_num, axis=1)
    for i in range(class_num):
        tmp = label_pro[:, i:i + 1]
        if i == 0:
            tmp[tmp == 255] = 0
        tmp[tmp != i] = -1
        tmp[tmp == i] = 1
        tmp[tmp == -1] = 0
    label_hr = Variable(torch.from_numpy(label_pro[:, :, :, :]).float())
    label = label_hr.cuda(gpuid)

    psnr = np.zeros(class_num+1)    #峰值信噪比
    for i in range(class_num):
        mask = label[:, i:i + 1, :, :].repeat(1, 3, 1, 1)    #？？？？？？？？？  label 1通道 mask 3通道 重复3次 repeat针对tensor   i:i+1才能保持维度
                                                             #都是给  label的每个channel
        masked_SR = torch.mul(mask, SR)    #mask x SR？
        masked_target = torch.mul(mask, target)    #mask x target
        mse = criterion(masked_SR, masked_target)    #MSE Loss 为什么可以在底下定义？
        if mse.data[0]==0:
            psnr[i]=0    #没psnr该位为0 这张图没有该class这张图不算个数
        else:
            nonzero_num = masked_SR.data[:].nonzero().size()[0]    #maskedSR的非0个数？
            mse.data[0] /= nonzero_num    #为什么要除？ 除mask有效点
            psnr[i] = 10 * log10(1 / mse.data[0])    #峰值信噪比计算公式？ 1表示最大

    mse = criterion(SR, target)/(SR.data[:].size()[1]*SR.data[:].size()[2]*SR.data[:].size()[3])
    psnr[class_num] = 10 * log10(1 / mse.data[0])    #前class个计算的带mask 为什么最后这个不带？  SR.data123代表什么？  算所有的平均值


##################看逻辑对不对？

    index = 0

    for key in classes.keys():
        classes[key][0] += psnr[index]    #classes[0]存入键值psnr
        if psnr[index] !=0:
            classes[key][1] +=1    #classes[1]存入键值是否有psnr 即检测到的该class个数
        index += 1                           #以及底下的那个key 表示什么？   dict.keys()返回一个字典所有键值


def test(testloader, model, deeplab, mid, criterion, gpuid, SR_dir):
    avg_psnr = 0
    interp = torch.nn.Upsample(size=(505, 505), mode='nearest')

    data_list = []
    for iteration, batch in enumerate(testloader, 1):
        input_ss, input, target, label_gt, size, name, label_hr = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), \
                                                     Variable(batch[2], volatile=True), batch[3], batch[4], batch[5], batch[6]    #前三个不更新weights

        #=======label transform h*w to 21*h*w=======#
        input = input.cuda(gpuid)
        target = target.cuda(gpuid)


        Blur_SR = model(input)    #model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
                                  #parser.add_argument("--model", default="model/SRResNet_VOC160_lr_finetune42.pth", type=str, help="model path")
                                  #Blur_SR 载入model生成的图像
        #output = model_pretrained(input)


        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)    #转为float32类型  试一下：.cpu() Vari转tensor要不要？ 要  data[:]的话就不要sequeeze了
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.    #遍历numpy
        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda(gpuid)    #生成SR图像 unsqueeze()升维

        result = transforms.ToPILImage()(SR.cpu().data[0])    #转为PIL(python image library)
        #path = join(SR_dir, '{0:04d}.jpg'.format(iteration))
        path = join(SR_dir, name[0] + '.png')
        result.save(path)

        ##########Per-class evaluation###############



        classwise_evaluate(SR, target, label_hr, NUM_CLASSES, classes)
        print("%s: %s.png" % (iteration, name[0]))


        #####################################
        #size = (target.size()[2], target.size()[3])
        #gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)
        #seg_out = torch.nn.Upsample(size, mode='bilinear')(seg)
        #seg_out = seg_out.cpu().data[0].numpy()
        #seg_out = seg_out.transpose(1, 2, 0)
        #seg_out = np.asarray(np.argmax(seg_out, axis=2), dtype=np.int)
        #data_list.append([gt.flatten(), seg_out.flatten()])
    #get_iou(data_list, NUM_CLASSES )

    #########################################################################show result
    for key in classes.keys():
        classes[key][0] /= classes[key][1]    #求每个class的平均psnr
    print("===> Avg. SR Per-Class PSNR: \n ")
    print(classes)

    for key in classes.keys():
        print('%s' % key)
        print('{:.4f} dB'.format(classes[key][0]))    #key是啥？
    #########################################################################show result

opt = parser.parse_args()
print(opt)


print("===> Loading datasets")
#root_dir = '/tmp4/hang_data/DIV2K'
root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/'
SR_dir = join(root_dir, 'SRResNet_x4')    #test数据集路径
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
deeplab_res = Res_Deeplab(num_classes=21)
saved_state_dict = torch.load('model/VOC12_scenes_20000.pth')
deeplab_res.load_state_dict(saved_state_dict)
deeplab_res = deeplab_res.eval()
mid = mid_layer()
criterion = torch.nn.MSELoss(size_average=False)



mid = mid.cuda(gpuid)
deeplab_res = deeplab_res.cuda(gpuid)
model = model.cuda(gpuid)
criterion = criterion.cuda(gpuid)

testloader = data.DataLoader(
    VOCDataValSet(root_dir, DATA_LIST_PATH, crop_size=(321, 321), mean=IMG_MEAN, scale=False, mirror=False),
    batch_size=1, shuffle=False, pin_memory=True)

test(testloader, model, deeplab_res, mid, criterion, gpuid, SR_dir)


