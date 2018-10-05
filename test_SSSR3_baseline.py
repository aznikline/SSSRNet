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
from SSSRNet3 import Net, mid_layer
from deeplab.datasets import VOCDataValSet
from torch.utils import data
from srresnet_1 import Net
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/baseline_SSSR3_finetune46.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

IMG_MEAN = [104.00698793,116.66876762,122.67891434]
DATA_LIST_PATH = '/tmp4/hang_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
NUM_CLASSES = 21
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

    psnr = np.zeros(class_num+1)
    for i in range(class_num):
        mask = label[:, i:i + 1, :, :].repeat(1, 3, 1, 1)
        masked_SR = torch.mul(mask, SR)
        masked_target = torch.mul(mask, target)
        mse = criterion(masked_SR, masked_target)
        if mse.data[0]==0:
            psnr[i]=0
        else:
            nonzero_num = masked_SR.data[:].nonzero().size()[0]
            mse.data[0] /= nonzero_num
            psnr[i] = 10 * log10(1 / mse.data[0])

    mse = criterion(SR, target)/(SR.data[:].size()[1]*SR.data[:].size()[2]*SR.data[:].size()[3])
    psnr[class_num] = 10 * log10(1 / mse.data[0])
    print('===>psnr: {:.4f} dB'.format(psnr[class_num]))
    index = 0
    for key in classes.keys():
        classes[key][0] += psnr[index]
        if psnr[index] !=0:
            classes[key][1] +=1
        index += 1

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


def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('input')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()


def test(testloader, model,  criterion, gpuid, SR_dir):
    avg_psnr = 0
    interp = torch.nn.Upsample(size=(505, 505), mode='nearest')

    data_list = []
    for iteration, batch in enumerate(testloader):
        input_ss, input, target, label, size, name, label_hr = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), \
                                                     Variable(batch[2], volatile=True), batch[3], batch[4], batch[5], batch[6]

        input = input.repeat(1, 21, 1, 1)
        input = input.cuda(gpuid)
        target = target.cuda(gpuid)

        Blur_SR = model(input)
        #output = model_pretrained(input)


        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.
        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda(gpuid)

        print("%s: %s.png" % (iteration, name[0]))
        classwise_evaluate(SR, target, label_hr, NUM_CLASSES, classes)

        result = transforms.ToPILImage()(SR.cpu().data[0])
        path = join(SR_dir, '{0:04d}.jpg'.format(iteration))
        #result.save(path)

    for key in classes.keys():
        classes[key][0] /= classes[key][1]
    print("===> Avg. SR Per-Class PSNR: \n ")

    for key in classes.keys():
        print('%s' % key)
        print('{:.4f} dB'.format(classes[key][0]))

opt = parser.parse_args()
print(opt)
gpuid = 1

print("===> Loading datasets")
#root_dir = '/tmp4/hang_data/DIV2K'
root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/'
SR_dir = join(root_dir, 'VOC_SSSR3_baseline')
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]


#model = torch.load(opt.model)["model"]

criterion = torch.nn.MSELoss(size_average=False)

model = model.cuda(gpuid)
criterion = criterion.cuda(gpuid)

testloader = data.DataLoader(
    VOCDataValSet(root_dir, DATA_LIST_PATH, crop_size=(321, 321), mean=IMG_MEAN, scale=False, mirror=False),
    batch_size=1, shuffle=False, pin_memory=True)

test(testloader, model, criterion, gpuid, SR_dir)


