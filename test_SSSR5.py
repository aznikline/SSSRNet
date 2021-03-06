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
from SSSRNet5_baseline import Net, mid_layer
from deeplab.datasets import VOCDataValSet
from torch.utils import data
import torchvision
from visualization import show_seg
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/SSSRNet5_baseline2_concate_lr_epoch_50.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

IMG_MEAN = [104.00698793,116.66876762,122.67891434]
DATA_LIST_PATH = '/tmp4/hang_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
NUM_CLASSES = 21
gpuid = 3

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
    for iteration, batch in enumerate(testloader, 1):
        input_ss, input, target, label, size, name = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), \
                                                     Variable(batch[2], volatile=True), batch[3], batch[4], batch[5]

        #size = (1,1, input.size()[2], input.size()[3])
        #label = torch.nn.Upsample(size=(input.size()[2], input.size()[3]) , mode='nearest')(label)
        #=======label transform h*w to 21*h*w=======#
        #label_pro = np.expand_dims(Label_patch, axis=0)
        Label_patch = label.numpy()
        label_pro = np.repeat(Label_patch , 20, axis=1)
        for i in range(1,21):
            tmp = label_pro[:,i-1:i]
            if i == 0:
                tmp[tmp==255] = 0
            tmp[tmp != i] = -1
            tmp[tmp == i] = 1
            tmp[tmp == -1] = 0
        #Label_patch_test = Label_patch.copy()
        #Label_patch_test [Label_patch_test == 255] = 0
        #if (np.argmax(label_pro, axis=1).reshape((label.size())) - Label_patch_test).any() != 0:
            #print(">>>>>>Transform Error!")
        #=======label transform h*w to 21*h*w=======#
        label = Variable(torch.from_numpy(label_pro[:,:,:,:]).float())

        input = input.cuda(gpuid)
        target = target.cuda(gpuid)
        label = label.cuda(gpuid)

        #=========image mask generation=========#
        for i in range(20):
            mask = label[:,i:i+1,:,:].repeat(1,3,1,1)
            mask_selected = torch.mul(mask, input)
            if i == 0:
                input_cls = mask_selected
            else:
                input_cls = torch.cat((input_cls, mask_selected), dim=1)
        input_cls = input_cls.cuda(gpuid)
        Blur_SR = model(input_cls, input)
        #output = model_pretrained(input)


        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.
        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda(gpuid)

        result = transforms.ToPILImage()(SR.cpu().data[0])
        path = join(SR_dir, '{0:04d}.jpg'.format(iteration))
        #result.save(path)
        mse = criterion(SR, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        print("%s: %s.png" % (iteration, name[0]))
        print('===>psnr: {:.4f} dB'.format(psnr))


        ##########show results###############
        is_show = False
        if is_show == True:
            label_show = label_pro[0].transpose((1, 2, 0))
            label_show = np.asarray(np.argmax(label_show, axis=2), dtype=np.int)
            image = input.cpu().data[0].numpy().transpose((1, 2, 0))
            image_out = SR.cpu().data[0].numpy().transpose((1, 2, 0))
            label_heatmap = label.cpu().data[0].view(21, 1, input.data[0].size(1), input.data[0].size(2))
            label_heatmap = torchvision.utils.make_grid(label_heatmap)
            label_heatmap = label_heatmap.numpy().transpose((1, 2, 0))
            images_cls = input_cls.cpu().data[0].view(21, 3, input.data[0].size(1), input.data[0].size(2))
            images_cls = torchvision.utils.make_grid(images_cls)
            images_cls = images_cls.numpy().transpose((1, 2, 0))

            show_seg(image, label_show, image_out, label_heatmap, images_cls)
        #####################################
        #size = (target.size()[2], target.size()[3])
        #gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)
        #seg_out = torch.nn.Upsample(size, mode='bilinear')(seg)
        #seg_out = seg_out.cpu().data[0].numpy()
        #seg_out = seg_out.transpose(1, 2, 0)
        #seg_out = np.asarray(np.argmax(seg_out, axis=2), dtype=np.int)
        #data_list.append([gt.flatten(), seg_out.flatten()])
    #get_iou(data_list, NUM_CLASSES )
    print("===> Avg. SR PSNR: {:.4f} dB".format(avg_psnr / iteration))

opt = parser.parse_args()
print(opt)


print("===> Loading datasets")
#root_dir = '/tmp4/hang_data/DIV2K'
root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/'
SR_dir = join(root_dir, 'VOC_SSSR5')
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

model = Net()
model_pre = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
model.load_state_dict(model_pre.state_dict())


#model = torch.load(opt.model)["model"]

criterion = torch.nn.MSELoss(size_average=True)

model = model.cuda(gpuid)
criterion = criterion.cuda(gpuid)

testloader = data.DataLoader(
    VOCDataValSet(root_dir, DATA_LIST_PATH, crop_size=(321, 321), mean=IMG_MEAN, scale=False, mirror=False),
    batch_size=1, shuffle=False, pin_memory=True)

test(testloader, model, criterion, gpuid, SR_dir)


