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
from srresnet import Net, mid_layer
from srresnet_1 import Net
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_scratch26.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
IMG_MEAN = [104.00698793,116.66876762,122.67891434]

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


def test(test_gen, model, deeplab, mid, criterion, gpuid, SR_dir):
    avg_psnr = 0
    interp = torch.nn.Upsample(size=(505, 505), mode='bilinear')


    for iteration, batch in enumerate(test_gen, 1):
        target, input = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)

        input_ss_r = input[:,0:1,...] * 255.0 -  IMG_MEAN[2]
        input_ss_g = input[:,1:2,...] * 255.0 -  IMG_MEAN[1]
        input_ss_b = input[:,2:3,...] * 255.0 -  IMG_MEAN[0]
        input_ss = torch.cat((input_ss_b, input_ss_g, input_ss_r), 1)

        input = input.cuda(gpuid)
        input_ss = input_ss.cuda(gpuid)
        target = target.cuda(gpuid)

        seg_out = deeplab(input_ss)

        size = (input_ss.size()[2], input_ss.size()[3])
        seg_out = mid(seg_out, size)
        seg_out= seg_out.detach()
        Blur_SR = model(input, seg_out)

        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.
        SR = Variable((torch.from_numpy(im_h)).view(1,3,Blur_SR.cpu().data[0].shape[1],Blur_SR.cpu().data[0].shape[2])).cuda(gpuid)

        result = transforms.ToPILImage()(SR.cpu().data[0])
        path = join(SR_dir, '{0:04d}.jpg'.format(iteration+800))
        result.save(path)
        mse = criterion(SR, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        print(iteration)
        print('===>psnr: {:.4f} dB'.format(psnr))


        ##########show results###############
        is_show = True
        if is_show == True:
            output0 = seg_out.cpu().data[0].numpy()
            output0 = output0.transpose(1, 2, 0)
            output0 = np.asarray(np.argmax(output0, axis=2), dtype=np.int)

            image_out = input_ss.cpu().data[0].numpy()
            image_out = image_out.transpose((1, 2, 0))
            image_out += IMG_MEAN
            image_out = image_out[:, :, ::-1]  # BRG2RGB
            image_out = np.asarray(image_out, np.uint8)


            image = input.cpu().data[0].numpy().transpose((1, 2, 0))
            show_all(image_out, output0)

        #####################################

    print("===> Avg. SR PSNR: {:.4f} dB".format(avg_psnr / iteration))

opt = parser.parse_args()
print(opt)
gpuid = 2

print("===> Loading datasets")
#root_dir = '/tmp4/hang_data/DIV2K'
root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/'
#root_dir = '/tmp5/hang_data/SS_SR_data/'
#test_dir = 'DIV2K_validate_HR_x4'
#test_dir = 'VOC_Val_HR_x4'
test_dir = 'HR_4x'
targets = dataload.load_data(root_dir, test_dir)
#test_dir = 'DIV2K_validate_LR_x4'
#test_dir = 'VOC_Val_LR_x4'
test_dir = 'LR_4x'
inputs = dataload.load_data(root_dir, test_dir)

test_images = {"targets": targets ,"inputs": inputs}
SR_dir = join(root_dir, 'VOC_SSSR')
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

#model = torch.load(opt.model)["model"]
model = model.cuda(gpuid)
mid = mid.cuda()
deeplab_res = deeplab_res.cuda(gpuid)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.cuda(gpuid)
test_gen = dataload.batch_generator(test_images, 1, scalar_scale=4, isTest=True)

test(test_gen, model, deeplab_res, mid, criterion, gpuid, SR_dir)


