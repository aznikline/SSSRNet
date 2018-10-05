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
from srresnet_1 import Net
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/baseline_VOC160_finetune48.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def test(test_gen, model, criterion, gpuid, SR_dir):

    avg_psnr = 0
    #model.eval()

    for iteration, batch in enumerate(test_gen, 1):
        target, input = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)



        input = input.cuda(gpuid)
        target = target.cuda(gpuid)
        Blur_SR = model(input)

        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.
        SR = Variable((torch.from_numpy(im_h)).view(1,3,Blur_SR.cpu().data[0].shape[1],Blur_SR.cpu().data[0].shape[2])).cuda(gpuid)

        result = transforms.ToPILImage()(SR.cpu().data[0])
        path = join(SR_dir, '{0:04d}.jpg'.format(iteration+800))
        #result.save(path)
        mse = criterion(SR, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        print(iteration)
        print('===>psnr: {:.4f} dB'.format(psnr))

    print("===> Avg. SR PSNR: {:.4f} dB".format(avg_psnr / iteration))

opt = parser.parse_args()
print(opt)
gpuid = 3

print("===> Loading datasets")
#root_dir = '/tmp4/hang_data/DIV2K'
root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/'
#root_dir = '/tmp5/hang_data/SS_SR_data/'
#test_dir = 'DIV2K_validate_HR_x4'
test_dir = 'Val_HR_4x'
#test_dir = 'HR_4x'
targets = dataload.load_data(root_dir, test_dir)
#test_dir = 'DIV2K_validate_LR_x4'
test_dir = 'Val_LR_4x'
#test_dir = 'LR_4x'
inputs = dataload.load_data(root_dir, test_dir)

test_images = {"targets": targets ,"inputs": inputs}
SR_dir = join(root_dir, 'VOC_baseline')
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
#model = torch.load(opt.model)["model"]
model = model.cuda(gpuid)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.cuda(gpuid)
test_gen = dataload.batch_generator(test_images, 1, scalar_scale=4, isTest=True)

test(test_gen, model, criterion, gpuid, SR_dir)


