import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SSSRNet6_skip import Net
from dataset_hdf5 import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.utils import data
from deeplab.model import Res_Deeplab
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from visualization import show_seg, show_features, get_n_params

##################################
# Author : Hang
#
# Date: Jan, 8, 2018
#
# Version: 1.0
#
# Description: try multiple depthwise convolution layers to extract features by semantic labels
#
# Details: add show_features() function
##################################


# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=10,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=10000, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")

IMG_MEAN = [104.00698793,116.66876762,122.67891434]
#IMG_MEAN = np.array((128.0,128.0,128.0), dtype=np.float32)


def main():
    print("SSSRNet3 training from scratch on VOC 160*160 patches.")
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    gpuid = 0
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True


    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))

        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    model = Net()
    print('Parameters: {}'.format(get_n_params(model)))
    model_pretrained = torch.load('model/model_DIV2K_noBN_96_epoch_36.pth', map_location=lambda storage, loc: storage)["model"]
    finetune = False

    if finetune == True:

        index = 0
        for (src, dst) in zip(model_pretrained.parameters(), model.parameters()):
            if index >1:
                list(model.parameters())[index].data = src.data
            index = index + 1

    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda(gpuid)
        model_pretrained = model_pretrained.cuda(gpuid)
        criterion = criterion.cuda(gpuid)
        if opt.vgg_loss:
            netContent = netContent.cuda(gpuid)

            # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training1")
    #root_dir = '/tmp4/hang_data/DIV2K/DIV2K_train_320_HDF5'
    root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/VOC_train_label160_HDF5'
    files_num = len(os.listdir(root_dir))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        #save_checkpoint(model, epoch)
        print("===> Loading datasets")
        x = random.sample(os.listdir(root_dir), files_num)
        for index in range(0, files_num):
            train_path = os.path.join(root_dir, x[index])
            print("===> Training datasets: '{}'".format(train_path))
            train_set = DatasetFromHdf5(train_path)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                            shuffle=True)
            avgloss = train(training_data_loader, optimizer, model, model_pretrained, criterion, epoch, gpuid)
        if epoch % 2 ==0 :
            save_checkpoint(model, epoch)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, model_pretrained, criterion, epoch, gpuid):
    avgloss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()
    input_size = (80, 80)


    for iteration, batch in enumerate(training_data_loader, 1):

        input, target, label = Variable(batch[0]), \
                        Variable(batch[1], requires_grad=False), batch[2]


        #=======label transform h*w to 21*h*w=======#
        #label_pro = np.expand_dims(Label_patch, axis=0)
        Label_patch = label.numpy()
        label_pro = np.repeat(Label_patch, 21, axis=1)
        for i in range(21):
            tmp = label_pro[:,i:i+1]
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

        #input_ss_r = input[:,0:1,...] * 255.0 -  IMG_MEAN[2]
        #input_ss_g = input[:,1:2,...] * 255.0 -  IMG_MEAN[1]
        #input_ss_b = input[:,2:3,...] * 255.0 -  IMG_MEAN[0]
        #input_ss = torch.cat((input_ss_b, input_ss_g, input_ss_r), 1)

        input = input.cuda(gpuid)
        target = target.cuda(gpuid)
        label = label.cuda(gpuid)

        for i in range(21):
            mask_in = label[:,i:i+1,:,:].repeat(1,3,1,1)
            mask_selected = torch.mul(mask_in, input)
            if i == 0:
                input_cls = mask_selected
            else:
                input_cls = torch.cat((input_cls, mask_selected), dim=1)

        for i in range(21):
            if i ==0:
                mask = label[:,i:i+1,:,:].repeat(1,64,1,1)
            else:
                mask = torch.cat((mask,label[:,i:i+1,:,:].repeat(1,64,1,1)),1)

        output = model(input_cls, mask)
        #output = model_pretrained(input)

        ##########show results###############
        is_show = False
        if is_show == True:
            label_show = label_pro[0].transpose((1, 2, 0))
            label_show = np.asarray(np.argmax(label_show, axis=2), dtype=np.int)
            image = input.cpu().data[0].numpy().transpose((1, 2, 0))
            image_out = output.cpu().data[0].numpy().transpose((1, 2, 0))
            label_heatmap = label.cpu().data[0].view(21, 1, input.data[0].size(1), input.data[0].size(2))
            label_heatmap = torchvision.utils.make_grid(label_heatmap)
            label_heatmap = label_heatmap.numpy().transpose((1, 2, 0))
            #images_cls = input_cls.cpu().data[0].view(21, 3, input.data[0].size(1), input.data[0].size(2))
            #images_cls = torchvision.utils.make_grid(images_cls)
            #images_cls = images_cls.numpy().transpose((1, 2, 0))

            show_seg(image, label_show, image_out, label_heatmap, image)
        #####################################


        loss = criterion(output, target) / opt.batchSize
        avgloss += loss.data[0]

        if opt.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_variables=True)

        loss.backward()
        total_norm = 0
        if(loss.data[0] < 10000):
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        if iteration % 50 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Total_norm:{:.6f} Loss: {:.10f} Content_loss {:.10f}".format(epoch, iteration,
                                                                                         len(training_data_loader),
                                                                                         total_norm, loss.data[0],
                                                                                         content_loss.data[0]))
            else:
                print("===> Epoch[{}]({}/{}): Total_norm:{:.6f} Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                    total_norm,loss.data[0]))
    print("===> Epoch {} Complete: Avg. SR Loss: {:.6f}".format(epoch, avgloss / len(training_data_loader)))
    return (avgloss / len(training_data_loader))

def save_checkpoint(model, epoch):
    model_out_path = "model/" + "SSSRNet6_skip160_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
