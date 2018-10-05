import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_semantic import Net, mid_layer, CrossEntropy_Probability
from dataset_hdf5 import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.utils import data
from deeplab.model import Res_Deeplab
import numpy as np
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20,
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
parser.add_argument("--semantic_loss", action="store_true", help="Use content loss?")

IMG_MEAN = [104.00698793,116.66876762,122.67891434]
NUM_CLASSES = 21

def main():
    print("SRResNet with Semantic_KL_loss training from scratch on VOC 160*160 patches.")    #改参数之前改一下
    global opt, model, netContent, deeplab_res, mid, semantic_criterion, semantic_kl_criterion, KL_DivLoss
    opt = parser.parse_args()
    print(opt)
    gpuid = 0
    cuda = True
    opt.vgg_loss = False
    opt.semantic_loss = True
    KL_DivLoss = True

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
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))    #建立vgg网络 从哪取loss照这个vgg loss写

        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    if opt.semantic_loss:
        print('===> Loading DeepLab model')
        deeplab_res = Res_Deeplab(num_classes=21)
        saved_state_dict = torch.load('model/VOC12_scenes_20000.pth')
        deeplab_res.load_state_dict(saved_state_dict)
        deeplab_res = deeplab_res.eval()
        semantic_criterion = CrossEntropy_Probability()
        semantic_kl_criterion = nn.KLDivLoss(size_average=False)

    print("===> Building model")
    model = Net()


    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda(gpuid)
        criterion = criterion.cuda(gpuid)
        if opt.vgg_loss:
            netContent = netContent.cuda(gpuid)
        if opt.semantic_loss:
            deeplab_res = deeplab_res.cuda(gpuid)
            semantic_criterion = semantic_criterion.cuda(gpuid)
            semantic_kl_criterion = semantic_kl_criterion.cuda(gpuid)

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
    root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/VOC_train_hrlabel160_HDF5'
    files_num = len(os.listdir(root_dir))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        #save_checkpoint(model, epoch)
        print("===> Loading datasets")
        x = random.sample(os.listdir(root_dir), files_num)
        for index in range(0, files_num):
            train_path = os.path.join(root_dir, x[index])
            print("===> Training datasets: '{}'".format(train_path))
            train_set = DatasetFromHdf5(train_path)    #看
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                            shuffle=True)
            avgloss = train(training_data_loader, optimizer, model, criterion, epoch, gpuid)
        if epoch % 2 ==0 :
            save_checkpoint(model, epoch)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, gpuid):
    avgloss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()


    for iteration, batch in enumerate(training_data_loader, 1):

        input, target, hr_label = Variable(batch[0]), \
                        Variable(batch[1], requires_grad=False), Variable(batch[3], requires_grad=False)

        input = input.cuda(gpuid)
        target = target.cuda(gpuid)
        output = model(input)    #model是SRNet的model

        loss = criterion(output, target) / opt.batchSize
        avgloss += loss.data[0]

        if opt.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target) / opt.batchSize * 0.01

        if opt.semantic_loss:
            output_ss_r = output[:, 0:1, ...] * 255.0 - IMG_MEAN[2]
            output_ss_g = output[:, 1:2, ...] * 255.0 - IMG_MEAN[1]
            output_ss_b = output[:, 2:3, ...] * 255.0 - IMG_MEAN[0]
            output_ss = torch.cat((output_ss_b, output_ss_g, output_ss_r), 1)
            output_ss = output_ss.cuda(gpuid)

            target_ss_r = target[:, 0:1, ...] * 255.0 - IMG_MEAN[2]
            target_ss_g = target[:, 1:2, ...] * 255.0 - IMG_MEAN[1]
            target_ss_b = target[:, 2:3, ...] * 255.0 - IMG_MEAN[0]
            target_ss = torch.cat((target_ss_b, target_ss_g, target_ss_r), 1)    #改成deeplab输入要求 rgb变bgr
            target_ss = target_ss.cuda(gpuid)

            sr_seg = deeplab_res(output_ss)    #nx21xhxw 是概率图  交叉熵输入要：nxhxwx21 单个像素属于21哪个概率最大 target:nxhxw 一维
                                               #-->nx1xhxw np.argmax() 方法1 但是argmax不能求导
            hr_seg = deeplab_res(target_ss)
            size = (target.size()[2], target.size()[3])
            hr_seg = hr_seg.detach()
            if KL_DivLoss:
                sr_seg = nn.Upsample(size, mode='bilinear')(sr_seg)    #deeplab会降低分辨率  实验验证要不要这两句
                hr_seg = nn.Upsample(size, mode='bilinear')(hr_seg)    #验证要不要
                sr_seg = nn.LogSoftmax()(sr_seg)    #要求log分布  softmax之后做成log分布
                hr_seg = nn.Softmax2d()(hr_seg)    #softmax
                semantic_loss = semantic_kl_criterion(sr_seg, hr_seg) / opt.batchSize * 0.1    #kl散度 描述两个概率分布相近程度 方法3  调系数
            else:
                semantic_loss = semantic_criterion(sr_seg, hr_seg, size) / opt.batchSize * 0.01    #semantic_criterion 针对crossentropy的概率求法
                                                                                                   #有空看能不能用


        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_variables=True)

        if opt.semantic_loss:
            deeplab_res.zero_grad()
            semantic_loss.backward(retain_variables=True)

        loss.backward()
        total_norm = 0
        if(loss.data[0] < 10000):
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        ##########show results###############
        is_show = False
        if is_show == True:
            label_show = label.cpu().data[0].numpy().transpose((1, 2, 0))
            label_show = np.asarray(np.argmax(label_show, axis=2), dtype=np.int)

            image_out = input_ss.cpu().data[0].numpy()
            image_out = image_out.transpose((1, 2, 0))
            image_out += IMG_MEAN
            image_out = image_out[:, :, ::-1]  # BRG2RGB
            image = np.asarray(image_out, np.uint8)
            #image = input.cpu().data[0].numpy().transpose((1, 2, 0))
            image_out = Blur_SR.cpu().data[0].numpy().transpose((1, 2, 0))

            label_heatmap = label.cpu().data[0].view(21, 1, input.data[0].size(1), input.data[0].size(2))
            label_heatmap = torchvision.utils.make_grid(label_heatmap)
            label_heatmap = label_heatmap.numpy().transpose((1, 2, 0))
            images_cls = input_cls.cpu().data[0].view(21, 3, input.data[0].size(1), input.data[0].size(2))
            images_cls = torchvision.utils.make_grid(images_cls)
            images_cls = images_cls.numpy().transpose((1, 2, 0))

            show_seg(image, label_show, image_out, label_heatmap, images_cls)
        #####################################

        if iteration % 100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Total_norm:{:.6f} Loss: {:.10f} Content_loss {:.10f}".format(epoch, iteration,
                                                                                         len(training_data_loader),
                                                                                         total_norm, loss.data[0],
                                                                                         content_loss.data[0]))
            elif opt.semantic_loss:
                print("===> Epoch[{}]({}/{}): Total_norm:{:.6f} Loss: {:.10f} Semantic_loss {:.10f}".format(epoch, iteration,
                                                                                         len(training_data_loader),
                                                                                         total_norm, loss.data[0],
                                                                                         semantic_loss.data[0]))
            else:
                print("===> Epoch[{}]({}/{}): Total_norm:{:.6f} Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                    total_norm,loss.data[0]))
    print("===> Epoch {} Complete: Avg. SR Loss: {:.6f}".format(epoch, avgloss / len(training_data_loader)))
    return (avgloss / len(training_data_loader))

def save_checkpoint(model, epoch):
    model_out_path = "model/" + "SRResNet_Semantic_KLLoss_scratch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
