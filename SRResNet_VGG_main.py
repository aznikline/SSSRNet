import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_1 import Net
from dataset_hdf5 import DatasetFromHdf5_SR
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


def main():
    print("SRResNet with 0.001 VGG-loss training from scratch on VOC 160*160 patches.")
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    gpuid = 0
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)    #为CPU设置随机种子
    if cuda:
        torch.cuda.manual_seed(opt.seed)    #为当前GPU设置随机种子

    cudnn.benchmark = True    #有用？
    opt.vgg_loss = True


    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()    #netVGG是什么？   建立vgg网络
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))

        class _content_model(nn.Module):    #nn.Module是什么？  需要继承这个类
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()    #netContent就是VGG19的内容吗？

    print("===> Building model")
    model = Net()    #srresnet / srresnet_1 用的哪个？ 从import看


    criterion = nn.MSELoss(size_average=False)    #SR的Loss target:nxcxhxw false就是没除

    print("===> Setting GPU")
    if cuda:
        model = model.cuda(gpuid)
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
    files_num = len(os.listdir(root_dir))    #os.listdir(path)返回path中的文件&文件夹名字的列表
                                             #files_num为path中的文件&文件夹数目
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        #save_checkpoint(model, epoch)
        print("===> Loading datasets")

        x = random.sample(os.listdir(root_dir), files_num)    #对root_dir文件夹下的list随机重排

        for index in range(0, files_num):
            train_path = os.path.join(root_dir, x[index])    #拼接路径 train_path遍历root_dir下h5文件
            print("===> Training datasets: '{}'".format(train_path))
            train_set = DatasetFromHdf5_SR(train_path)    #看懂###################################################################
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                            shuffle=True)    #变tensor
            avgloss = train(training_data_loader, optimizer, model, criterion, epoch, gpuid)
        if epoch % 2 ==0 :
            save_checkpoint(model, epoch)    #save_checkpoint(model, epoch)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))    #lr更新  为什么要这样更新？
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, gpuid):
    avgloss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()    #模型状态设置为训练状态，主要针对dropout层


    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), \
                        Variable(batch[1], requires_grad=False)

        input = input.cuda()
        target = target.cuda()

        output = model(input)    #对input做前向过程 得到输出

        loss = criterion(output, target) / opt.batchSize    #计算output和target之间的损失  为什么要除以opt.batchsize？ 除以平均到每张图片的MSE

        avgloss += loss.data[0]

        if opt.vgg_loss:
            content_input = netContent(output)    #？？？？？？？？？？？？？？？？output是SRnet的输出
            content_target = netContent(target)    #？？？？？？？？？？？？？？？？
            content_target = content_target.detach()    #？？？？？？？？？？？？？？？？不回传
            content_loss = criterion(content_input, content_target) / opt.batchSize * 0.001    #？？？？？？？？？？？？？？？？？减小尺度

        optimizer.zero_grad()    #梯度置零

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_variables=True)

        loss.backward()    #反向过程 计算损失关于各参数的梯度

        total_norm = 0
        if(loss.data[0] < 10000):
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)    #防止梯度爆炸

        optimizer.step()    #利用计算得到的梯度对参数进行更新

        if iteration % 200 == 0:
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
    model_out_path = "model/" + "SRResNet_VGG_0.001_scratch{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
