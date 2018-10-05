import argparse, os
import torch
import numpy as np
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet import Net
from srresnet import mid_layer
from deeplab.model import Res_Deeplab
from dataset_hdf5 import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo

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
parser.add_argument("--clip", type=float, default=100, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    gpuid = 1
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
    mid = mid_layer()
    #deeplab
    deeplab_res = Res_Deeplab(num_classes=21)
    saved_state_dict = torch.load('model/VOC12_scenes_20000.pth')
    deeplab_res.load_state_dict(saved_state_dict)
    #deeplab_res.eval()

    #model = torch.load('model/model_DIV2K_epoch_400.pth')['model']
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda(gpuid)
        mid_layer = mid_layer.cuda(gpuid)
        deeplab_res = deeplab_res.cuda(gpuid)
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

    print("===> Training")
    root_dir = '/tmp4/hang_data/DIV2K/DIV2K_train_320_HDF5'
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
            avgloss = train(training_data_loader, optimizer, model, mid_layer, deeplab_res, criterion, epoch, gpuid)
        if epoch % 2 ==0 :
            save_checkpoint(model, epoch)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, mid_layer, deeplab, criterion, epoch, gpuid):
    avgloss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        input_ss = input.data[:].numpy().astype(np.float32)
        input_ss = np.multiply(input_ss, 255)
        input_ss = np.transpose(input_ss,(0, 2, 3, 1))
        input_ss -= IMG_MEAN
        input_ss = np.transpose(input_ss, (0, 3, 1, 2))
        input_ss = Variable(torch.from_numpy(input_ss).float(), requires_grad=False)

        input = input.cuda(gpuid)
        input_ss = input_ss.cuda(gpuid)
        target = target.cuda(gpuid)

        seg_out = deeplab(input_ss)
        seg_out = mid_layer(seg_out)
        seg_out= seg_out.detach()

        output = model(input, seg_out)
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
        if(loss.data[0] < 500):
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        if iteration % 100 == 0:
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
    model_out_path = "model/" + "main_320_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
