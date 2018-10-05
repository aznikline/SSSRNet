import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_argmax import Net, mid_layer
from dataset_hdf5 import DatasetFromHdf5, DatasetFromHdf5_VOC
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.utils import data
from deeplab.model import Res_Deeplab
import numpy as np
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
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
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

def main():
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
    #deeplab
    deeplab_res = Res_Deeplab(num_classes=21)
    saved_state_dict = torch.load('model/VOC12_scenes_20000.pth')
    deeplab_res.load_state_dict(saved_state_dict)
    deeplab_res = deeplab_res.eval()
    #deeplab_res.eval()
    mid = mid_layer()
    #SRResNet
    print("===> Building model")
    model = Net()
    finetune = True

    if finetune == True:
        model_pretrained = torch.load('model/model_DIV2K_noBN_96_epoch_36.pth', map_location=lambda storage, loc: storage)["model"]
        index = 0
        for (src, dst) in zip(model_pretrained.parameters(), model.parameters()):
            if index != 0:
                list(model.parameters())[index].data = src.data
            index = index + 1

    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda(gpuid)
        mid = mid.cuda(gpuid)
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

    print("===> Training1")
    #root_dir = '/tmp4/hang_data/DIV2K/DIV2K_train_320_HDF5'
    root_dir = '/tmp4/hang_data/VOCdevkit/VOC2012/VOC_train1_HDF5'
    files_num = len(os.listdir(root_dir))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        #save_checkpoint(model, epoch)
        print("===> Loading datasets")
        x = random.sample(os.listdir(root_dir), files_num)
        for index in range(0, files_num):
            train_path = os.path.join(root_dir, x[index])
            print("===> Training datasets: '{}'".format(train_path))
            train_set = DatasetFromHdf5_VOC(train_path)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                            shuffle=True)
            avgloss = train(training_data_loader, optimizer, deeplab_res, model, mid, criterion, epoch, gpuid)
        if epoch % 2 ==0 :
            save_checkpoint(model, epoch)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, deeplab, model, mid, criterion, epoch, gpuid):
    avgloss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()
    input_size = (80, 80)


    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), \
                        Variable(batch[1], requires_grad=False)


        input_ss_r = input[:,0:1,...] * 255.0 -  IMG_MEAN[2]
        input_ss_g = input[:,1:2,...] * 255.0 -  IMG_MEAN[1]
        input_ss_b = input[:,2:3,...] * 255.0 -  IMG_MEAN[0]
        input_ss = torch.cat((input_ss_b, input_ss_g, input_ss_r), 1)

        input = input.cuda(gpuid)
        target = target.cuda(gpuid)
        input_ss = input_ss.cuda(gpuid)

        seg_out = deeplab(input_ss)
        size = (input_ss.size()[2], input_ss.size()[3])
        seg_out = mid(seg_out, size)
        seg_out= seg_out.detach()
        label = seg_out.cpu().data[0].numpy()
        label = np.asarray(np.argmax(label, axis=0), dtype=np.int)
        label = Variable(torch.from_numpy(label).float().view(1,1,size[0],size[1]))
        label = label.detach()

        output = model(input, label.cuda(gpuid))

        ##########show results###############
        is_show = False
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
            show_all(image, output0)

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

        if iteration % 10 == 0:
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
    model_out_path = "model/" + "model_epoch_argmax{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
