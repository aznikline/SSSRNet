import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


############without BN layers################

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn_mid = nn.BatchNorm2d(64)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.residual(out)
        out = self.conv_mid(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

class mid_layer(nn.Module):
    def __init__(self):
        super(mid_layer, self).__init__()

        #self.up = nn.Upsample((80, 80), mode='bilinear')
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()

    def forward(self, x, size):
        output = nn.Upsample(size, mode='bilinear')(x)
        output = self.softmax(output)
        return output

class CrossEntropy_Probability(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(CrossEntropy_Probability, self).__init__()
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, sr_label, hr_label, size=(160, 160), NUM_CLASSES=21):
        sr_label = nn.Upsample(size, mode='bilinear')(sr_label)
        sr_label = sr_label.transpose(1, 2).transpose(2, 3).contiguous().view(-1, NUM_CLASSES)
        hr_label = hr_label.transpose(1, 2).transpose(2, 3).contiguous().view(-1, NUM_CLASSES)
        loss = torch.sum(- hr_label * self.logsoftmax(sr_label))
        #loss = torch.sum(- hr_label * torch.log(sr_label))
        return loss

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=True)
        return loss

class KLLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(KLLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.kl_div = nn.KLDivLoss(size_average=True)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4

        n, c, h, w = predict.size()
        predict = self.logsoftmax(predict)

        loss = self.kl_div(predict, target)
        return loss

class KL_Loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, sr_label, hr_label, size=(160, 160), NUM_CLASSES=21):
        sr_label = nn.Upsample(size, mode='bilinear')(sr_label)
        sr_label = nn.LogSoftmax()(sr_label)
        loss = nn.KLDivLoss(size_average=True)(sr_label, hr_label)

        return loss