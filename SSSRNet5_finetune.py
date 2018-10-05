import torch
import torch.nn as nn
import math
import torch.nn.init as init
from visualization import show_features


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

class _Clswise_Residual_Block(nn.Module):
    def __init__(self):
        super(_Clswise_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=32*20, out_channels=32*20, kernel_size=3, stride=1, padding=1, bias=True, groups=20)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32*20, out_channels=32*20, kernel_size=3, stride=1, padding=1, bias=True, groups=20)
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
        self.conv_clswise_mid = nn.Conv2d(in_channels=20*32, out_channels=20*32, kernel_size=3, stride=1, padding=1, bias=True, groups=20)
        self.clswise_residual = self.make_layer(_Clswise_Residual_Block, 4)
        self.conv_clswise = nn.Conv2d(in_channels=20*3, out_channels=20*32, kernel_size=5, stride=1, padding=2, bias=True, groups=20)
        self.conv_pointwise = nn.Conv2d(in_channels=20 * 32, out_channels=64, kernel_size=1, stride=1, padding=0,
                                         bias=True)
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

    def forward(self, x, lr):
        #show_features(x, 0, 3)

        out = self.conv_clswise(x)
        residual = out
        out = self.clswise_residual(out)
        out = self.conv_clswise_mid(out)
        out = torch.add(out, residual)
        #show_features(out, 0, 1)
        out = self.conv_pointwise(out)

        out_lr = self.conv_input(lr)
        residual = out_lr
        out_lr = self.residual(out_lr)
        out_lr = self.conv_mid(out_lr)
        out_lr = torch.add(out_lr, residual)

        out = torch.add(out_lr, out)
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