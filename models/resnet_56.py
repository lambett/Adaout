import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaout import adaout, LinearScheduler

sys.path.append("..")

def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(in_plane, out_plane,
                     kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


class BasicBlock(nn.Module):

    def __init__(self, in_plane, out_plane, stride=1, downsample=None,
                 dist_prob=0.05, block_size=6, alpha=30, nr_steps=5e3, keep_prob_1=0, keep_prob_2=0, keep_prob_3=0, keep_prob_4=0, start_mask=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_plane, out_plane)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return out


class ResNet56(nn.Module):
    def __init__(self, depth, wide_factor=1, num_classes=10):
        super(ResNet56, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(BasicBlock, 16 * wide_factor, n)
        self.layer2 = self._make_layer(BasicBlock, 32 * wide_factor, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64 * wide_factor, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = linear(64 * wide_factor, num_classes)

        self._init_weight()

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_plane, n_blocks, stride=1, dist_prob=0.05, block_size=6, alpha=30, nr_steps=5e3):

        downsample = None
        if stride != 1 or self.in_plane != out_plane:
            downsample = conv1x1(self.in_plane, out_plane, stride=stride)

        layers = []
        layers.append(block(self.in_plane, out_plane, stride, downsample,
                            dist_prob=dist_prob, block_size=block_size, alpha=alpha, nr_steps=nr_steps))
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(block(self.in_plane, out_plane, dist_prob=dist_prob, block_size=block_size, alpha=alpha, nr_steps=nr_steps))
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.training:
            modulelist = list(self.modules())
            num_module = len(modulelist)
            dploc = []
            convloc = []
            for idb in range(num_module):
                if isinstance(modulelist[idb], adaout):
                    dploc.append(idb)
                    for iconv in range(idb, num_module):
                        if isinstance(modulelist[iconv], nn.Conv2d):
                            convloc.append(iconv)
                            break
            dploc = dploc[:len(convloc)]
            assert len(dploc) == len(convloc)
            for imodu in range(len(dploc)):
                modulelist[dploc[imodu]].weight_behind = modulelist[convloc[imodu]].weight.data

            for module in self.modules():
                if isinstance(module, LinearScheduler):
                    module.step()

        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Calculate the covariance
        covout = out.clone().detach()
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out, covout