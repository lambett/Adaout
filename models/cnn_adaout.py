import torch.nn as nn
from cnn_adaout import adaout, LinearScheduler
import torch.nn.init as init


class Block(nn.Module):
    def __init__(self, inplanes, planes):
        super(Block, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class CnnAdaout(nn.Module):
    def __init__(self, num_class=10, dist_prob=0.5, alpha=5, nr_steps=5e3):

        super(CnnAdaout, self).__init__()

        self.block1 = Block(3, 96)
        self.block2 = Block(96, 128)
        self.block3 = Block(128, 256)

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            LinearScheduler(adaout(dist_prob=dist_prob, alpha=alpha),
                            start_value=dist_prob, stop_value=dist_prob, nr_steps=nr_steps),
            nn.Linear(2048, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            LinearScheduler(adaout(dist_prob=dist_prob, alpha=alpha),
                            start_value=dist_prob, stop_value=dist_prob, nr_steps=nr_steps)
        )

        self._init_weight()

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        if self.training:
            modulelist = list(self.modules())
            num_module = len(modulelist)
            dp_loc = []
            fc_loc = []
            for idb in range(num_module):
                if isinstance(modulelist[idb], adaout):
                    dp_loc.append(idb)
                    for iFC in range(idb, num_module):
                        if isinstance(modulelist[iFC], nn.Linear):
                            fc_loc.append(iFC)
                            break
            dp_loc = dp_loc[:len(fc_loc)]
            assert len(dp_loc) == len(fc_loc)
            for imodu in range(len(dp_loc)):
                modulelist[dp_loc[imodu]].weight_behind = modulelist[fc_loc[imodu]].weight.data

            for module in self.modules():
                if isinstance(module, LinearScheduler):
                    module.step()

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.reshape((out.shape[0], -1))
        out = self.fc(out)
        out = self.out(out)
        return out