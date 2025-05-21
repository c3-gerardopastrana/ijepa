import torch
import torch.nn as nn
import torch.nn.functional as F
from lda import LDA, lda_loss, sina_loss, SphericalLDA
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_checkpoint=True):
        super(BasicBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        def block_fn(x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(identity)
            out = F.relu(out)
            return out

        if self.use_checkpoint:
            return checkpoint(block_fn, x, use_reentrant=False)
        else:
            return block_fn(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, lda_args=None, use_checkpoint=True, segments=2):
        super(ResNet, self).__init__()
        self.lda_args = lda_args
        self.use_checkpoint = use_checkpoint
        self.segments = segments
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        if self.lda_args:
            self.lda = LDA(num_classes, lda_args['lamb'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_checkpoint=self.use_checkpoint))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        def first_part_fn(x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x

        if self.use_checkpoint:
            x = checkpoint(first_part_fn, x, use_reentrant=False)
        else:
            x = first_part_fn(x)

        modules = [self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_checkpoint:
            x = checkpoint_sequential(modules, self.segments, x, use_reentrant=False)
        else:
            for layer in modules:
                x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, y=None, epoch=0):
        fea = self._forward_impl(x)

        if self.lda_args:
            fea = F.normalize(fea, p=2, dim=1)
            xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t = self.lda(fea, y)
            return xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t
        else:
            out = self.linear(fea)
            return out

