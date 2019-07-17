import PIL
import torch
import torch.nn as nn
import random
import datetime
#import adabound
import os
from torchvision import transforms, utils
from PIL import Image
from torch.nn import functional as F
from torchvision import models

"""
Baseline model settting: LeNet++ described in the Center Loss paper
"""

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes=28, feature_dim=64, use_bn=True):
        super(ConvNet, self).__init__()
        self.feature = False
        self.use_bn = use_bn
        self.feature_dim = feature_dim
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv1_2_bn = nn.BatchNorm2d(32)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv3_2_bn = nn.BatchNorm2d(128)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, feature_dim)
        self.prelu_fc1 = nn.PReLU()
        # self.fc2 = nn.Linear(2, num_classes, bias=False)
        self.fc2 = nn.Linear(feature_dim, num_classes, bias=False)
        # self.fc2 = Linear_Custem(feature_dim, num_classes)
        # self.fc2 = AngleLinear(feature_dim, num_classes, m=1.1, phiflag=False)

    def forward(self, x):
        if self.use_bn:
            x = self.prelu1_1(self.conv1_1_bn(self.conv1_1(x)))
        else:
            x = self.prelu1_1(self.conv1_1(x))
        if self.use_bn:
            x = self.prelu1_2(self.conv1_2_bn(self.conv1_2(x)))
        else:
            x = self.prelu1_2(self.conv1_2(x))

        x = F.max_pool2d(x, 2)

        if self.use_bn:
            x = self.prelu2_1(self.conv2_1_bn(self.conv2_1(x)))
        else:
            x = self.prelu2_1(self.conv2_1(x))
        if self.use_bn:
            x = self.prelu2_2(self.conv2_2_bn(self.conv2_2(x)))
        else:
            x = self.prelu2_2(self.conv2_2(x))

        x = F.max_pool2d(x, 2)

        if self.use_bn:
            x = self.prelu3_1(self.conv3_1_bn(self.conv3_1(x)))
        else:
            x = self.prelu3_1(self.conv3_1(x))
        if self.use_bn:
            x = self.prelu3_2(self.conv3_2_bn(self.conv3_2(x)))
        else:
            x = self.prelu3_2(self.conv3_2(x))

        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        x = self.prelu_fc1(self.fc1(x))
        # x = self.fc1(x)
        if self.feature:
            return x
        y = self.fc2(x)
        return x, y

"""
Improved model setting, VGG-DR, where DR means dimension reduction
"""
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', num_classes=28):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.reg = nn.Dropout(0.5)
        self.classifier = nn.Linear(64, num_classes)
        
        # feature extraction Trigger
        self.feature = None

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.reg(out)
        if self.feature:
            return out
        else:
            return out, self.classifier(out)
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # if turn on the batchnormalization, put the batchnorm between conv and relu
                # results show that if you turn off the batchnorm, training will be ten times harder and slower.
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           # use Parameter RELU instead of normal RELU
                           nn.PReLU()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        # dimension reduction
        layers += [nn.Conv2d(512, 64, kernel_size=1)]
        return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, num_classes=28):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18()
        self.dim_red = nn.Conv2d(512, 64, (1,1))
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x, feature=False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.dim_red(x)
        x = x.reshape(x.size(0), -1)
        if feature:
            return x
        return x, self.classifier(x)


def test(model, x):
    net = model()
    feature, y = net(x)
    print(feature.size(), y.size())

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    x1 = torch.randn(2, 3, 28, 28)
    test(ResNet18, x)
    test(VGG, x)
    test(ConvNet, x1)