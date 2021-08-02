import math
import os

import torch
import torch.nn as nn
import torchvision


class Bottleneck(nn.Module):
    """Bottleneck unit"""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Modified ResNet, easy to custom"""

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # TODO: the stride of the last block has been changed
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # TODO: change to adaptive pool to size 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet50(torch.nn.Module):
    """ResNet 50 layers"""

    def __init__(self, num_classes=512):
        super(ResNet50, self).__init__()

        pthpath = '/home/jzzz/Proj/Medical/MedSeg/checkpoints/init/resnet50.pth'  # TODO: Change to path of the saved pth
        assert os.path.exists(pthpath), 'pls specify the pre-trained models'
        saved_state_dict = torch.load(pthpath)
        # TODO: Download from https://download.pytorch.org/models/resnet50-19c8e357.pth

        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        resnet50.load_state_dict(saved_state_dict)
        print('loading ResNet50 pre-trained weights w/ fc layer')

        num_ftrs = resnet50.fc.in_features
        resnet50.fc = torch.nn.Linear(num_ftrs, num_classes)

        self.resnet = resnet50

    def forward(self, x):

        x = self.resnet(x)

        return x


class Resnet34(torch.nn.Module):

    def __init__(self, num_classes=512):
        super(Resnet34, self).__init__()

        pthpath = '/Users/txy15/Dropbox/Summer18/RunComm/model'
        if os.path.exists(pthpath):
            # TODO: wrong in load pre-trained weights, only read the weights， not loading to network
            resnet34 = torch.load(pthpath)
        else:
            resnet34 = torchvision.models.resnet34(pretrained=True)
            torch.save(resnet34, pthpath)

        num_ftrs = resnet34.fc.in_features
        resnet34.fc = torch.nn.Linear(num_ftrs, num_classes)

        # ct = []
        # for name, child in resnet152.named_children():
        #     if "Conv2d_4a_3x3" in ct:
        #         for params in child.parameters():
        #             params.requires_grad = True
        #     ct.append(name)

        self.resnet = resnet34

    def forward(self, x):

        x = self.resnet(x)

        return x


if __name__ == '__main__':
    model = ResNet50(num_classes=512).cuda()
    inputs = torch.randn(1, 3, 224, 224).cuda()
    outputs = model(inputs)
    pass
