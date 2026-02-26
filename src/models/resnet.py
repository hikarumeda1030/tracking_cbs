import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    """
    dataset_mode: 'cifar' or 'imagenet'
      - 'cifar'    : stem = 3x3 s1, no maxpool（32x32 向け）
      - 'imagenet' : stem = 7x7 s2 + maxpool s2（224x224 向け）
    """
    def __init__(self, block, num_block, num_classes=100, dataset_mode: str = 'cifar'):
        super().__init__()
        assert dataset_mode in ['cifar', 'imagenet']
        self.in_channels = 64

        # --- stem ---
        if dataset_mode == 'cifar':
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def resnet18(num_classes=100, dataset_mode='cifar'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dataset_mode=dataset_mode)


def resnet34(num_classes=100, dataset_mode='cifar'):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, dataset_mode=dataset_mode)


def resnet50(num_classes=100, dataset_mode='cifar'):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, dataset_mode=dataset_mode)


def resnet101(num_classes=100, dataset_mode='cifar'):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, dataset_mode=dataset_mode)


def resnet152(num_classes=100, dataset_mode='cifar'):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, dataset_mode=dataset_mode)
