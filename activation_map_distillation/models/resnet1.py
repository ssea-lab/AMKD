import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 2

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_feature,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = num_feature[0]

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_feature[0], blocks_num[0])
        self.layer2 = self._make_layer(block, num_feature[1], blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, num_feature[2], blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, num_feature[3], blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(num_feature[3] * block.expansion, num_classes)

        self.num_features = num_feature[3] * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x):
        """ this forward_features function is used to get logits
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_multi_stage(self, x, is_feat=False):
        """ this forward_multi_stage function is used to get hierarchical representation for distillation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ls = []
        x = self.layer1(x)
        ls.append(x)
        x = self.layer2(x)
        ls.append(x)
        x = self.layer3(x)
        ls.append(x)
        x = self.layer4(x)
        ls.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ls.append(x)
        logits = self.fc(x)  # ls: [f0, f1, f2, f3, f4], logits
        if is_feat:
            return ls, logits
        else:
            return logits


def resnet_little(num_classes=5):
    """ return a ResNet 18 object
    """
    return ResNet(Bottleneck, [2, 2, 2, 2], [24, 48, 96, 192], num_classes=num_classes)


def resnet_lite(num_classes=5):
    """ return a ResNet 18 object
    """
    return ResNet(Bottleneck, [2, 2, 2, 2], [16, 32, 64, 128], num_classes=num_classes)


if __name__ == '__main__':
    model = resnet_lite(num_classes=5)

    # for name, param in model.named_parameters():
    #     print(name)
    for name, param in model.layer4.named_parameters():
        print(name)

    # a = torch.rand(2, 3, 224, 224)
    # out1 = model.forward_features(a)
    # print(out1.shape)
    # logits = model.forward_head(out1)
    # print(logits.shape)
    import torch
    from thop import profile
    from thop import clever_format

    # 假设你有一个模型叫做model
    # model = ...  # 这里是你的模型定义

    # 创建一个随机输入张量，其大小应该与模型的输入大小相匹配
    input_tensor = torch.randn(1, 3, 512, 1024)  # 例如，对于常见的图像模型，输入可能是[1, 3, 224, 224]

    # 使用thop的profile函数来计算FLOPs
    flops, params = profile(model, inputs=(input_tensor,))

    # 将FLOPs转换为更易读的格式
    flops, params = clever_format([flops, params], "%.2f")

    print(f"FLOPs: {flops}, Parameters: {params}")
