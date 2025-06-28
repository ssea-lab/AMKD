import os

import torch
from torchvision.models.resnet import ResNet, BasicBlock

"""
这个脚本主要用来定义向Resnet模型中注入两个自定义的函数
"""


# hack: inject the forward_multi_stage function of `timm.models.convnext.ConvNeXt`
def forward_features(self: ResNet, x):
    """ this forward_features function is used to get hierarchical representation
    """
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)  # [B, C, H, W]
    return x


def forward_head(self: ResNet, x):
    """ this forward_features function is used to get logits
    """
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x


ResNet.forward_features = forward_features
ResNet.forward_head = forward_head


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        weight_path = os.path.join('model_weight', 'resnet18.pth')
        print('loading weight from {}'.format(weight_path))
        pretrained_dict = torch.load(weight_path)
        if 'fc.weight' in pretrained_dict:
            pretrained_dict.pop('fc.weight')
        if 'fc.bias' in pretrained_dict:
            pretrained_dict.pop('fc.bias')
        state = model.load_state_dict(pretrained_dict, strict=False)
        print(state)
    return model


if __name__ == '__main__':
    model = resnet18(num_classes=5)

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

