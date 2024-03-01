from __future__ import absolute_import

import torch.nn as nn


__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']
"""为了让别的module导入这个backbones.py的东西时，只能导入__all__后面的部分"""


class _BatchNorm2d(nn.BatchNorm2d):
    """nn.BatchNorm2d:防止梯度爆炸和梯度消失，加快网络学习，输入参数为上一层通道数"""
    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):
    """三个版本的Alexnet共用同样的forward函数
    都集成了类_AlexNet，所以他们都是使用同样的forward函数，
    依次通过五个卷积层，每个卷积层使用nn.Sequential()堆叠，
    只是他们各自的total_stride和具体每层卷积层实现稍有不同"""
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(  # 以模板图像为例，进行维度推导。
            nn.Conv2d(3, 96, 11, 2),  # [1,3,127,127]->[1,96,59,59]
            # nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding(默认为1和0))
            # 第1维指的是batch_size，比如训练过程中batch_size为(8, 16, 32等等)，代码中假设为1
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),  # nn.ReLu:对每一层的输出激活处理，增加网络的非线性学习能力
            nn.MaxPool2d(3, 2))  # [1,96,59,59]->[1,96,29,29]
            # nn.Maxpool2d:池化下采样，代码中一共有两次下采样(kernel_size=3,stride=2)，每次缩小1/2特征图大小
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),  #[1,96,29,29]->[1,256,25,25]
            # groups=2的分组卷积，减少参数
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))  #[1,256,25,25]->[1,256,12,12]
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),  #[1,256,12,12]->[1,384,10,10]
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),  #[1,384,10,10]->[1,384,8,8]
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))  #[1,384,8,8]->[1,256,6,6]，即最终输入互相关层的feature map大小


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))

if __name__ == '__main__':
    """测试一下算的维度对不对"""
    alexnetv1 = AlexNetV1()
    import torch
    z = torch.randn(1, 3, 127, 127)
    output = alexnetv1(z)
    print(output.shape)  # torch.Size([1, 256, 6, 6])
    x = torch.randn(1, 3, 256, 256)
    output = alexnetv1(x)
    print(output.shape)  # torch.Size([1, 256, 22, 22])
    # 换成AlexNetV2依次是：
    # torch.Size([1, 32, 17, 17])、torch.Size([1, 32, 49, 49])
    # 换成AlexNetV3依次是：
    # torch.Size([1, 512, 6, 6])、torch.Size([1, 512, 22, 22])