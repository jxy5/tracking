import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SE(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            b, c= x.shape[0:2]
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c,1, 1)
            return y
            # Fscale操作：将得到的权重乘以原来的特征图x
            # return x * y.expand_as(x)

# 定义一个简单的网络，使用CBAM模块和SE模块，其中用SE模块替换CBAM的通道注意力
class RSCAttetion(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,dilation=1):
        super().__init__()
        if kernel_size % 2 == 0:
            assert("the kernel_size must be  odd.")
        self.kernel_size = kernel_size
        # 生成感受野特征，用分组卷积实现
        self.generate = nn.Sequential(nn.Conv2d(in_channel,in_channel * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=in_channel,bias =False),
                                      nn.BatchNorm2d(in_channel * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        # 生成感受野注意力，这是CBAM的空间注意力的一部分
        self. get_weight = nn.Sequential(nn.Conv2d(2,1,kernel_size=3,padding=1,bias=False),nn.Sigmoid())
        self.se =  SE(in_channel)  # 通道注意力

        self.conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride=kernel_size),nn.BatchNorm2d(out_channel),nn.ReLu())
        
    def forward(self,x):
        b,c = x.shape[0:2]
        channel_attention =  self.se(x)
        generate_feature = self.generate(x)  # 生成感受野特征

        h,w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)
        
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        # 生成感受野特征乘以通道注意力
        unfold_feature = generate_feature * channel_attention
        # 这是我的idea，但我的不一定对，我只是为了创新
        # max_feature,_ = torch.max(unfold_feature,dim=1,keepdim=True)
        # mean_feature = torch.mean(unfold_feature,dim=1,keepdim=True)

        # CBAM中的空间注意力，在感受野特征上提取空间注意力
        max_feature,_ = torch.max(generate_feature,dim=1,keepdim=True)
        mean_feature = torch.mean(generate_feature,dim=1,keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature,mean_feature),dim=1))
        conv_data = unfold_feature  * receptive_field_attention
        return self.conv(conv_data)


class SPP(nn.Module):
   def __init__(self, c1, c2, k=(5, 9, 13)):
       super().__init__()
       c_ = c1 // 2  # hidden channels
       self.cv1 = Conv(c1, c_, 1, 1)
       self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
       self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
   def forward(self, x):
       x = self.cv1(x)
       with warnings.catch_warnings():
           warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
           return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# YOLOv5中使用的SPPF层，更加简洁，减少计算量
class SPPF(nn.Module):
   def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
       super().__init__()
       c_ = c1 // 2  # hidden channels
       self.cv1 = Conv(c1, c_, 1, 1)
       self.cv2 = Conv(c_ * 4, c2, 1, 1)
       self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
   def forward(self, x):
       x = self.cv1(x)
       with warnings.catch_warnings():
           warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
           y1 = self.m(x)
           y2 = self.m(y1)
           return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# 这写的也不知道对不对，反正是这个思路。具体还的参考一下YOLOv5或者FPN
class FANet(nn.Module):
    def __init__(self):
        super(FANet, self).__init__()
        self.rsc1 = RSCAttetion(3, 32)
        self.rsc2 = RSCAttetion(3, 32)
        self.spp = SPPF(3, 32)
        self.rsc3 = RSCAttetion(3, 32)
    def forward(self, x):
        x1 = self.rsc1(x)
        x2 = self.rsc2(x)
        x3 = self.spp(x)
        x3 = x3 + x2
        x4 = self.rsc3(x3)  # x4 = m1
        x5 = x4 + x1  # x5 = m2
        return x4, x5
