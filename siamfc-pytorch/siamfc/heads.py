from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    # z和x互相关之后的值太大，经过sigmoid函数之后会使值处于梯度饱和的那块，梯度太小，乘以out_scale就是为了避免这个。
    
    def _fast_xcorr(self, z, x):
        # 互相关运算函数
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)  # 这个函数F.conv2d()是关键  # shape:[nx/nz, nz, H, W]
        out = out.view(nx, -1, out.size(-2), out.size(-1))  # [nx, 1, H, W]
        return out

if __name__ == '__main__':
    import torch
    z = torch.randn(8, 256, 6, 6)
    x = torch.randn(8, 256, 20, 20)
    siamfc = SiamFC()
    output = siamfc(z, x)
    print(output.shape)  # torch.Size([8, 1, 15, 15])