from __future__ import absolute_import, division

import cv2
import numpy as np
import numbers
import torch

from . import ops


__all__ = ['SiamFCTransforms']


class Compose(object):
    """就是把一系列的transforms串起来"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomStretch(object):
    """随机的resize图片的大小"""
    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch
    
    def __call__(self, img):
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),
            round(img.shape[0] * scale))
        return cv2.resize(img, out_size, interpolation=interp)


class CenterCrop(object):
    """从img中间抠一块(size, size)大小的patch，如果不够大，以图片均值进行pad之后再crop"""
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.)
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad

        return img[i:i + th, j:j + tw]


class RandomCrop(object):
    """用法类似CenterCrop，只不过从随机的位置抠，没有pad的考虑"""
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


class ToTensor(object):
    """把np.ndarray转化成torch tensor类型"""
    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2, 0, 1))


class SiamFCTransforms(object):
    """对输入的ground_truth的z, x, bbox_z, bbox_x进行一系列变换，构成孪生网络的输入"""
    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz  # 模板输入尺寸127*127
        self.instance_sz = instance_sz  # 搜索输入尺寸255*255
        self.context = context

        self.transforms_z = Compose([  # 转换输入的模板图像
            RandomStretch(),  # 随机缩放
            CenterCrop(instance_sz - 8),  # 中心裁剪
            RandomCrop(instance_sz - 2 * 8),
            CenterCrop(exemplar_sz),  # 随机裁剪
            ToTensor()])
        self.transforms_x = Compose([  # 转换输入的搜索图像
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            ToTensor()])
    
    def __call__(self, z, x, box_z, box_x):
        # 对模板图像进行变换
        z = self._crop(z, box_z, self.instance_sz)
        z = self.transforms_z(z)

        # 对搜索图像进行变换
        x = self._crop(x, box_x, self.instance_sz)
        x = self.transforms_x(x)
        return z, x
    
    def _crop(self, img, box, out_size):
        """因为GOT-10k里面对于目标的bbox是以ltwh(即left, top, width, height)形式给出的，
        下面的代码一开始就先把输入的box变成center based，坐标形式变为[y, x, h, w]"""
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,  # y = top + height / 2
            box[0] - 1 + (box[2] - 1) / 2,  # x = left + width / 2
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz

        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = ops.crop_and_resize(  # ops.py 对原图进行裁剪和缩放，以目标为中心
            img, center, size, out_size,
            border_value=avg_color, interp=interp)
        
        return patch
