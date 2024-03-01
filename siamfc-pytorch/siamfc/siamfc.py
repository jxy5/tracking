from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):
    """在这个类里面就是进行数据增强，构造和加载，然后进行训练"""
    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),  # 用到的是v1版本的AlexNet
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)  # 初始化权重
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(  # 优化器SGD
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # 默认参数
        cfg = {
            # 基础参数
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # 参考参数
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # 训练参数
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()  # 装饰器，表示该函数不需要计算梯度。
    def init(self, img, box):
        """初始化函数，输入参数为模板图像img和ground_truth box"""

        self.net.eval()  # 将模型设置为评估模式，不进行训练。

        # 把输入的[l, t, w, h]格式的box转变为[y, x, h, w]格式的
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]  # 将目标框的中心坐标和宽度、高度分别保存到self.center和self.target_sz中

        # 汉宁窗(hanning window)，也叫余弦窗——增加惩罚
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  # 响应图上采样后的大小
        self.hann_window = np.outer(  # 生成汉宁窗函数
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()  # 汉宁窗函数归一化

        # search scale factors
        # 计算尺度因子，用于在不同尺度下进行搜索。
        # 论文中提到两个变体，一个是5个尺度的，一个是3个尺度的
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)  # 计算边界的语义信息
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))  # 计算模板图像的大小
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz         # 计算搜索图像的大小
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))  # 计算图像的平均颜色
        z = ops.crop_and_resize(           # 从原图像中裁剪出模板图像
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(       # 将模板图像转换为张量，并调整维度顺序
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)   # 将模板图像输入网络，得到模板特征
        # 这个kernel就是后面互相关的固定卷积核

    @torch.no_grad()
    def update(self, img):
        """更新函数，输入参数为当前帧图像img，对后续的帧更新出bbox来"""
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,  #这个center存疑，从哪里来的，是模板帧，还是上一帧，这对我很重要
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]  # 在不同尺度下，从当前帧图像中裁剪出搜索图像，resize成255*255
        x = np.stack(x, axis=0)  # 将三个尺度（搜索范围）的搜索图像堆叠成一个数组
        x = torch.from_numpy(x).to(  # 将搜索图像转换为张量，并调整维度顺序
            self.device).permute(0, 3, 1, 2).float()
        
        # 响应图
        x = self.net.backbone(x)  # 将搜索图像输入网络，得到特征图
        responses = self.net.head(self.kernel, x)  # 将模板特征与搜索特征进行相关运算heads.py，得到响应图
        responses = responses.squeeze(1).cpu().numpy()  # 将响应图从4D张量转换为3D张量3*17*17，转换为numpy数组

        # upsample responses and penalize scale changes
        # 将响应图上采样到指定大小 272*272
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        # 尺度惩罚。因为中间的尺度肯定是接近于1，其他两边的尺度不是缩一点就是放大一点，所以给以惩罚
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty  # 对响应图的前一半进行惩罚
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty  # 对响应图的后一半进行惩罚

        # 选出3个通道里面最大的那个
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location —— response上的峰值点，进行归一化和余弦窗惩罚
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)  # 峰值点

        # 定位目标的中心。计算位移
        # 我们原本都是以目标为中心的，认为最大峰值点应该在response的中心。
        # 因为之前在img上crop下一块instance patch，然后resize，然后送入CNN的backbone，然后score map又进行上采样成response
        # 所以要根据这过程，逆回去计算对应在img上的位移，
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2  # 峰值点和response中心的位移
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image  # 中心点加上峰值相对于原图img上的位移。修正center

        # 更新目标的尺寸
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]  # 线性插值
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # r根据ops.show_image输入的需要，又得把bbox格式改回[l, t, w, h]的格式
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)  # video sequence
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box  # first frame中的ground truth bbox
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)  # 传入第一帧的标签和图片，初始化一些参数，计算一些之后搜索区域的中心等等
            else:
                boxes[f, :] = self.update(img)  # 传入后续帧
                # 我懂了，完全懂了！这个update是更新后续帧，是循环进行的tracking，所以每次传入的center都是上一帧的center
            times[f] = time.time() - begin

            if visualize:  # 根据这些坐标来show，起到一个demo的效果
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        # print("batch_z shape:", z.shape)  # torch.Size([8, 3, 127, 127])
        # print("batch_x shape:", x.shape)  # torch.Size([8, 3, 239, 239])
        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)
            # print("responses shape:", responses.shape) # torch.Size([8, 1, 15, 15])
            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            # criterion使用的BalancedLoss，是调用F.binary_cross_entropy_with_logits，
            # 进行一个element - wise的交叉熵计算，所以创建出来的labels的shape其实就是和responses的shape是一样的
            if backward:
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        """数据准备"""
        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(  # datasets.py  Pair类
            seqs=seqs,
            transforms=transforms)
        # SiamFC的输入是pair<frame i , frame j>，其中同一视频的两帧间隔不超过T帧
        # 我认为输入应该是当前帧图像和上一帧图像预测的目标中心坐标，方便对当前帧crop，具体还要看tranforms()的实现
        # 这块还是不太清楚，先这样吧，做RPN去了

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        """开始训练"""
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):  # 50 epochs
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        """创建标签，以目标为中心，中心为1，非中心为0"""
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels  这里除以stride，是相对score map上来说
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
