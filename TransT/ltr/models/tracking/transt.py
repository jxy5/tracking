import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network


class TransT(nn.Module):
    """ 做单目标检测的TransT模型 """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ 初始化模型.
        参数:
            1. backbone: transt_backbone.py
            2. featurefusion_network: transformer的变体——featurefusion_network.py
            3. num_classes: 目标跟踪的类别，对于单目标跟踪是 1
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        # 分类和回归网络
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, search, template):
        """前向传播函数，输入为 NestedTensor，包括：
               - search.tensors: 经过批量处理的搜索图像，其形状为 [batch_size x 3 x H_search x W_search]
               - search.mask: 二进制搜索图像掩模，形状为 [batch_size x H_search x W_search]，在填充像素上为 1
               - template.tensors: 经过批量处理的模板图像，其形状为 [batch_size x 3 x H_template x W_template]
               - template.mask: 二进制模板图像掩模，形状为 [batch_size x H_template x W_template]，在填充像素上为 1

            返回值为字典，包括以下元素:
               - "pred_logits": 所有特征向量的分类 logit。
                            形状为 [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": 所有特征向量的标准化框坐标，表示为 (center_x, center_y, height, width)。这些值在 [0, 1] 范围内标准化，
                           相对于每个单独图像的大小。
        """
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_search, pos_search = self.backbone(search)  # 特征提取
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search= feature_search[-1].decompose()  # 分解 NestedTensor
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])
        # 使用MLP作为预测头
        outputs_class = self.class_embed(hs)  # 分类网络
        outputs_coord = self.bbox_embed(hs).sigmoid()  # 回归网络
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def track(self, search):
        """ 目标跟踪函数，输入为 NestedTensor，包括：
               - search.tensors: 经过批量处理的图像，其形状为 [batch_size x 3 x H_search x W_search]
               - search.mask: 二进制掩模，形状为 [batch_size x H_search x W_search]，在填充像素上为 1

            返回值为字典，包括以下元素:
               - "pred_logits": 所有特征向量的分类 logit。
                                形状为 [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": 所有特征向量的标准化框坐标，表示为 (center_x, center_y, height, width)。这些值在 [0, 1] 范围内标准化，
                               相对于每个单独图像的大小。

        """
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        """ 模板函数，输入为 NestedTensor，包括：
                  - z.tensors: 经过批量处理的图像，其形状为 [batch_size x 3 x H x W]
                  - z.mask: 二进制掩模，形状为 [batch_size x H x W]，在填充像素上为 1

               无返回值，但是将特征和位置信息存储到类的成员变量中。
           """
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class SetCriterion(nn.Module):
    """ 计算TransT模型的损失。
    计算结果分两步:
        1) 计算真实边界框和模型输出之间的匹配关系
        2) 监督每个匹配的真实边界框和预测结果对（监督类别和边界框）。
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ 创建损失函数
        参数：
            num_classes：目标分类数，单目标跟踪时始终为1。
            matcher：能够计算目标和预测框之间匹配关系的模块。
            weight_dict：包含损失名称作为键和它们相对权重作为值的字典。
            eos_coef：应用于无目标类别的相对分类权重。
            losses：应用的所有损失的列表。请参见get_loss以获取可用损失列表。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """ 分类损失函数（NLL）
        targets dicts必须包含键“labels”，包含维度为[nb_target_boxes]的张量。
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ 计算与边界框相关的损失，包括L1回归损失和GIoU损失
                   targets dicts必须包含键“boxes”，包含维度为[nb_target_boxes，4]的张量，
                   目标框应该是格式为（center_x，center_y，h，w）的，通过图像大小进行归一化。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # 按照indices对预测进行排列
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 按照indices对目标进行排列
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """这个函数计算损失。
        参数:
        outputs: tensor 字典，格式请见模型的输出说明。
        targets: 字典列表，每个字典对应一个样本的目标，len(targets)==batch_size。
        每个字典中的键值取决于所应用的损失函数，详见每个损失函数的说明。
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # 去除辅助输出结果，只保留主输出结果
        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)
        # 计算匹配结果，参考损失函数的输出说明了解indices的格式
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)  # 统计所有节点的正样本数

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()  # 最少为1

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:  # 计算每个损失函数的损失值
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

class MLP(nn.Module):
    """定义一个多层感知机MLP（也称为前馈神经网络FFN）"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers  # 网络层数
        h = [hidden_dim] * (num_layers - 1)  # 中间层维度，使用hidden_dim的值构建num_layers - 1次维度
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    """
    使用PyTorch的ModuleList（包含一组模型）模块，对模型层进行逐一初始化
    输入维度：input_dim，中间层维度h，输出维度：output_dim
    """
    def forward(self, x):
        for i, layer in enumerate(self.layers):  # 使用ReLU函数作为激活函数
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):  # 模型定义
    num_classes = 1  # 单目标追踪
    backbone_net = build_backbone(settings, backbone_pretrained=True)  # 骨干网络
    featurefusion_network = build_featurefusion_network(settings)  # 特征融合网络，基于Transformer
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    """损失函数
    （1）分类：交叉熵损失函数，正负样本 损失权重正负1:16
    （2）回归：Giou + L1 仅计算正样本
    """
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
