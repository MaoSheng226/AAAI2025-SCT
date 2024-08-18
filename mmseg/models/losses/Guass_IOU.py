# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from torch import Tensor
import cv2
import numpy as np
from .utils import get_class_weight, weight_reduce_loss, weighted_loss
import torchvision.transforms.functional as TF


# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

def generate_distance(mask):
    mask[mask == 255] = 0
    b,h,w = mask.shape
    out = []
    for i in range(b):
        maski = mask[i,:,:]
        mask_np = maski.cpu().numpy()
        mask1 = mask_np.astype(np.uint8)
        body = cv2.distanceTransform(mask1, distanceType=cv2.DIST_L2, maskSize=5)
        body = cv2.convertScaleAbs(body)
        # print(body)
        # tmp = body[np.where(body > 0)]
        # print(np.unique(tmp))
        # if len(tmp) != 0:
        #    a = np.where(0< body & body < 0.1 * tmp.max())
        #    body[a] = 255
        # b = np.where(body <= 0.5*tmp.mean())
        # body[b] = 0
        # _ret, result = cv2.threshold(body, 0.7 * body.max(), 255, cv2.THRESH_BINARY)

        body = cv2.normalize(body, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        body = (mask1 - body)
        bodys = body
        bodys = torch.from_numpy(bodys)
        out.append(bodys.unsqueeze(0))
    gt = torch.cat(out, dim=0)
    gt = gt.cuda()
    return gt

def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


def GuassBlur(target, kernel=7, std=10.0):
    """
    :param target:
    :param kernel:
    :param std:
    :return:
    """
    blur = TF.gaussian_blur(target, kernel_size=kernel, sigma=std)
    return blur

def Guass_IoUm_loss(pred,
                  target,
                  weight=None,
                  class_weight=None,
                  reduction: Union[str, None] = 'mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False,
                  kernel=5,
    distance, pred1, pred2, pred3 = pred[0], pred[1], pred[2], pred[3]
    GT_distance = generate_distance(target).to(torch.int64)
    # print(GT_distance.dtype, target.dtype)
    GT_2 = GuassBlur(target, kernel=kernel, std=std)
    GT_1 = GuassBlur(GT_2, kernel=kernel, std=std)
    GT1one_hot_target = GT_1  
    GT2one_hot_target = GT_2  
    GT3one_hot_target = target  
    GT_distance_onehot = GT_distance
    if (pred3.shape != target.shape):
        GT_distance_onehot = _expand_onehot_labels_dice(distance, GT_distance)
        GT1one_hot_target = _expand_onehot_labels_dice(pred1, GT_1)
        GT2one_hot_target = _expand_onehot_labels_dice(pred2, GT_2)
        GT3one_hot_target = _expand_onehot_labels_dice(pred3, target)
    iou1 = iou_loss(pred1, GT1one_hot_target)#iou_loss(pred1, blur_pred2)
    iou2 = iou_loss(pred2, GT2one_hot_target)#iou_loss(pred2, blur_pred3)
    # print(sumloss.shape, len(iou1), iou2)
    loss = iou1 + iou2 + iou_loss(pred3, GT3one_hot_target) + iou_loss(distance, GT_distance_onehot)
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class Guass_Iou_Loss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 kernel=5,
                 std=2,
                 loss_name='loss_guass_iou'):
        """Compute guass IoU loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.kernel = kernel    # 
        self.std = std          #

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred[0].shape[1] != 1:
                # softmax does not work when there is only 1 class
                for i in range(len(pred)):
                   pred[i] = pred[i].softmax(dim=1)
        loss = self.loss_weight * Guass_IoUm_loss(
            pred,
            target,
            weight,
            reduction= reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index,
            kernel=self.kernel,
            std=self.std)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

