# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor
from typing import List

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from mmseg.registry import MODELS
from ..utils import resize
from ..losses import accuracy
import torch.nn.functional as F


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d, guide_channel=32, channel=32, innerchannel=256):
        super(ConvGuidedFilter, self).__init__()
        self.norm = norm

        self.box_filter = nn.Conv2d(channel, channel, kernel_size=3, padding=radius, dilation=radius, bias=False,
                                    groups=1)
        self.box_filter2 = nn.Conv2d(guide_channel//3, channel, kernel_size=3, padding=radius, dilation=radius, bias=False,
                                     groups=1)
        self.conv_a = nn.Sequential(nn.Conv2d(2 * channel, innerchannel, kernel_size=1, bias=False),
                                    self.norm(innerchannel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(innerchannel, innerchannel, kernel_size=1, bias=False),
                                    self.norm(innerchannel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(innerchannel, channel, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr):
        # 先将x_lr转为相同的维度
        x_lr = self.box_filter2(x_lr)
        # y_lr 输入  x_lr 引导图
        b, c, h_lrx, w_lrx = x_lr.size()
        # _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((b, c, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N
        ## mean_y
        mean_y = self.box_filter(y_lr) / N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))  # 将协方差和方差两个变量得到A的原本计算过程变为可学习的卷积过程
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_lrx, w_lrx), mode='bilinear', align_corners=True)

        return mean_A * x_lr + mean_b


class EnhanceGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d, guide_channel=32, channel=32, innerchannel=256, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.guidedfilter1 = ConvGuidedFilter(radius=radius, norm=norm, guide_channel=guide_channel, channel=channel, innerchannel=innerchannel)
        self.guidedfilter2 = ConvGuidedFilter(radius=radius, norm=norm, guide_channel=guide_channel, channel=channel, innerchannel=innerchannel)
        self.guidedfilter3 = ConvGuidedFilter(radius=radius, norm=norm, guide_channel=guide_channel, channel=channel, innerchannel=innerchannel)
        self.weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.fusion_convdistance = ConvModule(
            in_channels=channel * 3,
            out_channels=channel,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x_lr, y_lr):
        #  y_lr input  x_lr guide image
        nor_weights = F.softmax(self.weight, dim=0)
        chunks = torch.chunk(x_lr, chunks=3, dim=1)
        guide1 = self.guidedfilter1(chunks[0], y_lr)
        guide2 = self.guidedfilter2(chunks[1], y_lr)
        guide3 = self.guidedfilter3(chunks[2], y_lr)
        guide = torch.cat([guide1 * nor_weights[0], guide2 * nor_weights[1], guide3 * nor_weights[2]], 1)
        x = self.fusion_convdistance(guide)
        return x


@MODELS.register_module()
class StructureContextualDecoderHead(BaseDecodeHead):
    """
    This head is the implementation of SCT's decoder: SCDecoder.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        """
        In the context of the Orientation Guided Attention Module (OGAM), 
        self.gfilter0-self.gfilter3 is the use of guided filtering and the final processing stages.
        """
        self.gfilter0 = EnhanceGuidedFilter(guide_channel=45, channel=self.in_channels[0], innerchannel=256, norm_cfg=self.norm_cfg,  act_cfg=self.act_cfg)
        self.gfilter1 = EnhanceGuidedFilter(guide_channel=self.in_channels[0]*3, channel=self.in_channels[1], innerchannel=256, norm_cfg=self.norm_cfg,  act_cfg=self.act_cfg)  
        self.gfilter2 = EnhanceGuidedFilter(guide_channel=self.in_channels[1]*3, channel=self.in_channels[2], innerchannel=256, norm_cfg=self.norm_cfg,  act_cfg=self.act_cfg)  
        self.gfilter3 = EnhanceGuidedFilter(guide_channel=self.in_channels[2]*3, channel=self.in_channels[3], innerchannel=256, norm_cfg=self.norm_cfg,  act_cfg=self.act_cfg)  
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # The SDB module
        self.convs1_1 = ConvModule(
            in_channels=self.in_channels[0] + self.in_channels[1],
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding=1
        )
        self.convs1_2 = ConvModule(
            in_channels=self.channels * 2 + self.in_channels[0],
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding=1
        )
        self.convs1_3 = ConvModule(
            in_channels=self.channels * 3 + self.in_channels[0],
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding=1
        )
        self.convs2_1 = ConvModule(
            in_channels=self.in_channels[1] + self.in_channels[2],
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding=1
        )
        self.convs2_2 = ConvModule(
            in_channels=self.channels * 2 + self.in_channels[1],
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding=1
        )
        self.convs3_1 = ConvModule(
            in_channels=self.in_channels[2] + self.in_channels[3],
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding=1
        )
        self.fusion_conv1 = ConvModule(
            in_channels=self.channels * 1,  # + self.in_channels[0]
            out_channels=self.channels,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            # act_cfg=self.act_cfg,
            padding=1)
        self.fusion_conv2 = ConvModule(
            in_channels=self.channels * 2,  # + self.in_channels[0]
            out_channels=self.channels,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            # act_cfg=self.act_cfg,
            padding=1)
        self.cls_seg1 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        self.cls_seg2 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        self.cls_seg_distance = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        # self.cls_seg3 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        self.fusion_conv = ConvModule(
            in_channels=self.channels * (num_inputs - 1),  # + self.in_channels[0]
            out_channels=self.channels,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            # act_cfg=self.act_cfg,
            padding=1)

        # The BSRB moudle
        self.convs_distance = nn.ModuleList()
        for i in range(num_inputs):
            self.convs_distance.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_convdistance = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        high_inform = inputs[4]
        inputs = [inputs[i] for i in self.in_index]

        inputs = self._transform_inputs(inputs)
        x1, x2, x3, x4 = inputs[0], inputs[1], inputs[2], inputs[3] 
        h1, h2, h3, h4 = high_inform[0], high_inform[1], high_inform[2], high_inform[3]

        x1 = self.gfilter0(h1, x1)
        x2 = self.gfilter1(h2, x2)
        x3 = self.gfilter2(h3, x3)
        x4 = self.gfilter3(h4, x4)

        outs = []
        outs_distance = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs_distance[idx]
            outs_distance.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        distance = self.fusion_convdistance(torch.cat(outs_distance, dim=1))
        distance1 = self.cls_seg_distance(distance)
        outs.append(distance1)

        x2_up = resize(input=x2, size=x1.shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)
        x1_1 = self.convs1_1(torch.cat((x1, x2_up), 1))  

        x4_up = resize(input=x4, size=x3.shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)
        x3_1 = self.convs3_1(torch.cat((x3, x4_up), 1)) 
        x3_up = resize(input=x3, size=x2.shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)
        x2_1 = self.convs2_1(torch.cat((x2, x3_up), 1))  
        x2_1_up = resize(input=x2_1, size=x1.shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)
        x1_2 = self.convs1_2(torch.cat((x1, x1_1, x2_1_up), 1))  

        x3_1_up = resize(input=x3_1, size=x2.shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)
        x2_2 = self.convs2_2(torch.cat((x2, x2_1, x3_1_up), 1))  
        x2_2_up = resize(input=x2_2, size=x1.shape[2:], mode=self.interpolate_mode, align_corners=self.align_corners)
        x1_3 = self.convs1_3(torch.cat((x1_2, x1_1, x1, x2_2_up), 1))

        x1_1_distance = torch.mul(x1_1, distance)
        fusion1 = self.fusion_conv1(x1_1_distance)

        segout1 = self.cls_seg1(fusion1)
        outs.append(segout1)
        x1_2_distance = torch.mul(x1_2, distance)
        fusion2 = self.fusion_conv2(torch.cat((x1_2_distance, fusion1), 1))

        segout2 = self.cls_seg2(fusion2)
        outs.append(segout2)

        x1_3_distance = torch.mul(x1_3, distance)
        out = self.fusion_conv(torch.cat((fusion1, fusion2, x1_3_distance), dim=1))

        out = self.cls_seg(out)
        outs.append(out)
        return outs

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        batch_size = len(seg_logits)
        for i in range(batch_size):
            seg_logits[i] = resize(
                input=seg_logits[i],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name == 'loss_ce':
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits[batch_size - 1],
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                continue
            if loss_decode.loss_name == 'loss_focal':
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits[batch_size - 1],
                    seg_label,
                    weight=seg_weight,
                    ignore_index=300)
                continue
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        # print(seg_logits[batch_size-1].shape, seg_label.shape)
        loss['acc_seg'] = accuracy(seg_logits[batch_size - 1], seg_label, ignore_index=None)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        batch_size = len(seg_logits)
        seg_logits = resize(
            input=seg_logits[batch_size - 1],
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
