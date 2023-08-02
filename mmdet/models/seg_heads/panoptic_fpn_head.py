# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union,List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ModuleList
from torch import Tensor
from mmdet.structures.mask.structures import BitmapMasks
import numpy as np
from scipy.ndimage import gaussian_filter

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers import ConvUpsample
from ..utils import interpolate_as
from .base_semantic_head import BaseSemanticHead


def getcenterfield(mask):
    if not mask.any():
        return np.zeros_like(mask)
    # Calculate the bounding box around the mask
    positive_indices = np.where(mask)
    min_row, min_col = np.min(positive_indices, axis=1)
    max_row, max_col = np.max(positive_indices, axis=1)
    center = ((min_col + max_col) // 2, (min_row + max_row) // 2)
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    # Generate the Gaussian kernel
    deviation = (width /3, height / 3)
    def generate_2d_gaussian_kernel(shape, center, deviation):
        x, y = np.indices(shape)
        y_center, x_center = center
        exponent = -((x - x_center)**2 / (2 * deviation[1]**2) + (y - y_center)**2 / (2 * deviation[0]**2))
        gaussian_kernel = np.exp(exponent)
        return gaussian_kernel
    kernel = generate_2d_gaussian_kernel(mask.shape, center, deviation)
    filtered_mask = kernel * mask
    return filtered_mask

def getdistancefield(mask):
    if not mask.any():
        distances_x = np.zeros_like(mask)
        distances_y = np.zeros_like(mask)
        distances = np.stack((distances_x, distances_y), axis=-1)
        return distances

    # Find the indices of positive elements in the mask
    positive_indices = np.where(mask)

    # Calculate the bounding box around the positive elements
    min_row, min_col = np.min(positive_indices, axis=1)
    max_row, max_col = np.max(positive_indices, axis=1)

    # Calculate the center point of the bounding box
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2

    # Calculate the width and height of the bounding box
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    # Calculate the coordinates of each pixel in the image
    rows, cols = np.indices(mask.shape[:2])

    # Calculate the distances between each pixel and the center point
    # -1 yerine +1 yazdim burda. Bizim durumda farketmez diye
    distances_x = (cols - center_col) / (width + 1)
    distances_y = (rows - center_row) / (height + 1)

    # Stack the distances along the channel axis
    distances = np.stack((distances_x, distances_y), axis=-1)
    masked_tensor = distances * mask[..., None]
    return masked_tensor

def getfields(mask):
    distancefield = getdistancefield(mask)
    centerfield = getcenterfield(mask)
    return np.concatenate((distancefield, centerfield[..., None]), axis=-1)

thinglabels={2,3,4,5,8,9,10,11}
#labels = gt_instance_mask_labels
#masks = gt_instance_masks
def batch_getfields_classwise(bitmap_batch,tensorlabels):
    batched_masks = [mask.to_ndarray() for mask in bitmap_batch]
    labels = [label.cpu().numpy() for label in tensorlabels]
    batch_fields = []
    out_tensors = []
    for labels,masks in zip(labels,batched_masks):
        dummy = getfields(masks[0])*0
        fields = {i:torch.from_numpy(dummy.copy()) for i in thinglabels}
        for label,mask in zip(labels,masks):
            fields[int(label)] += torch.from_numpy(getfields(mask))
        batch_fields.append(fields)
        out_tensors.append(torch.cat(tuple(fields.values()),dim = -1).permute([2,0,1]))
    return torch.stack(out_tensors,dim=0)


@MODELS.register_module()
class PanopticFPNHead(BaseSemanticHead):
    """PanopticFPNHead used in Panoptic FPN.

    In this head, the number of output channels is ``num_stuff_classes
    + 1``, including all stuff classes and one thing class. The stuff
    classes will be reset from ``0`` to ``num_stuff_classes - 1``, the
    thing classes will be merged to ``num_stuff_classes``-th channel.

    Arg:
        num_things_classes (int): Number of thing classes. Default: 80.
        num_stuff_classes (int): Number of stuff classes. Default: 53.
        in_channels (int): Number of channels in the input feature
            map.
        inner_channels (int): Number of channels in inner features.
        start_level (int): The start level of the input features
            used in PanopticFPN.
        end_level (int): The end level of the used features, the
            ``end_level``-th layer will not be used.
        conv_cfg (Optional[Union[ConfigDict, dict]]): Dictionary to construct
            and config conv layer.
        norm_cfg (Union[ConfigDict, dict]): Dictionary to construct and config
            norm layer. Use ``GN`` by default.
        init_cfg (Optional[Union[ConfigDict, dict]]): Initialization config
            dict.
        loss_seg (Union[ConfigDict, dict]): the loss of the semantic head.
    """

    def __init__(self,
                 num_things_classes: int = 8,
                 num_stuff_classes: int = 21,
                 in_channels: int = 256,
                 inner_channels: int = 128,
                 start_level: int = 0,
                 end_level: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_seg: ConfigType = dict(
                     type='MSELoss',
                     loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        seg_rescale_factor = 1 / 2**(start_level + 2)
        super().__init__(
            num_classes=num_stuff_classes+1,
            seg_rescale_factor=seg_rescale_factor,
            loss_seg=loss_seg,
            init_cfg=init_cfg)
        
        loss_fields_dict = dict(
                     type='MSELoss',
                     loss_weight=1.0)
        self.loss_fields = MODELS.build(loss_fields_dict)

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        self.end_level = end_level
        self.num_stages = end_level - start_level
        self.inner_channels = inner_channels

        self.conv_upsample_layers = ModuleList()
        for i in range(start_level, end_level):
            self.conv_upsample_layers.append(
                ConvUpsample(
                    in_channels,
                    inner_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ))
        #Burada numclasses 3 kati oldu
        self.conv_logits = nn.Conv2d(inner_channels, self.num_classes, 1)
        self.conv_field_upsample = ConvUpsample(inner_channels,inner_channels,num_layers=2,num_upsample=2)
        self.conv_fields = nn.Conv2d(inner_channels, self.num_things_classes*3, 1)

    def _set_things_to_void(self, gt_semantic_seg: Tensor) -> Tensor:
        """Merge thing classes to one class.

        In PanopticFPN, the background labels will be reset from `0` to
        `self.num_stuff_classes-1`, the foreground labels will be merged to
        `self.num_stuff_classes`-th channel.
        """
        gt_semantic_seg = gt_semantic_seg.int()
        fg_mask = gt_semantic_seg < self.num_things_classes
        bg_mask = (gt_semantic_seg >= self.num_things_classes) * (
            gt_semantic_seg < self.num_things_classes + self.num_stuff_classes)

        new_gt_seg = torch.clone(gt_semantic_seg)
        new_gt_seg = torch.where(bg_mask,
                                 gt_semantic_seg - self.num_things_classes,
                                 new_gt_seg)
        new_gt_seg = torch.where(fg_mask,
                                 fg_mask.int() * self.num_stuff_classes,
                                 new_gt_seg)
        return new_gt_seg

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Dict[str, Tensor]: The loss of semantic head.
        """
        #Burada kodum var

        sonuclar = self(x)
        seg_preds = sonuclar['seg_preds']
        seg_fields = sonuclar['seg_fields']
        gt_semantic_segs = [
            data_sample.gt_sem_seg.sem_seg
            for data_sample in batch_data_samples
        ]
        gt_instance_masks = [
            data_sample.gt_instances.masks
            for data_sample in batch_data_samples
        ]
        
        gt_instance_mask_labels = [
            data_sample.gt_instances.labels
            for data_sample in batch_data_samples
        ]

        gt_fields = batch_getfields_classwise(gt_instance_masks,gt_instance_mask_labels).to(torch.float32)
        

        gt_semantic_segs = torch.stack(gt_semantic_segs)
        if self.seg_rescale_factor != 1.0:
            gt_semantic_segs = F.interpolate(
                gt_semantic_segs.float(),
                scale_factor=self.seg_rescale_factor,
                mode='nearest').squeeze(1)

        # Things classes will be merged to one class in PanopticFPN.
        gt_semantic_segs = self._set_things_to_void(gt_semantic_segs)

        if seg_preds.shape[-2:] != gt_semantic_segs.shape[-2:]:
            seg_preds = interpolate_as(seg_preds, gt_semantic_segs)
        seg_preds = seg_preds.permute((0, 2, 3, 1))

        loss_seg = self.loss_seg(
            seg_preds.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_segs.reshape(-1).long())

        loss_fields = self.loss_fields(seg_fields,gt_fields.cuda())
        #Bura benden
        torch.autograd.set_detect_anomaly(True)
        loss_seg += loss_fields
        torch.autograd.set_detect_anomaly(False)
        return dict(loss_seg=loss_seg)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        nn.init.normal_(self.conv_logits.weight.data, 0, 0.01)
        self.conv_logits.bias.data.zero_()
        nn.init.normal_(self.conv_fields.weight.data, 0, 0.01)
        self.conv_fields.bias.data.zero_()

    def forward(self, x: Tuple[Tensor]) -> Dict[str, Tensor]:
        """Forward.

        Args:
            x (Tuple[Tensor]): Multi scale Feature maps.

        Returns:
            dict[str, Tensor]: semantic segmentation predictions and
                feature maps.
        """
        # the number of subnets must be not more than
        # the length of features.
        assert self.num_stages <= len(x)

        feats = []
        for i, layer in enumerate(self.conv_upsample_layers):
            f = layer(x[self.start_level + i])
            feats.append(f)

        seg_feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        seg_preds = self.conv_logits(seg_feats)
        upsampledfields = self.conv_field_upsample(seg_feats)
        seg_fields = self.conv_fields(upsampledfields)
        out = dict(seg_preds=seg_preds, seg_feats=seg_feats,seg_fields=seg_fields)
        return out
    
    def predict(self,
                x: Union[Tensor, Tuple[Tensor]],
                batch_img_metas: List[dict],
                rescale: bool = False) -> List[Tensor]:
        """Test without Augmentation.

        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_img_metas (List[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[Tensor]: semantic segmentation logits.
        """
        seg_preds = self.forward(x)['seg_preds']
        seg_preds = F.interpolate(
            seg_preds,
            size=batch_img_metas[0]['batch_input_shape'],
            mode='bilinear',
            align_corners=False)
        seg_preds = [seg_preds[i] for i in range(len(batch_img_metas))]

        if rescale:
            seg_pred_list = []
            for i in range(len(batch_img_metas)):
                h, w = batch_img_metas[i]['img_shape']
                seg_pred = seg_preds[i][:, :h, :w]

                h, w = batch_img_metas[i]['ori_shape']
                seg_pred = F.interpolate(
                    seg_pred[None],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)[0]
                seg_pred_list.append(seg_pred)
        else:
            seg_pred_list = seg_preds

        return seg_pred_list