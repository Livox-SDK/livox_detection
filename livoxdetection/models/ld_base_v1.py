#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boolmap import BoolMap
from .resfpn import ResBEVBackboneConcat
from .centerhead import CenterHead

class LD_base(nn.Module):
    def __init__(self):
        super(LD_base, self).__init__()
        self.voxel_size = [0.2, 0.2, 0.2]
        self.point_cloud_range = [0, -44.8, -2, 224, 44.8, 4] 
        self.point_to_bev = BoolMap(self.point_cloud_range, voxel_size=self.voxel_size)
        self.backbone = ResBEVBackboneConcat(30)
        self.head = CenterHead(input_channels=128, 
                               num_class=3, 
                               class_names=['Vehicle', 'Pedestrian', 'Cyclist'], 
                               point_cloud_range=self.point_cloud_range, 
                               voxel_size=self.voxel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, batch_dict):
        batch_dict = self.point_to_bev(batch_dict)
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.head(batch_dict)
        return batch_dict['final_box_dicts']

