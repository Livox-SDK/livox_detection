"""
Modified from the original implementation of CenterPoint in OpenPCDet.
Copyright [2020] [Shaoshuai.Shi] SPDX-License-Identifier: Apache-2.0(http://www.apache.org/licenses/LICENSE-2.0)
Modifications Copyright [2022] [LIVOX Perception Group]
"""

import copy
import torch
import numpy as np
import torch.nn as nn
from . import model_nms_utils

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_normal_(m.weight)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, 
                 input_channels, 
                 num_class, 
                 class_names, 
                 point_cloud_range, 
                 voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.num_class = num_class
        self.class_names = class_names
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # config
        self.class_name_branch_each_head = [class_names]

        self.separate_head_cfg = {
            'HEAD_ORDER': ['center', 'center_z', 'dim', 'rot'],
            'HEAD_DICT': {
                'center': {'out_channels':2, 'num_conv':1},
                'center_z': {'out_channels':1, 'num_conv':1},
                'dim': {'out_channels':3, 'num_conv':1},
                'rot': {'out_channels':2, 'num_conv':1}
            }
        }

        self.feature_map_stride = 1

        self.POST_PROCESSING = {
            'SCORE_THRESH': [0.2, 0.3, 0.3],
            'POST_CENTER_LIMIT_RANGE': self.point_cloud_range,
            'MAX_OBJ_PER_SAMPLE': 500,
            'NMS_CONFIG': {
                'NMS_TYPE': 'nms_gpu',
                'NMS_THRESH': 0.1,
                'NMS_PRE_MAXSIZE': 4096,
                'NMS_POST_MAXSIZE': 500
            }
        }

        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for cur_class_names in self.class_name_branch_each_head:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)
        
        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, 3, stride=1, padding=1,
                bias=True
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg['HEAD_DICT'])
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=1)
            self.heads_list.append(
                SeparateHead(
                    input_channels=64,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=True
                )
            )
        self.forward_ret_dict = {}

    @staticmethod
    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    @staticmethod
    def _transpose_and_gather_feat(feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = CenterHead._gather_feat(feat, ind)
        return feat

    @staticmethod
    def _topk(scores, K=40):
        batch, num_class, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_classes = (topk_ind // K).int()
        topk_inds = CenterHead._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = CenterHead._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = CenterHead._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_classes, topk_ys, topk_xs

    @staticmethod
    def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                                 point_cloud_range=None, voxel_size=None, feature_map_stride=None, K=100,
                                 score_thresh=None, post_center_limit_range=None):
        batch_size, num_class, _, _ = heatmap.size()

        scores, inds, class_ids, ys, xs = CenterHead._topk(heatmap, K=K)
        center = CenterHead._transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot_sin = CenterHead._transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
        rot_cos = CenterHead._transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
        center_z = CenterHead._transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = CenterHead._transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

        angle = torch.atan2(rot_sin, rot_cos)
        xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]

        final_box_preds = torch.cat((box_part_list), dim=-1)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        assert post_center_limit_range is not None
        mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)
        
        if score_thresh is not None:
            for i in range(len(score_thresh)):
                mask &= (~(final_class_ids == i)) | ((final_class_ids == i) & (final_scores > score_thresh[i])) 

        ret_pred_dicts = []
        for k in range(batch_size):
            cur_mask = mask[k]
            cur_boxes = final_box_preds[k, cur_mask]
            cur_scores = final_scores[k, cur_mask]
            cur_labels = final_class_ids[k, cur_mask]

            ret_pred_dicts.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels
            })
        return ret_pred_dicts

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg['POST_CENTER_LIMIT_RANGE']).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg['HEAD_ORDER'] else None

            final_pred_dicts = CenterHead.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, 
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg['MAX_OBJ_PER_SAMPLE'],
                score_thresh=post_process_cfg['SCORE_THRESH'],
                post_center_limit_range=post_center_limit_range
            )
                
            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                    nms_config=post_process_cfg['NMS_CONFIG'],
                    score_thresh=None
                )

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
        
        self.forward_ret_dict['pred_dicts'] = pred_dicts

        pred_dicts = self.generate_predicted_boxes(
            data_dict['batch_size'], pred_dicts
        )

        data_dict['final_box_dicts'] = pred_dicts
        return data_dict
