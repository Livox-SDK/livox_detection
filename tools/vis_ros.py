import glob
import time
import copy
import argparse
from pathlib import Path

# numpy and torch
import torch
import numpy as np
from torch.utils.data import DataLoader

# rospy
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray 
import ros_numpy

# ros marker
gtbox_array = MarkerArray()
marker_array = MarkerArray()
marker_array_text = MarkerArray()

"""
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
"""

lines = [[0, 1], [1, 2], [2, 3], [3, 0], 
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]]

color_maps = {'car': [0, 1, 1], 'Car' : [0, 1, 1], 'truck': [0, 1, 1], 'Vehicle': [0, 1, 1], 'construction_vehicle': [0, 1, 1], 'bus': [0, 1, 1], 'trailer': [0, 1, 1],
        'motorcycle': [0, 1, 0], 'bicycle': [0, 1, 0], 'Cyclist': [0, 1, 0],
        'Pedestrian': [1, 1, 0], 'pedestrian': [1, 1, 0], 
        'barrier' : [1, 1, 1], 'traffic_cone': [1, 1, 1]}

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

class ROS_MODULE:
    def __init__(self):
        self.class_names = ['Vehicle', 'Pedestrian', 'Cyclist'] 
        
        # ros config
        rospy.init_node('ros_demo', anonymous=True)

        # create Publisher for visualization.
        self.pointcloud_pub = rospy.Publisher(
                '/pointcloud', PointCloud2, queue_size=10
                ) 
        self.gtbox_array_pub = rospy.Publisher(
                '/detect_gtbox', MarkerArray, queue_size=10
                )
        self.marker_pub = rospy.Publisher(
                '/detect_box3d', MarkerArray, queue_size=10
                )

        self.marker_text_pub = rospy.Publisher(
                '/text_det', MarkerArray, queue_size=10
                )

    @staticmethod
    def gpu2cpu(data_dict, pred_dicts):
        data_dict['points'] = data_dict['points'].cpu().numpy()
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'].cpu().numpy()
        pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu().numpy()
        pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].cpu().numpy()
        pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].cpu().numpy()
        torch.cuda.empty_cache()
        return data_dict, pred_dicts
 
    def ros_print(self, pts, pred_dicts=None, last_box_num=None, gt_boxes=None, last_gtbox_num=None):
        def xyzr_to_pc2(pts, stamp, frame_id):
            data = np.zeros(pts.shape[0], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)])
            data['x'] = pts[:, 0]
            data['y'] = pts[:, 1]
            data['z'] = pts[:, 2]
            data['intensity'] = pts[:, 3]
            msg = ros_numpy.msgify(PointCloud2, data)
            msg.header.stamp = stamp
            msg.header.frame_id = frame_id
            return msg

        # ROS Header
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'livox_frame'

        # point cloud visualization

        #import pandas as pd
        #pts_des = pd.DataFrame(pts, columns=['batch_id', 'x', 'y', 'z', 'intensity'])
        #print(pts_des.describe(include='all'))
        
        pointcloud_msg = xyzr_to_pc2(pts, header.stamp, header.frame_id)
        self.pointcloud_pub.publish(pointcloud_msg)
        
        # print(pointcloud_msg)
        # input()

        # print("format of boxes\n", pred_dicts[0]['pred_boxes'][0])
        # print("format of scores\n", pred_dicts[0]['pred_scores'][0])
        # print("format of labels\n", pred_dicts[0]['pred_labels'][0])

                
        if gt_boxes is not None:
            gtbox_array.markers.clear()
            gt_boxes = boxes_to_corners_3d(gt_boxes)
            for obid in range(gt_boxes.shape[0]):
                ob = gt_boxes[obid]

                # boxes
                marker = Marker()
                marker.header.frame_id = header.frame_id
                marker.header.stamp = header.stamp
                marker.id = obid
                marker.action = Marker.ADD
                marker.type = Marker.LINE_LIST
                marker.lifetime = rospy.Duration(0)

                # print(labs)
                # print(ob)
                marker.color.r = 1
                marker.color.g = 1
                marker.color.b = 1
                marker.color.a = 1
                marker.scale.x = 0.05
                
                marker.points = []
                for line in lines:
                    ptu = gt_boxes[obid][line[0]]
                    ptv = gt_boxes[obid][line[1]]
                    marker.points.append(Point(ptu[0], ptu[1], ptu[2]))
                    marker.points.append(Point(ptv[0], ptv[1], ptv[2]))
                
                gtbox_array.markers.append(marker)

            # clear ros cache   
            if last_gtbox_num > gt_boxes.shape[0]:
                for i in range(gt_boxes.shape[0], last_gtbox_num):
                    marker = Marker()
                    marker.header.frame_id = header.frame_id
                    marker.header.stamp = header.stamp
                    marker.id = i
                    marker.action = Marker.ADD
                    marker.type = Marker.LINE_LIST
                    marker.lifetime = rospy.Duration(0.01)
                    marker.color.a = 0
                    gtbox_array.markers.append(marker)

            self.gtbox_array_pub.publish(gtbox_array)

        if pred_dicts is not None:
            boxes = boxes_to_corners_3d(pred_dicts[0]['pred_boxes'])
            score = pred_dicts[0]['pred_scores']
            label = pred_dicts[0]['pred_labels']
            # print('corner points \n', pts)

            marker_array.markers.clear()
            marker_array_text.markers.clear()
            for obid in range(boxes.shape[0]):
                ob = boxes[obid]

                # boxes
                marker = Marker()
                marker.header.frame_id = header.frame_id
                marker.header.stamp = header.stamp
                marker.id = obid * 2
                marker.action = Marker.ADD
                marker.type = Marker.LINE_LIST
                marker.lifetime = rospy.Duration(0)

                # print(labs)
                color = color_maps[self.class_names[np.int(label[obid])-1]]

                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.8
                marker.scale.x = 0.05
                
                marker.points = []
                for line in lines:
                    ptu = boxes[obid][line[0]]
                    ptv = boxes[obid][line[1]]
                    marker.points.append(Point(ptu[0], ptu[1], ptu[2]))
                    marker.points.append(Point(ptv[0], ptv[1], ptv[2]))
                marker_array.markers.append(marker)
                
                # confidence
                markert = Marker()
                markert.header.frame_id = header.frame_id
                markert.header.stamp = header.stamp
                markert.id = obid * 2 + 1
                markert.action = Marker.ADD
                markert.type = Marker.TEXT_VIEW_FACING
                markert.lifetime = rospy.Duration(0)

                # print(labs)
                color = color_maps[self.class_names[np.int(label[obid])-1]]

                markert.color.r = color[0]
                markert.color.g = color[1]
                markert.color.b = color[2]
                markert.color.a = 1
                markert.scale.z = 0.6
               
                markert.pose.orientation.w = 1.0
                
                markert.pose.position.x = (boxes[obid][0][0] + boxes[obid][2][0]) / 2
                markert.pose.position.y = (boxes[obid][0][1] + boxes[obid][2][1]) / 2
                markert.pose.position.z = (boxes[obid][0][2] + boxes[obid][4][2]) / 2 
                markert.text = self.class_names[label[obid]-1] + ':' + str(np.floor(score[obid] * 100)/100)
                marker_array_text.markers.append(markert)

            # clear ros cache   
            if last_box_num > boxes.shape[0]:
                for i in range(boxes.shape[0], last_box_num):
                    marker = Marker()
                    marker.header.frame_id = header.frame_id
                    marker.header.stamp = header.stamp
                    marker.id = i * 2
                    marker.action = Marker.ADD
                    marker.type = Marker.LINE_LIST
                    marker.lifetime = rospy.Duration(0.01)
                    marker.color.a = 0
                    marker_array.markers.append(marker)

                    markert = Marker()
                    markert.header.frame_id = header.frame_id
                    markert.header.stamp = header.stamp
                    markert.id = i * 2 + 1
                    markert.action = Marker.ADD
                    markert.type = Marker.TEXT_VIEW_FACING
                    markert.lifetime = rospy.Duration(0.01)
                    markert.color.a = 0
                    marker_array_text.markers.append(markert)


            # publish
            self.marker_pub.publish(marker_array)
            self.marker_text_pub.publish(marker_array_text)
        
        box_size = 0 if pred_dicts is None else boxes.shape[0]
        gtbox_size = 0 if gt_boxes is None else gt_boxes.shape[0]

        return box_size, gtbox_size
        # input()
