import os
import numpy as np
import tensorflow as tf
import config.config as cfg
from networks.model import *
import csv
import lib_cpp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_INDEX)

# voxel size: Related to detection range
nh = round((cfg.RANGE['X_MAX']-cfg.RANGE['X_MIN']) / cfg.VOXEL_SIZE[0])
nw = round((cfg.RANGE['Y_MAX']-cfg.RANGE['Y_MIN']) / cfg.VOXEL_SIZE[1])
nz = round((cfg.RANGE['Z_MAX']-cfg.RANGE['Z_MIN']) / cfg.VOXEL_SIZE[2])

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
              )


class Detector(object):
    def __init__(self, *, nms_threshold=0.1, weight_file=None):
        self.nx = int(
            (cfg.RANGE['X_MAX']-cfg.RANGE['X_MIN'])/cfg.VOXEL_SIZE[0]+0.5)
        self.ny = int(
            (cfg.RANGE['Y_MAX']-cfg.RANGE['Y_MIN'])/cfg.VOXEL_SIZE[1]+0.5)
        self.net = livox_model(self.nx, self.ny)
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
                input_bev_img_pl, label_gt_total_channels = \
                    self.net.placeholder_inputs(cfg.BATCH_SIZE)

                end_points = self.net.get_model(input_bev_img_pl,
                                                label_gt_total_channels)

                saver = tf.train.Saver()
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                self.sess = tf.Session(config=config)
                saver.restore(self.sess, cfg.MODEL_PATH)
                self.ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                            'end_points': end_points,  # output
                            }

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    def get_3d_box(self, box_size, heading_angle, center):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (l,w,h)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.roty(heading_angle)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def detect(self, batch_bev_img):
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)
        result = lib_cpp.cal_result_single(feature_out[0, :, :, :],
                                           cfg.BOX_THRESHOLD, nh, nw, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)

        is_obj_list = result[:, 0].tolist()
        reg_theta_list = result[:, 2].tolist()
        reg_m_x_list = result[:, 5].tolist()
        reg_m_y_list = result[:, 6].tolist()
        reg_w_list = result[:, 4].tolist()
        reg_l_list = result[:, 3].tolist()
        obj_cls_list = result[:, 1].tolist()

        reg_m_z_list = result[:, 7].tolist()
        reg_h_list = result[:, 8].tolist()

        results = []
        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)
            box3d_pts_3d[:, 0:3] = self.get_3d_box(
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]),
                reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i]+3.8-reg_h_list[i], reg_m_y_list[i]))
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T
            if int(obj_cls_list[i]) == 0:
                cls_name = "car"
            elif int(obj_cls_list[i]) == 1:
                cls_name = "bus"
            elif int(obj_cls_list[i]) == 2:
                cls_name = "truck"
            elif int(obj_cls_list[i]) == 3:
                cls_name = "pedestrian"
            else:
                cls_name = "bimo"
            results.append([cls_name,
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i]])
        return results


def data2voxel(datapath):
    dw = cfg.VOXEL_SIZE[1]
    dh = cfg.VOXEL_SIZE[0]
    dz = cfg.VOXEL_SIZE[2]
    offw = -cfg.RANGE['Y_MIN']
    offh = -cfg.RANGE['X_MIN']
    offz = -cfg.RANGE['Z_MIN']
    data = [i*0 for i in range(nh*nw*nz)]
    with open(datapath, 'r') as c:
        r = csv.reader(c)
        for line in r:
            Z = float(line[0])
            X = -1.0*float(line[1])
            Y = -1.0*float(line[2])  # 注意地面高度在-1.9m左右
            if(X > cfg.RANGE['Y_MIN'] and X < cfg.RANGE['Y_MAX'] and
               Z > cfg.RANGE['X_MIN'] and Z < cfg.RANGE['X_MAX'] and
               Y > cfg.RANGE['Z_MIN'] and Y < cfg.RANGE['Z_MAX']):
                pixel_y = int(Z/dh)
                pixel_x = int(X/dw+offw/dw)
                channel = int(Y/dz+offz/dz)
                data[pixel_y*nw*30+pixel_x*30+channel] = 1
    voxel = np.reshape(data, (nh, nw, nz))
    return voxel


def main():
    detector = Detector()
    datapath = './data/test_data.csv'
    vox = data2voxel(datapath)
    vox = np.expand_dims(vox, axis=0)
    result = detector.detect(vox)
    with open('res2.txt', 'w') as fr:
        for i in range(len(result)):
            for j in range(len(result[i])):
                fr.write(str(result[i][j])+' ')
            fr.write('\n')


if __name__ == '__main__':
    main()
