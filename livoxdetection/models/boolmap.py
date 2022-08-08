import copy
import torch
import torch.nn as nn

class BoolMap(nn.Module):
    '''
             ^ X,W
             |
             |
     Y,H     |
    <--------o
    '''
    def __init__(self, point_cloud_range,voxel_size=[0.2,0.2,0.2], **kwargs):
        super().__init__()
        # VOXEL SIZE
        self.DX, self.DY, self.DZ = \
            voxel_size[0], voxel_size[1],voxel_size[2]
        
        # ROI in meter
        self.m_x_min = point_cloud_range[0]
        self.m_x_max = point_cloud_range[3]

        self.m_y_min = point_cloud_range[1]
        self.m_y_max = point_cloud_range[4]

        self.m_z_min = point_cloud_range[2]
        self.m_z_max = point_cloud_range[5]

        # SIZE of BEV map
        self.BEV_W = round((point_cloud_range[3]-point_cloud_range[0])/self.DX)
        self.BEV_H = round((point_cloud_range[4]-point_cloud_range[1])/self.DY)
        self.BEV_C = round((point_cloud_range[5]-point_cloud_range[2])/self.DZ)

        self.num_bev_features = self.BEV_C
        

    def forward(self, batch_dict):
        """
        Args:
            data_dict:
                points: (num_points, 1+4)

        Returns:
            batch_dict:
                batch_dict['spatial_features'] = spatial_features (B,C,H,W)

        """
        pc_lidar = batch_dict['points'].clone() 

        bev_img = torch.cuda.BoolTensor(batch_dict['batch_size'],self.BEV_C,self.BEV_H,self.BEV_W).fill_(0)
        pc_lidar[:,1]=((pc_lidar[:,1]-self.m_x_min)/self.DX)
        pc_lidar[:,2]=((pc_lidar[:,2]-self.m_y_min)/self.DY)
        pc_lidar[:,3]=((pc_lidar[:,3]-self.m_z_min)/self.DZ)
        pc_lidar = pc_lidar.trunc().long()
        bev_img[pc_lidar[:,0], pc_lidar[:,3], pc_lidar[:,2], pc_lidar[:,1]] = 1
        bev_img = bev_img.float() 
        batch_dict['spatial_features'] = bev_img

        return batch_dict
