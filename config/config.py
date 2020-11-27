CLASSES = ['car', 'bus', 'truck', 'pedestrian', 'bimo']

RANGE = {'X_MIN': -89.6,
         'X_MAX': 89.6,
         'Y_MIN': -22.4,
         'Y_MAX': 22.4,
         'Z_MIN': -3.0,
         'Z_MAX': 3.0}

VOXEL_SIZE = [0.2, 0.2, 0.2]
BATCH_SIZE = 1
MODEL_PATH = "model/livoxmodel"

OVERLAP = 11.2

GPU_INDEX = 0
NMS_THRESHOLD = 0.1
BOX_THRESHOLD = 0.6
