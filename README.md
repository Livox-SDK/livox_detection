# Livox Detection V2.0

The detector in Livox Detection v2.0 now supports multiple point cloud datasets with different patterns. Combined with the advantages of HAP, this detector can achieve better perception performance compared with the Horizon. Another improvement is that we adopt anchor-free method inspired by CenterPoint to make the detector more flexible dealing with the multiple datasets.  The range of the available detection filed is forward 200m * 89.6m and the latency is about 45ms on 2080Ti.

Demo: [HAP1](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/demo/HAP/newHAP_HIGH.mp4) [HAP2](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/demo/HAP/newHAP_PED.mp4) [Horizon](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/demo/HAP/newHorizon.mp4) [64](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/demo/HAP/new64.mp4)

## News
- `2022.08.08` : livox_detection V2.0 released: Support multiple point cloud datasets with different patterns. The range of the available detection filed is forward 200m * 89.6m and the latency is about 45ms on 2080Ti
- `2020.11.26` : livox_detection V1.1 released: Support 360 degree detection (200m * 100m) with Livox lidars, run at least 20FPS on 2080Ti
- `2020.08.31` : livox_detection V1.0 released: 100m*50m detction for single Livox lidars, run at least 90FPS on 2080Ti
	
## Features
- Anchor-free method
- Support multiple point cloud datasets with different patterns.

## Setup
1. Install dependencies (Following dependencies have been tested.).
	- CUDA Toolkit: 10.2
	- python: 3.8
	- numpy: 1.23.1
	- pytorch: 1.8.2
	- ros: melodic
	- rospkg: 1.4.0
	- ros_numpy: 0.0.3 (sudo apt-get install ros-$ros_release-ros-numpy)
	- pyyaml
	- argparse 
2. Build this project
```bash
python3 setup.py develop
```

## Usage
1. Run ROS.
```
roscore
```
2. Move to 'tools' directory and run test_ros.py (pretrained model: ../pt/livox_model_1.pt or ../pt/livox_model_2.pt).
```
cd tools
python3 test_ros.py --pt ../pt/livox_model_1.pt
```
3. Play rosbag. (Please adjust the ground plane to 0m and keep it horizontal. The topic of pointcloud2 should be /livox/lidar)
```
rosbag play [bag path]
```
4. Visualize the results.
```
rviz -d rviz.rviz
```

## Acknowledgements
- This project is based on the framework,  [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which provides powerful and flexible cuda operators.
- This detector was inspired by the anchor-free method, CenterPoint, which was proposed in [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275).

## Contact
You can get support from LIVOX through the following methods:
- Send an email to cs@livoxtech.com with a clear description of your problem.
- Submit issues on github.
