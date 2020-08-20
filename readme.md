## Livox Detection: trained based on LivoxDataset_v1.0 [\[LivoxDataset\]](https://www.livoxtech.com/cn/dataset)
The detector can run at least 90 FPS on 2080TI. The provided model was trained on LivoxDataset_v1.0 within 1w pointcloud sequence.

## Demo
highway_scene:
<div align="center"><img src="./data/highway_scene1.gif" width=90% /></div>

<div align="center"><img src="./data/highway_scene2.gif" width=90% /></div>

<div align="center"><img src="./data/highway_scene3.gif" width=90% /></div>

urban_scene:
<div align="center"><img src="./data/urban_scene1.gif" width=90% /></div>

<div align="center"><img src="./data/urban_scene2.gif" width=90% /></div>

<div align="center"><img src="./data/urban_scene3.gif" width=90% /></div>

# Introduction
Livox Detection is a robust,real time detection package for [*Livox LiDARs*](https://www.livoxtech.com/). The detector is designed for L3 and L4 autonomous driving. It can effectively detect within 100m under different vehicle speed conditions(`0~120km/h`). In addition, the detector can perform effective detection in different scenarios, such as high-speed scenes, structured urban road scenes, complex intersection scenes and some unstructured road scenes, etc. In addition, the detector is currently able to effectively detect 3D bounding boxes of five types of objects: `cars`, `trucks`, `bus`, `bimo` and `pedestrians`.

# Dependencies
- `python3.6+`
- `tensorflow1.13+` (tested on 1.13.0)
- `pybind11`
- `ros`

# Installation

1. Clone this repository.
2. Clone `pybind11` from [pybind11](https://github.com/pybind/pybind11).
```bash
$ cd utils/lib_cpp
$ git clone https://github.com/pybind/pybind11.git
```
3. Compile C++ module in utils/lib_cpp by running the following command.
```bash
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
4. copy the `lib_cpp.so` to root directory:
```bash
$ cp lib_cpp.so ../../../
```
5. download the [pre_trained model](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Showcase/model.zip) and unzip it to the root directory.

# Run

### 1. For single frame detection

For single frame data testing, the single-frame point cloud data can be stored in any format, such as csv, txt, pcd, bin, etc., but it is necessary to ensure that each frame of data contains at least the x, y, and z coordinates of the current frame of 3D point cloud.

We provide a frame of independent point cloud file in csv format for test instructions, in the `data` directory. Run directly:
```bash
$ python test.py
```
Then you can get a `res.txt` file of the detection results, each line of a detection object contains detection class, 8 corner coordinates of its bounding box and its confidence score.

For testing your own data, you need to change the file path to be loaded on line 166 in the `test.py` file to your own file path. In addition, for point cloud files of different formats, you need to change the point cloud analysis method in the `data2voxel` function on line 141 and modify it to the corresponding analysis method.

### 2. For sequence frame detection

Download the provided rosbags : [highwayscene1](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Showcase/highwayscene1.bag), [highwayscene2](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Showcase/highwayscene2.bag),[highwayscene3](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Showcase/highwayscene3.bag), [urban_scene](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Showcase/urban_scene.bag), and then

```bash
$ roscore

$ python livox_rosdetection.py

$ rosbag play *.bag -r 0.1
```
The network inference time is around `11ms`, but the point cloud data preprocessing module takes a lot of time based on python. If you want to get a faster real time detection demo, you can modify the point cloud data preprocessing module with c++.

To play with your own rosbag, please change your rosbag topic to `/livox/lidar`.

# Support
You can get support from Livox with the following methods :
- Send email to dev@livoxtech.com with a clear description of your problem and your setup
- Report issue on github
