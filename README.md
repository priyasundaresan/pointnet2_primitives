# KITE: Keypoint-Conditioned Policies for Semantic Manipulation
## [Pointnet2 Architecture for Waypoint-Defined Primitives]

*Priya Sundaresan, Suneel Belkhale, Dorsa Sadigh, Jeannette Bohg*

[[Project]](http://tinyurl.com/kite-site)
[[arXiv]](https://arxiv.org/abs/2306.16605)

## Description
* KITE is a framework for semantic manipulation using keypoints as a mechanism for grounding language instructions in a visual scene, and a library of keypoint-conditioned skills for execution.
* This repo provides the code for training a keypoint-conditioned skill policy from point cloud input (point cloud + keypoint --> waypoints)
* See [our simulated semantic grasping demo](https://github.com/priyasundaresan/kite_semantic_grasping.git) for an end-to-end example of KITE and [our keypoint training repo](https://github.com/priyasundaresan/kite_keypoint_training.git) to train your own (image + language --> keypoint) model

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

## Installation

Create a conda environment per env.yml (am using `torch=1.9.0+cu102`)

## Usage

We provide an example dataset for opening different drawer cabinets (top/middle/bottom). Given an image, KITE's grounding module outputs a keypoint for the appropriate drawer handle, and we deproject this keypoint onto the 3D point cloud. This annotated point cloud serves as input to a skill policy, which outputs waypoints (gripper position/orientation) for the robot arm to go to in order to grasp and open the cabinet handle. 

To train the model:
```python
python train_start.py
```

To run inference and visualize predictions:
```python
python inference.py
```

This will visualize the ground truth input point cloud (xyz, color, and a mask for the deprojected keypoint), and the predicted waypoint (position / orientation).

For an example of training a skill parameterized by more than one waypoint, see `train_start_end.py`

## Datasets
Data should be organized in the `data/` folder as follows:

```
data/dset_open/
├── test
└── train
```

where `train` is organized as follows:
```
train
├── 00000.npy
├── 00001.npy
├── 00002.npy
...
├── 00047.npy
├── 00048.npy
└── 00049.npy
```
