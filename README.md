# Pointnet2 Architecture for Waypoint-Defined Primitives

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

## Installation

Create a conda environment per env.yml (am using `torch=1.9.0+cu102`)

## Usage

To train a model that outputs a single start waypoint (position/orientation)

```python
python train_start.py
```

To train a model that outputs a start/end waypoint (position/orientation)
```python
python train_start_end.py
```

`inference_cls_off_rot.py` should have an example of how to get a predicted waypoint from an input point cloud (though this code might be semi-broken)

## Datasets

Data should be organized in the `data/` folder as follows:

```
data/dset_toolcabinet/
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
