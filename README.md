# SRFDet3D: Sparse Region Fusion based 3D Object Detection

This is the official PyTorch implementation of the paper **SRFDet3D: Sparse Region Fusion based 3D Object Detection**, by Gopi Krishna Erabati and Helder Araujo.

**Contents**
1. [Overview](https://github.com/gopi-erabati/SRFDet3D#overview)
2. [Results](https://github.com/gopi-erabati/SRFDet3D/tree/master#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/SRFDet3D/tree/master#requirements-installation-and-usage)
    1. [Prerequistes](https://github.com/gopi-erabati/SRFDet3D/tree/master#prerequisites)
    2. [Installation](https://github.com/gopi-erabati/SRFDet3D/tree/master#installation)
    3. [Training](https://github.com/gopi-erabati/SRFDet3D/tree/master#training)
    4. [Testing](https://github.com/gopi-erabati/SRFDet3D/tree/master#testing)
4. [Acknowledgements](https://github.com/gopi-erabati/SRFDet3D/tree/master#acknowlegements)

## Overview

![SRFDet3D](https://github.com/gopi-erabati/SRFDet/assets/22390149/39994836-0355-4423-adb6-5c328548b2e4)

## Results

### Predictions on nuScenes dataset
![SRFDet3D_ Sparse Region Fusion based 3D Object Detection _ 3D Object Detection _ Autonomous Driving](https://github.com/gopi-erabati/SRFDet/assets/22390149/499732fb-7968-4bcd-ba43-b18cf7ce81f5)

### nuScenes dataset
| Config | mAP | NDS | |
| :---: | :---: |:---: |:---: |
| [srfdet_voxel_nusc_L](configs/nus/srfdet_voxel_nusc_L.py) | 63.1 | 68.5 | [weights](https://drive.google.com/file/d/1d9g7kbtCceGsvmh1iX3BtkJyIZPzdkV5/view?usp=sharing) |
| [srfdet_voxel_nusc_LC](configs/nus/srfdet_voxel_nusc_LC.py) | 64.7 | 68.6 | [weights](https://drive.google.com/file/d/1kD-nz7dYpg804YF1Qmsrz3jlSz13f_qU/view?usp=sharing) |

### KITTI dataset
| Config | Car easy | Car mod. | Car hard | Ped. easy | Ped. mod. | Ped. hard | Cyc. easy | Cyc. mod. | Cyc. hard | |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---: | :---:  | :---:  | :---:  |
| [srfdet_voxel_kitti_L](configs/kitti/srfdet_voxel_kitti_L.py) | 94.2 | 89.3 | 85.6 | 57.9 | 52.1 | 47.9 | 81.5 | 63.6 | 58.9 | [weights](https://drive.google.com/file/d/1Gbl_LBL0o367jmGLXkiLlYg-Uy4Qxvnh/view?usp=sharing) |
| [srfdet_voxel_kitti_LC](configs/kitti/srfdet_voxel_kitti_LC.py) | 94.9 | 87.8 | 84.9 | 58.1 | 55.5 | 49.7 | 82.3 | 63.7 | 59.8 | [weights](https://drive.google.com/file/d/1S3tDxPWJ9Ic_-NPUFuvf8AfeafIioAOA/view?usp=sharing) |

### Waymo dataset (mAPH)
| Config | Veh. L1 | Veh. L2 | Ped. L1  | Ped. L2  | Cyc. L1 | Cyc. L2 |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| [srfdet_dvoxel_waymo_L](configs/waymo/srfdet_dvoxel_waymo_L.py) | 71.9 | 63.6 | 64.9 | 57.0 | 66.2 | 63.7 |

We can not distribute the model weights on Waymo dataset due to the [Waymo license terms](https://waymo.com/open/terms).

## Requirements, Installation and Usage

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==1.13.1
- [mmcv](https://github.com/open-mmlab/mmcv)==1.7.0
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.28.2
- [mmseg](https://github.com/open-mmlab/mmsegmentation)==0.30.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.0.0.rc6

### Installation
```
mkvirtualenv srfdet3d

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmcv-full==1.7.0

pip install -r requirements.txt
```
For evaluation on Waymo, please follow the below code to build the binary file `compute_detection_metrics_main` for metrics computation and put it into ```mmdet3d_plugin/core/evaluation/waymo_utils/```.
```
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../SRFDet3D/mmdet3d_plugin/core/evaluation/waymo_utils/
```

### Data
Follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) to prepare the [nuScenes](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html), [Waymo](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) and [KITTI](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html) datasets and symlink the data directories to `data/` folder of this repository.
**Warning:** Please strictly follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Clone the repository
```
git clone https://github.com/gopi-erabati/SRFDet3D.git
cd SRFDet3D
```

### Training

#### nuScenes dataset
- Download the [backbone pretrained weights](https://drive.google.com/drive/folders/1FMigS3eA4gdd-g9c46nc5qtvEicgdAfA?usp=sharing) to `ckpts/`
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/nus/srfdet_voxel_nusc_L.py --work-dir {WORK_DIR}` for LiDAR-only model and `python tools/train.py configs/kitti/srfdet_voxel_nusc_LC.py --work-dir {WORK_DIR} --cfg-options load_from=/path/to/lidar-only/model` for LiDAR-Camera fusion model.
- Multi GPU training
  `tools/dist_train.sh configs/nus/srfdet_voxel_nusc_L.py {GPU_NUM} --work-dir {WORK_DIR}` for LiDAR-only model and `tools/dist_train.sh configs/nus/srfdet_voxel_nusc_LC.py {GPU_NUM} --work-dir {WORK_DIR} --cfg-options load_from=/path/to/lidar-only/model` for LiDAR-Camera fusion model.
 
#### KITTI dataset
- Download the [backbone pretrained weights](https://drive.google.com/drive/folders/1FMigS3eA4gdd-g9c46nc5qtvEicgdAfA?usp=sharing) to `ckpts/` 
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/kitti/srfdet_voxel_kitti_L.py --work-dir {WORK_DIR}` for LiDAR-only model and `python tools/train.py configs/kitti/srfdet_voxel_kitti_LC.py --work-dir {WORK_DIR} --cfg-options load_from=/path/to/lidar-only/model` for LiDAR-Camera fusion model.
- Multi GPU training
  `tools/dist_train.sh configs/kitti/srfdet_voxel_kitti_L.py {GPU_NUM} --work-dir {WORK_DIR}` for LiDAR-only model and `tools/dist_train.sh configs/kitti/srfdet_voxel_kitti_LC.py {GPU_NUM} --work-dir {WORK_DIR} --cfg-options load_from=/path/to/lidar-only/model` for LiDAR-Camera fusion model.

#### Waymo dataset 
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/waymo/srfdet_dvoxel_waymo_L.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/waymo/srfdet_dvoxel_waymo_L.py {GPU_NUM} --work-dir {WORK_DIR}`

### Testing

#### nuScenes dataset
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/nus/srfdet_voxel_nusc_L.py /path/to/ckpt --eval mAP` for LiDAR-only model and `python tools/test.py configs/nus/srfdet_voxel_nusc_LC.py /path/to/ckpt --eval mAP` for LiDAR-Camera fusion model.
- Multi GPU training
  `tools/dist_test.sh configs/nus/srfdet_voxel_nusc_L.py /path/to/ckpt {GPU_NUM} --eval mAP` for LiDAR-only model and `tools/dist_test.sh configs/nus/srfdet_voxel_nusc_LC.py /path/to/ckpt {GPU_NUM} --eval mAP` for LiDAR-Camera fusion model.
 
#### KITTI dataset
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/kitti/srfdet_voxel_kitti_L.py /path/to/ckpt --eval mAP` for LiDAR-only model and `python tools/test.py configs/kitti/srfdet_voxel_kitti_LC.py /path/to/ckpt --eval mAP` for LiDAR-Camera fusion model.
- Multi GPU training
  `tools/dist_test.sh configs/kitti/srfdet_voxel_kitti_L.py /path/to/ckpt {GPU_NUM} --eval mAP` for LiDAR-only model and `tools/dist_test.sh configs/kitti/srfdet_voxel_kitti_LC.py /path/to/ckpt {GPU_NUM} --eval mAP` for LiDAR-Camera fusion model.

#### Waymo dataset 
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/waymo/srfdet_dvoxel_waymo_L.py /path/to/ckpt --eval waymo`
  `tools/dist_test.sh configs/waymo/srfdet_dvoxel_waymo_L.py /path/to/ckpt {GPU_NUM} --eval waymo`

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
