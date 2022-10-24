
# CASNet.PyTorch
This is a network developed under the framework of pytorch and python 3.6;

It is mainly used to segment small samples in road scenes, such as light poles, signboards, etc;


# Usage

We not only tested public data sets such as  S3DIS and PSL, but also applied the network to the data sets collected by the team.

S3DIS: https://github.com/charlesq34/pointnet/blob/master/sem_seg/download_data.sh
PSL: https://cloud.minesparis.psl.eu/index.php/s/JhIxgyt0ALgRZ1O/authenticate
Our datas: Uploading, to be published later.

The network main framework refers to PointNet++. Replace pointnet2_utils.py content in models with the code provided by us to run.
PointNet++：https://github.com/yanx27/Pointnet_Pointnet2_pytorch
```python
python train_semseg.py
```

# License
Our code is released under MIT License (see LICENSE file for details).
