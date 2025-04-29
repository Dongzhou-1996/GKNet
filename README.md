# GKNet

Monocular Pose estimation of noncooperative spacecraft is significant for on-orbit service (OOS) tasks, such as satellite maintenance, space debris removal, and station assembly. Considering the high demands on pose estimation accuracy, mainstream monocular pose estimation methods typically consists of keypoints detector and PnP solver. However, current keypoint detectors remain vulnerable to structural symmetry and partial occlusion of noncooperative spacecrafts. To this end, we propose a graph-based keypoints network for the monocular pose estimation of noncooperative spacecraft, GKNet, which leverages the geometric constraint of keypoints graph. In order to better validate the effectiveness of the proposed method, we present a tiny-scale dataset for the spacecraft keypoints detection, named SKD, which consists of 3 spacecraft targets, 300 simulated image sequences, and 90k precise point-level annotations. Extensive experiment and ablation study have demonstrated the high accuracy and effectiveness of our GKNet, compared to the state-of-the-art spacecraft keypoints detector.

## Getting Started
### Conda Environment
We train and evaluate our model on Python 3.10 and Pytorch 2.3.1 with CUDA 11.8.


### SKD Dataset

Please prepare the SKD dataset for training and evaluation.

You can download the SKD dataset [HERE](https://pan.baidu.com/s/1nQGjsgY6AGTI_V38qAQqrw?pwd=kdu7).

Organize the dataset with the following directory structure:

```bash
├─data
│  ├─000001
│  │  │─image
│  │  │  │─000001.jpg
│  │  │  │─000002.jpg
│  │  │  │─......
│  │  │  └─000300.jpg
│  │  │─annotations.csv
│  │  │─camera_matrix.txt
│  │  │─keypoints.txt
│  │  │─Satellite_01.gif
│  │  │─skeleton.pth
│  │  │─skeleton_pres.pth
│  │  └─trajectory.pdf
│  ├─000002
│  ├─000003
│  ├─......
│  └─000300
├─test.json
├─train.json
├─val.json
└─camera_matrix.txt
```
`skeleton.pth` stores the keypoint edge definitions that are predefined according to geometric relationships. <br>
`skeleton_pres.pth` stores the edges used for displaying keypoints. <br>
`annotations.csv` records the satellite pose annotations.<br>
<br>
You can change the data used for training, validation, and testing by modifying the contents of `test.json`, `train.json`, and `val.json`.

## Train
Training on Satallite01 with GKNet.
```
python train.py --name-model="GKNet" --categories=Satellite_01 --root_dit=/path/to/your/data
```
## Test
Training on Satallite01 with GKNet.
```
python test.py --name-model="GKNet" --categories=Satellite_01 --root_dit=/path/to/your/data --param-pth=/path/to/your/weights
```
