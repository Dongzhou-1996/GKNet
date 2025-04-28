import time
import json
import numpy as np
import cv2
import os
import pandas as pd
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class SKD2(Dataset):
    def __init__(self, root_dir='SPE', subset='train', categories: list = None, kp_num=8, img_size=512,nshot = 10,
                 transformation=True):
        super(SKD2, self).__init__()
        if not os.path.exists(root_dir):
            raise FileNotFoundError('Root directory does not exist!')
        self.root_dir = root_dir
        assert subset in ['train', 'val', 'test'], 'Invalid subset!'
        self.subset = subset
        self.data_dir = os.path.join(self.root_dir, 'data')
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError('Data directory does not exist!')
        if len(os.listdir(self.data_dir)) < 1:
            raise FileNotFoundError('No sequence in data directory!')

        self.subset_sequences_file = os.path.join(self.root_dir, '{}.json'.format(subset))
        self.sequences = []
        if not os.path.exists(self.subset_sequences_file):
            raise FileNotFoundError('subset sequences file does not exist!')
        else:
            with open(self.subset_sequences_file, 'r') as f:
                self.subset_sequences = json.load(f)

            if categories is None or categories == []:
                categories = self.subset_sequences.keys()
            for category in categories:
                assert category in self.subset_sequences, 'Category {} is not in {} subset!'.format(category,
                                                                                                    self.subset)
                self.sequences += self.subset_sequences[category]

        self.sequences = [os.path.join(self.root_dir, s) for s in self.sequences]
        self.nshot = nshot
        self.org_img_size = 1024
        self.seq_len = 300
        self.kp_num = kp_num
        self.img_size = img_size
        self.transform_mat = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        self.transformation = transformation

        intrinsic_matrix_file = os.path.join(root_dir, 'camera_matrix.txt')
        self.camera_matrix = np.loadtxt(intrinsic_matrix_file, delimiter=',', dtype=np.float32)
        self.camera_matrix = self.camera_matrix @ self.transform_mat

    def __len__(self):
        return len(self.sequences) * 300

    def __getitem__(self, idx):
        seq_idx = idx // self.seq_len
        frame_idx = idx % self.seq_len
        seq_dir = self.sequences[seq_idx]

        annotation_file = os.path.join(seq_dir, 'annotations.csv')
        annos = pd.read_csv(annotation_file, index_col=None)
        # category = annos['category'].iloc[frame_idx]
        pos = eval(annos['position'].iloc[frame_idx])
        orient = eval(annos['orientation'].iloc[frame_idx])
        img_file = os.path.join(seq_dir, 'image', '{:06d}.jpg'.format(frame_idx + 1))
        img = Image.open(img_file)

        skeleton_file = os.path.join(seq_dir, 'skeleton.pth')
        skeleton = torch.load(skeleton_file,weights_only=True)

        skeleton_pres_file = os.path.join(seq_dir, 'skeleton_pres.pth')
        skeleton_pres = torch.load(skeleton_pres_file,weights_only=True)

        # transpose
        orient = self.transform_mat.T @ self.quaternion2matrix(orient) @ self.transform_mat
        orient = self.matrix2quaternion(orient)
        pos = pos @ self.transform_mat

        keypoints_file = os.path.join(seq_dir, 'keypoints.txt')
        kp_body = np.loadtxt(keypoints_file, delimiter=',')
        kp_body = kp_body @ self.transform_mat
        _, kp_2d = self.keypoints(None, kp_body, pos, orient, self.camera_matrix)
        # guarantee the keypoints is in-view
        kp_2d = [kp for kp in kp_2d if (0 <= kp[0] < self.org_img_size) and (0 <= kp[1] < self.org_img_size)]
        kp_2d = np.asarray(kp_2d)
        if self.transformation:
            img, kp_2d, bbox = self.transform(img, self.img_size, kp_2d)

        gt_kp_num = kp_2d.shape[0]

        # adjust
        if kp_2d.shape[0] < self.kp_num:
            for i in range(kp_2d.shape[0], self.kp_num):
                for element in skeleton:
                    (a, b) = element
                    if a == 0:
                        extra_element = (i, b)
                        skeleton.append(extra_element)
                    elif b == 0:
                        extra_element = (i, a)
                        skeleton.append(extra_element)

            num_to_add = self.kp_num - kp_2d.shape[0]
            arr = np.arange(0, kp_2d.shape[0])
            sam_idx1 = np.full(num_to_add, 0)
            sample_idx = np.append(arr, sam_idx1)
            kp_2d = kp_2d[sample_idx]

        else:
            newskeleton = []
            for element in skeleton:
                (a, b) = element
                if a < self.kp_num and b < self.kp_num:
                    newskeleton.append(element)
            skeleton = newskeleton

            kp_2d = kp_2d[:self.kp_num]

        adj = self.skeleton2adj(skeleton)
        adj_pres = self.skeleton2adj(skeleton_pres)

        if kp_body.shape[0] < self.kp_num:

            num_to_add = self.kp_num - kp_body.shape[0]
            arr = np.arange(0, kp_body.shape[0])
            sam_idx1 = np.full(num_to_add, 0)
            sample_idx = np.append(arr, sam_idx1)

            kp_body = kp_body[sample_idx]
        else:

            kp_body = kp_body[:self.kp_num]

        return img, np.asarray(pos), np.asarray(orient), kp_2d, kp_body, adj, gt_kp_num, bbox, adj_pres

    def transform(self, img: Image.Image, out_size: int, kp_2d: np.ndarray) -> (torch.Tensor, np.ndarray):
        img = F.pil_to_tensor(img)
        bbox = self.get_bbox(kp_2d)
        img = F.crop(img, bbox[0], bbox[1], bbox[2], bbox[3])
        kp_2d = kp_2d - np.array([bbox[1], bbox[0]])

        roi_size = max(bbox[2], bbox[3])
        padding_top = (roi_size - bbox[2]) // 2
        padding_bottom = (roi_size - bbox[2] + 1) // 2
        padding_left = (roi_size - bbox[3]) // 2
        padding_right = (roi_size - bbox[3] + 1) // 2
        img = F.pad(img, [padding_left, padding_top, padding_right, padding_bottom])
        kp_2d = kp_2d + np.array([padding_left, padding_top])
        img = F.resize(img, [out_size, out_size]) / 255.0
        kp_2d = np.int32(kp_2d * (out_size / roi_size))
        return img.float(), kp_2d, bbox
        # return img.float(), kp_2d, (bbox[1], bbox[0])

    @staticmethod
    def get_bbox(kp_2d: np.ndarray, delta: int = 10):
        x0 = np.min(kp_2d[:, 0]) - delta
        y0 = np.min(kp_2d[:, 1]) - delta
        x1 = np.max(kp_2d[:, 0]) + delta
        y1 = np.max(kp_2d[:, 1]) + delta
        w = x1 - x0
        h = y1 - y0
        return y0, x0, h, w

    def keypoints(
            self,
            img: None,
            keypoints_body: np.ndarray,
            obj_pos_cam: np.ndarray,
            obj_orientation_cam: np.ndarray,
            camera_matrix: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        Project target keypoints in body frame to camera frame and pixel frame
        :param img: the input image
        :param keypoints_body: keypoints coordinates in body frame
        :param obj_pos_cam: the position of the target in camera frame
        :param obj_orientation_cam: the orientation of the target in camera frame
        :param camera_matrix: the intrinsic matrix of the camera
        :return: keypoints_cam, keypoints_2d
        """

        r_b2c = self.quaternion2matrix(obj_orientation_cam)
        keypoints_camera = keypoints_body @ r_b2c.transpose() + obj_pos_cam
        keypoints_pixel = keypoints_camera @ np.transpose(camera_matrix)
        keypoints_uv = np.vstack([
            keypoints_pixel[:, 0] / keypoints_pixel[:, -1], keypoints_pixel[:, 1] / keypoints_pixel[:, -1]]
        ).transpose().astype(np.int32)  # Nx2

        # remove out-view keypoints
        def condition(row):
            return 0 <= row[0] <= self.org_img_size >= row[1] >= 0

        mask = np.apply_along_axis(condition, axis=1, arr=keypoints_uv)
        keypoints_uv = keypoints_uv[mask]

        if img is None:
            return keypoints_camera, keypoints_uv
        else:
            for keypoint in keypoints_uv:
                cv2.circle(img, center=tuple(keypoint), radius=3, color=(255, 0, 255), thickness=-1)

            cv2.namedWindow('keypoints', cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('keypoints', img)
            return keypoints_camera, keypoints_uv

    @staticmethod
    def draw_keypoints(img, kp_2d):
        for keypoint in kp_2d:
            img = cv2.circle(img, center=tuple(keypoint), radius=2, color=(255, 0, 255), thickness=-1)

        cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
        cv2.imshow('keypoints', img)
        # cv2.waitKey()
        return

    def show_pose(
            self,
            img,
            obj_pos_cam: np.ndarray,
            obj_orientation_cam: np.ndarray
    ) -> cv2.UMat:
        """
        Show the pose of the target

        :param img: the input image
        :param obj_pos_cam: the position of the target in camera frame
        :param obj_orientation_cam: the orientation of the target in camera frame
        :return: img, the image with pose annotation
        """
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * 3
        r_b2c = self.quaternion2matrix(obj_orientation_cam)
        points_camera = points @ r_b2c.transpose() + obj_pos_cam
        points_pixel = points_camera @ np.transpose(self.camera_matrix)
        points_uv = np.vstack([
            points_pixel[:, 0] / points_pixel[:, -1], points_pixel[:, 1] / points_pixel[:, -1]]
        ).transpose().astype(np.int32)

        # draw original
        img = cv2.circle(img, center=tuple(points_uv[0]), radius=3, color=(255, 0, 255), thickness=-1)
        # draw axes
        img = cv2.arrowedLine(img, tuple(points_uv[0]), tuple(points_uv[1]), color=(0, 0, 255), thickness=2,
                              tipLength=0.1,
                              line_type=cv2.LINE_AA)  # x-axis
        img = cv2.putText(img, 'X', tuple(points_uv[1] + np.array([5, 15])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                          2)
        img = cv2.arrowedLine(img, tuple(points_uv[0]), tuple(points_uv[2]), color=(0, 255, 0), thickness=2,
                              tipLength=0.1,
                              line_type=cv2.LINE_AA)  # y-axis
        img = cv2.putText(img, 'Y', tuple(points_uv[2] + np.array([5, 15])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                          2)
        img = cv2.arrowedLine(img, tuple(points_uv[0]), tuple(points_uv[3]), color=(255, 0, 0), thickness=2,
                              tipLength=0.1,
                              line_type=cv2.LINE_AA)  # z-axis
        img = cv2.putText(img, 'Z', tuple(points_uv[3] + np.array([5, 15])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                          2)
        cv2.namedWindow('pose', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('pose', img)
        return img

    @staticmethod
    def skeleton2adj(skeleton):
        num_nodes = max(max(pair) for pair in skeleton) + 1
        adj = torch.zeros((num_nodes, num_nodes))
        for (i, j) in skeleton:
            adj[i, j] = 1
            adj[j, i] = 1

        adj_with_self_loops = adj + torch.eye(num_nodes)
        degree = torch.sum(adj_with_self_loops, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(degree_inv_sqrt, adj_with_self_loops), degree_inv_sqrt)

        return adj_normalized
    @staticmethod
    def metric(
            pos_p: np.ndarray,
            orient_p: np.ndarray,
            pos_gt: np.ndarray,
            orient_gt: np.ndarray
    ) -> (float, float, float):
        """
        Calculate the pose estimation metric
        :param pos_p: the predicted position of the target, format Nx3
        :param orient_p: the predicted orientation of the target in quaternion, format Nx4
        :param pos_gt: the ground truth position of the target, format Nx3
        :param orient_gt: the ground truth orientation of the target, format Nx4
        :return:
        err, the pose estimation metric.
        pos_err: the average position error
        orient_err: the average orientation error
        """
        pos_p = pos_p.reshape(-1, 3)
        orient_p = orient_p.reshape(-1, 4)
        pos_gt = pos_gt.reshape(-1, 3)
        orient_gt = orient_gt.reshape(-1, 4)
        pos_error = np.linalg.norm((pos_p - pos_gt), 2, axis=1) / np.linalg.norm(pos_gt, 2, axis=1)
        orient_error = 2 * np.arccos(np.clip(np.einsum('ij,ij->i', orient_p, orient_gt) / (
                np.linalg.norm(orient_gt, 2, axis=1) * np.linalg.norm(orient_p, 2, axis=1)), -1, 1))
        err = np.mean(pos_error + orient_error)
        return err, np.mean(pos_error), np.mean(orient_error)

    @staticmethod
    def quaternion2matrix(q: np.ndarray) -> np.ndarray:
        """
        calculate rotation matrix with quaternion
        :param q: [x, y, z, w] format
        :return: rotation matrix, 3x3 dimension
        """
        x, y, z, w = q
        rotation_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * x * x - 2 * y * y]
        ])
        return rotation_matrix

    @staticmethod
    def euler2matrix(euler_vector, degrees=False):
        # euler vector format: [alpha, beta, gamma]
        if degrees:
            alpha, beta, gamma = np.radians(euler_vector)
        else:
            alpha, beta, gamma = euler_vector
        r_x = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]])
        r_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]])
        r_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]])
        rotation_matrix = np.matmul(r_x, np.matmul(r_y, r_z))
        return rotation_matrix

    @staticmethod
    def quaternion2euler(q: np.ndarray, degrees=False):
        """
        transfer quaternion to euler vector
        :param q:
        :param degrees:
        :return:
        """
        x, y, z, w = q

        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4)

        if degrees:
            return np.degrees([roll, pitch, yaw])
        else:
            return np.array([roll, pitch, yaw])

    @staticmethod
    def euler2quaternion(euler_vector, degrees=False):
        if degrees:
            roll, pitch, yaw = np.radians(euler_vector)
        else:
            roll, pitch, yaw = euler_vector

        t0 = math.cos(roll * 0.5)
        t1 = math.sin(roll * 0.5)
        t2 = math.cos(pitch * 0.5)
        t3 = math.sin(pitch * 0.5)
        t4 = math.cos(yaw * 0.5)
        t5 = math.sin(yaw * 0.5)

        w = t0 * t2 * t4 + t1 * t3 * t5
        x = t1 * t2 * t4 - t0 * t3 * t5
        y = t0 * t3 * t4 + t1 * t2 * t5
        z = t0 * t2 * t5 - t1 * t3 * t4

        return np.array([x, y, z, w])

    @staticmethod
    def rvec2euler(r_vec: np.ndarray, degrees=False):
        theta = np.linalg.norm(r_vec)
        if theta < 1e-6:
            return np.array([0, 0, 0])
        r = r_vec / theta  # 旋转轴的单位向量
        roll = np.arctan2(r[1], r[0])
        pitch = np.arctan2(-r[2], np.sqrt(r[0] ** 2 + r[1] ** 2))
        yaw = np.arctan2(r[1], r[0])
        if degrees:
            return np.degrees([roll, pitch, yaw])
        else:
            return np.array([roll, pitch, yaw])

    @staticmethod
    def matrix2quaternion(rotation_matrix) -> np.ndarray:
        q_tc_w = 0.5 * np.sqrt(max(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2], 1e-5))
        q_tc_x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * q_tc_w)
        q_tc_y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * q_tc_w)
        q_tc_z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * q_tc_w)

        q = np.array([q_tc_x, q_tc_y, q_tc_z, q_tc_w])
        return q

    @staticmethod
    def inverse_transform(kp, bbox, outsize):
        roi_size = max(bbox[2], bbox[3])
        padding_top = (roi_size - bbox[2]) // 2
        padding_left = (roi_size - bbox[3]) // 2
        kp = kp * (roi_size / outsize)
        kp = kp - np.array([padding_left, padding_top])
        kp = kp + np.array([bbox[1], bbox[0]])
        kp = np.int32(kp)
        return kp

    @staticmethod
    def PnP(kp_2d, kp_3d, camera_matrix):

        kp_2d = np.array(kp_2d, dtype=np.float32)
        kp_3d = np.array(kp_3d, dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(kp_3d, kp_2d, camera_matrix, distCoeffs=None)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quaternion = SKD2.matrix2quaternion(rotation_matrix)
        return quaternion, tvec


    def category_sequence_idx_generate(self):
        import json
        category_sequence_file = os.path.join(self.root_dir, 'category_sequences.json')
        if os.path.exists(category_sequence_file):
            with open(category_sequence_file, 'r') as f:
                category_sequences = json.load(f)
        else:
            category_sequences = {}
        subset_category_seqs = {}
        bar = tqdm(self.sequences, desc='{}'.format(self.subset))
        for s, seq_dir in enumerate(bar):
            annotation_file = os.path.join(seq_dir, 'annotations.csv')
            annos = pd.read_csv(annotation_file, index_col=None)
            category = annos['category'].iloc[1]
            seq_dir = seq_dir.replace('\\', '/')
            if category in subset_category_seqs.keys():
                subset_category_seqs[category].append(seq_dir[2:])
            else:
                subset_category_seqs.update({category: [seq_dir[2:]]})

        category_sequences.update({self.subset: subset_category_seqs})
        with open(category_sequence_file, 'w') as f:
            json.dump(category_sequences, f, indent=4)


