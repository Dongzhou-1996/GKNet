import numpy as np
import torch
from typing import Tuple



def get_max_preds(batch_heatmaps):

    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps.device)

    preds[:, :, 0] = idx % w
    preds[:, :, 1] = torch.floor(idx / w)

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256 // 8, 256 // 8),
                 gaussian_sigma = None,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = keypoints_weights is not None
        self.kps_weights = keypoints_weights

        kernel_size = 2 * self.kernel_radius + 1
        self.kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                self.kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))

    def __call__(self, images, targets):
        batch_size = images.shape[0]
        num_kps = targets["keypoints"].shape[1]
        heatmap = np.zeros((batch_size, num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)

        for b in range(batch_size):
            kps = targets["keypoints"][b]
            kps_weights = np.ones((num_kps,), dtype=np.float32)
            if "visible" in targets:
                kps_weights = targets["visible"][b]

            heatmap_kps = (kps / 8 + 0.5).int()

            for kp_id in range(num_kps):
                v = kps_weights[kp_id]
                if v < 0.5:
                    continue

                x, y = heatmap_kps[kp_id]
                ul = [x - self.kernel_radius, y - self.kernel_radius]
                br = [x + self.kernel_radius, y + self.kernel_radius]

                if ul[0] > self.heatmap_hw[1] - 1 or ul[1] > self.heatmap_hw[0] - 1 or br[0] < 0 or br[1] < 0:
                    kps_weights[kp_id] = 0
                    continue

                g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
                g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
                img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
                img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

                if kps_weights[kp_id] > 0.5:
                    heatmap[b, kp_id, img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] += self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        targets["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        return images, targets




