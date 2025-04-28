from numpy import sqrt
from .heatmap import *
import torch


def get_loss(bs, kp_num, gt_num, keypoints_2d, sigma, device, image, predicted_keypoints_heatmap, criterion):

    criterion_1 = torch.nn.MSELoss(reduction='none')
    visible = torch.zeros((bs, kp_num), dtype=torch.float32).to(device)
    for i in range(bs):
        visible[i, :gt_num[i]] = 1.0
    target = {
        "keypoints": keypoints_2d,
        "visible": visible
    }
    heatmap_generator = KeypointToHeatMap(heatmap_hw=(256   // 8, 256 // 8), gaussian_sigma=sigma)
    output_image, output_target = heatmap_generator(image, target)
    target_numpy = output_target["heatmap"].numpy()
    target_heatmaps = torch.as_tensor(target_numpy, dtype=torch.float32).to(device)
    loss_hm = criterion_1(predicted_keypoints_heatmap, target_heatmaps).mean(dim=[-2, -1])
    loss_hm = torch.sum(loss_hm) / bs
    rmse_hm = sqrt(loss_hm.item())
    predicted_keypoints, _ = get_max_preds(predicted_keypoints_heatmap)
    loss_kp = 0.0
    for i in range(bs):
        loss_kp += criterion(predicted_keypoints[i, :gt_num[i]] * 8, keypoints_2d[i, :gt_num[i]])

    loss_kp = loss_kp / bs
    rmse_kp = sqrt(loss_kp.item())

    return loss_hm, loss_kp, rmse_hm, rmse_kp, predicted_keypoints