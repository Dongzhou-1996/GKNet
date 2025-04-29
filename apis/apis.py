import os
from PIL import Image
import numpy
from tools.heatmap import *
from tools import get_loss
from tqdm import tqdm
import torch
from SKD2 import SKD2
import cv2

def train_one_epoch(train_loader, device, model, optimizer, criterion, logging, epoch, kp_num, sigma, graph):

    total_train_step = 1
    total_train_loss = 0
    for j, (image, target_pos, target_orient, keypoints_2d, kp_3d, adj ,gt_num , bbox, adj_pres) in tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc='Training Epoch {}, lr {}'.format(epoch + 1, optimizer.param_groups[0]['lr'])):

        gt_num = gt_num.to(device)
        keypoints_2d = keypoints_2d.float().to(device)
        image = image.to(device)

        if graph:
            adj = adj.squeeze(1).to(device)
            predicted_keypoints_heatmap = model(image,adj)
        else:
            predicted_keypoints_heatmap = model(image)
        bs = image.size(0)

        loss_hm, loss_kp, rmse_hm, rmse_kp, pre_kp = get_loss(
                                              bs = bs,
                                              kp_num = kp_num,
                                              image=image,
                                              keypoints_2d = keypoints_2d,
                                              sigma = sigma,
                                              device = device,
                                              criterion = criterion,
                                              gt_num = gt_num,
                                              predicted_keypoints_heatmap = predicted_keypoints_heatmap
                                              ,)

        optimizer.zero_grad()
        loss_hm.backward()
        optimizer.step()

        total_train_loss += rmse_kp.item()

        if total_train_step % 100 == 0:
            logging.info(
                f"Epoch [{epoch + 1}], Step [{total_train_step}], Loss (heatmap): {rmse_hm:.4f}, Loss (keypoints): {rmse_kp:.4f}")

        total_train_step += 1
    logging.info(f"Epoch [{epoch + 1}] Training finished! Total RMSE(): {total_train_loss:.4f}")


def val_one_epoch(test_loader, device, model, criterion, logging, save_dir, epoch, kp_num, sigma, graph):
    total_val_step = 1
    total_val_loss = 0
    with torch.no_grad():
        for k, (image, target_pos, target_orient, keypoints_2d, adj, gt_num) in tqdm(
                enumerate(test_loader), total=len(test_loader), desc='Validating'):
            gt_num = gt_num.to(device)
            keypoints_2d_numpy = keypoints_2d.int()

            image = image.to(device)
            bs = image.size(0)
            keypoints_2d = keypoints_2d.float()
            keypoints_2d = keypoints_2d.to(device)

            if graph:
                adj = adj.squeeze(1).to(device)
                predicted_keypoints_heatmap = model(image, adj)
            else:
                predicted_keypoints_heatmap = model(image)

            predicted_keypoints, _ = get_max_preds(predicted_keypoints_heatmap)

            loss_hm, loss_kp, rmse_hm, rmse_kp, pre_kp = get_loss(bs=bs,
                                                                  kp_num=kp_num,
                                                                  keypoints_2d=keypoints_2d,
                                                                  sigma=sigma,
                                                                  device=device,
                                                                  criterion=criterion,
                                                                  gt_num=gt_num,
                                                                  predicted_keypoints_heatmap=predicted_keypoints_heatmap,
                                                                  image=image, )

            total_val_loss += rmse_kp.item()

            if total_val_step % 50 == 0:
                predicted_keypoints_np = (predicted_keypoints * 8).cpu().numpy()
                for m in range(len(image)):
                    img = numpy.array(image[m].cpu())
                    img = np.transpose(img, (1, 2, 0))
                    img = np.clip(img * 255., 0., 255.)
                    img = img.astype(np.uint8)

                    tuple_list = [tuple(int(b) for b in keypoints_2d_numpy[m][n]) for n in range(min(gt_num[m].item(), len(keypoints_2d_numpy[m])))]

                    img_unpre = img.copy()
                    for keypoint in tuple_list:
                        img_unpre = cv2.circle(img_unpre, center=keypoint, radius=2, color=(255, 0, 255), thickness=-1)

                    img_pre = img.copy()
                    tuple_list_pred = [tuple(int(b) for b in predicted_keypoints_np[m][n]) for n in
                                       range(min(gt_num[m].item(), len(predicted_keypoints_np[m])))]
                    for keypoint in tuple_list_pred:
                        img_pre = cv2.circle(img_pre, center=keypoint, radius=2, color=(255, 0, 255), thickness=-1)

                    img_cat = np.concatenate((img_unpre, img_pre), axis=1)
                    img_concate = Image.fromarray(img_cat)
                    mse_kp = criterion(predicted_keypoints[m, :gt_num[m]] * 8, keypoints_2d[m, :gt_num[m]])

                    rmse_per_image = np.sqrt(mse_kp.item())

                    concate_path = os.path.join('../', save_dir, 'test',
                                                f'epoch_{epoch+1}_out{k},{m + 1}_rmse{rmse_per_image:.4f}.jpg')
                    img_concate.save(concate_path)

            if total_val_step % 100 == 0:
                logging.info(
                    f"Epoch [{epoch + 1}], Step [{total_val_step}], Loss (heatmap): {rmse_hm:.4f}, Loss (keypoints): {rmse_kp:.4f}")

            total_val_step += 1
    logging.info(f"Epoch [{epoch + 1}] Valing finished! Total RMSE: {total_val_loss:.4f}")
    return total_val_loss

def test(test_loader, device, model, criterion, logging, save_dir, kp_num, sigma, img_size, camera, graph):

    total_test_step = 1
    total_test_loss = 0
    total_position_error = 0
    total_orientation_error = 0
    test_folder = save_dir
    len_test_loader = len(test_loader)
    len_test_dataset = len(test_loader.dataset)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    with torch.no_grad():
        for k, (image, target_pos, target_orient, keypoints_2d, kp_3d, adj ,gt_num , bbox, adj_pres) in tqdm(
                    enumerate(test_loader), total=len(test_loader), desc='Testing'):

                gt_num = gt_num.to(device)
                adj_pres = adj_pres.to(device)
                keypoints_2d_numpy = keypoints_2d.int()
                target_pos, target_orient = target_pos.cpu().numpy(), target_orient.cpu().numpy()
                image = image.to(device)
                bs = image.size(0)
                keypoints_2d = keypoints_2d.float()
                keypoints_2d = keypoints_2d.to(device)

                if graph:
                    adj = adj.squeeze(1).to(device)
                    predicted_keypoints_heatmap = model(image, adj)
                else:
                    predicted_keypoints_heatmap = model(image)

                predicted_keypoints, _ = get_max_preds(predicted_keypoints_heatmap)

                loss_hm, loss_kp, rmse_hm, rmse_kp, pre_kp = get_loss(
                                                            bs=bs,
                                                            kp_num=kp_num,
                                                            keypoints_2d=keypoints_2d,
                                                            sigma=sigma,
                                                            device=device,
                                                            criterion=criterion,
                                                            gt_num=gt_num,
                                                            predicted_keypoints_heatmap=predicted_keypoints_heatmap,
                                                            image=image,)

                total_test_loss += rmse_kp.item()

                predicted_keypoints_np = (predicted_keypoints * 8).cpu().numpy()
                for m in range(len(image)):
                    img = numpy.array(image[m].cpu())
                    img = np.transpose(img, (1, 2, 0))
                    img = np.clip(img * 255., 0., 255.)
                    img = img.astype(np.uint8)
                    tuple_list = [tuple(int(b) for b in keypoints_2d_numpy[m][n]) for n in
                                  range(min(gt_num[m].item(), len(keypoints_2d_numpy[m])))]

                    img_unpre = img.copy()
                    for keypoint in tuple_list:
                        img_unpre = cv2.circle(img_unpre, center=keypoint, radius=2, color=(255, 0, 255), thickness=-2)

                    img_pre = img.copy()
                    tuple_list_pred = [tuple(int(b) for b in predicted_keypoints_np[m][n]) for n in
                                       range(min(gt_num[m].item(), len(predicted_keypoints_np[m])))]
                    for keypoint in tuple_list_pred:
                        img_pre = cv2.circle(img_pre, center=keypoint, radius=2, color=(255, 0, 255), thickness=-2)

                    num_keypoints = gt_num[m].item()
                    for i in range(num_keypoints):
                        for j in range(i + 1, num_keypoints):
                            if adj_pres[m][i][j] != 0:

                                start_point = tuple(int(x) for x in predicted_keypoints_np[m][i])
                                end_point = tuple(int(x) for x in predicted_keypoints_np[m][j])
                                img_pre = cv2.line(img_pre, start_point, end_point, (255, 0, 255), 1)

                                start_point_unpre = tuple(int(x) for x in keypoints_2d_numpy[m][i])
                                end_point_unpre = tuple(int(x) for x in keypoints_2d_numpy[m][j])
                                img_unpre = cv2.line(img_unpre, start_point_unpre, end_point_unpre, (255, 0, 255), 1)

                    img_cat = np.concatenate((img_unpre, img_pre), axis=1)
                    img_concate = Image.fromarray(img_cat)

                    # caculate loss of every picture
                    mse_kp = criterion(predicted_keypoints[m, :gt_num[m]] * 8, keypoints_2d[m, :gt_num[m]])
                    rmse_per_image = np.sqrt(mse_kp.item())

                    # caculate pos err and orient err
                    bbox = [torch.tensor(b) if not isinstance(b, torch.Tensor) else b for b in bbox]

                    bbox = torch.stack(bbox)
                    bbox = bbox.view(-1, 4)
                    bbox = bbox.numpy()
                    kp_2d_origin = SKD2.inverse_transform(predicted_keypoints_np[m, :gt_num[m]],bbox[m],img_size)
                    orient_pre, pos_pre = SKD2.PnP(kp_2d_origin, kp_3d[m, :gt_num[m]], camera)

                    err, err_pos, err_orient = SKD2.metric(pos_pre, orient_pre,target_pos[m],target_orient[m])
                    total_position_error += err_pos
                    total_orientation_error += err_orient

                    # save picture
                    concate_path = os.path.join(test_folder,
                                                f'out{k},{m + 1}_rmse{rmse_per_image:.4f}_err{err}_pos{err_pos}_orient_{err_orient}.jpg')
                    img_concate.save(concate_path)


                if total_test_step % 100 == 0:
                    logging.info(f"Step [{total_test_step}], Loss (heatmap): {rmse_hm:.4f}, Loss (keypoints): {rmse_kp:.4f}")

                total_test_step += 1

        avg_position_error = total_position_error / len_test_dataset
        avg_orientation_error = total_orientation_error / len_test_dataset

        logging.info(f"Testing finished! Total RMSE: {total_test_loss:.4f},Average RMSE: {total_test_loss/len_test_loader:.4f} "
                    f"Average Position Error: {avg_position_error:.4f}, "
                    f"Average Orientation Error: {avg_orientation_error:.4f}")
        return total_test_loss