import argparse
import time
import logging
from torch.utils.data import DataLoader
from apis import *
from tools.heatmap import *
from model import *
from tools.commons import setup_logging, set_seed
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='Test model')

    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--kp-num', type=int, default=16, help='Number of keypoints')
    parser.add_argument('--Gauss-sigma', type=float, default=1, help='Gauss sigma')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for input images')
    parser.add_argument('--dropput', type=float, default=0.3, help='rate of dropput')
    parser.add_argument('--nhead', type=int, default=8,help='Number of attention heads in model')
    parser.add_argument('--num-layers', type=int, default=3,help='Number of decoder layers in model')
    parser.add_argument('--num-mlp-layers', type=int, default=2,help='Number of mlp layers in model')

    parser.add_argument('--categories', type=str, nargs='+', default=['Satellite_01'],help='Categories for testing')
    parser.add_argument('--name-model', type=str, required=True,help='Name of the model')
    parser.add_argument('--seed', type=int, default=150,help='Random seed')
    parser.add_argument('--order', type=int, default=1,help='Order of train')

    parser.add_argument('--root-dir', type=str, default= "./", help='Path of data')
    parser.add_argument('--param-path',type=str, help='Parameter path')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    graph = False
    work_dir = f'./result/{args.name_model}/test'
    set_seed(args.seed)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    log_dir = os.path.join(work_dir, 'logs', timestamp)
    setup_logging(log_dir)

    logging.info(f'{args.name_model}')
    logging.info("Training hyperparameters:")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Keypoints number: {args.kp_num}")
    logging.info(f"Gauss sigma: {args.Gauss_sigma}")
    logging.info(f"Image size: {args.img_size}")
    logging.info(f"Categories: {args.categories}")
    logging.info(f"Random seed: {args.seed}")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    if args.name_model == "ResUNet":
        model = ResUNet(num_classes= args.kp_num).to(device)

    elif args.name_model == "UNet":
        model = UNet(num_classes= args.kp_num).to(device)

    elif args.name_model == "GKNet":
        model = GKNet(num_classes=args.kp_num,
                        num_layers_mlp=args.num_mlp_layers,
                        num_decoder_layer=args.num_layers,
                        dropout=args.dropput,
                        in_channels=3).to(device)
        graph = True
        logging.info(f"Num mlp layers: {args.num_mlp_layers}")
        logging.info(f"Num decoder layers: {args.num_layers}")
        logging.info(f"Heads number: {args.nhead}")
        logging.info(f"Dropout rate: {args.dropput}")

    elif args.name_model == "HRNet":
        model = HRNet(num_joints=args.kp_num).to(device)

    val_skd = SKD2(root_dir=args.root_dir,
                    subset='test',
                    kp_num=args.kp_num,
                    img_size=args.img_size,
                    categories=args.categories,
                    transformation=True)

    camera_matrix = val_skd.camera_matrix
    val_loader = DataLoader(val_skd,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=16,
                             drop_last=True,
                             pin_memory=True)

    criterion = nn.MSELoss()

    if args.param_path is not None:
        param_path  = args.param_path
        checkpoint = torch.load(param_path, map_location=device)
        pretrained_dict = checkpoint['model']
        model_dict = model.state_dict()
        matching_layers = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matching_layers)
        model.load_state_dict(model_dict)
        logging.info(f"Successfully loaded model weights for index.")

    work_dir = os.path.join(f'./result/{args.name_model}/test', timestamp)
    logging.info(f"Starting testing ...")
    model.eval()
    val_loss = test(test_loader=val_loader,
                    device=device,
                    model=model,
                    criterion=criterion,
                    logging=logging,
                    save_dir=work_dir,
                    sigma=args.Gauss_sigma,
                    kp_num=args.kp_num,
                    img_size=args.img_size,
                    camera=camera_matrix,
                    graph = graph,)
    logging.info(f"Test Loss: {val_loss}")

    logging.info("Testing finished.")



if __name__ == '__main__':
    main()

