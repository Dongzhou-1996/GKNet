import argparse
import logging
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from apis import *
from model import *
from tools.commons import set_seed, setup_logging, init_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--T', type=int, default=16, help = 'T of CosineAnnealingLR')
    parser.add_argument('--eta-min', type=float, default=1e-6,help = 'eta_min of CosineAnnealingLR')

    parser.add_argument('--kp-num', type=int, default=16, help='Number of keypoints')
    parser.add_argument('--Gauss-sigma', type=float, default=1, help='Gauss sigma')
    parser.add_argument('--dropput', type=float, default=0.1, help='rate of dropput')
    parser.add_argument('--nhead', type=int, default=8,help='Number of attention heads in model')
    parser.add_argument('--num-layers', type=int, default=3,help='Number of decoder layers in model')
    parser.add_argument('--num-mlp-layers', type=int, default=2,help='Number of mlp layers in model')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for input images')

    parser.add_argument('--categories', type=str, nargs='+', default=['Satellite_04'],help='Categories for training')
    parser.add_argument('--name-model', type=str, required=True,help='Name of the model')
    parser.add_argument('--seed', type=int, default=150,help='Random seed')
    parser.add_argument('--order', type=int, default=1,help='Order of train')

    parser.add_argument('--root-dir', type=str, default= "./", help='Path of data')
    parser.add_argument('--param-path',type=str, help='Parameter path')

    parser.add_argument('--auto-resume', type=bool, default=False, help='Automatically resume from the latest checkpoint')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    graph = False
    work_dir = f'./result/{args.name_model}/{args.order}'
    set_seed(args.seed)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    init_path(work_dir)

    log_dir = os.path.join(work_dir, 'logs', timestamp)
    setup_logging(log_dir)
    logging.info(f'{args.order}')
    logging.info("Training hyperparameters:")
    logging.info(f"Name of model:{args.name_model}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Weight decay: {args.weight_decay}")
    logging.info(f"Keypoints number: {args.kp_num}")
    logging.info(f"Gauss sigma: {args.Gauss_sigma}")
    logging.info(f"Image size: {args.img_size}")
    logging.info(f"Categories: {args.categories}")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f'T of scheduler: {args.T}')

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
        logging.info(f"Heads number: {args.num_heads}")
        logging.info(f"Dropout rate: {args.dropput}")

    elif args.name_model == "HRNet":
        model = HRNet(num_joints=args.kp_num).to(device)

    train_skd = SKD2(root_dir=args.root_dir,
                     subset='train',
                     kp_num=args.kp_num,
                     img_size=args.img_size,
                     categories=args.categories,
                     transformation=True)
    test_skd = SKD2(root_dir=args.root_dir,
                    subset='val',
                    kp_num=args.kp_num,
                    img_size=args.img_size,
                    categories=args.categories,
                    transformation=True)

    train_loader = DataLoader(train_skd,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=16,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_skd,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=16,
                             drop_last=True,
                             pin_memory=True)


    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T, eta_min=args.eta_min)


    if args.auto_resume:
        checkpoint_path = os.path.join(work_dir, 'last.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            epoch = checkpoint['epoch']
            logging.info(f"Successfully loaded model weights, resuming training at epoch {epoch}.")
        else:
            epoch = 0
            logging.info('No checkpoint found, starting a new training session.')
    else:
        epoch = 0
        logging.info('No checkpoint found, starting a new training session.')

    if args.param_path is not None:
        param_path  = args.param_path
        checkpoint = torch.load(param_path, map_location=device)
        pretrained_dict = checkpoint['model']
        model_dict = model.state_dict()
        matching_layers = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matching_layers)
        model.load_state_dict(model_dict)
        logging.info(f"Successfully loaded model weights.")


    for epoch in range(epoch, args.epochs):

        logging.info(f"Starting training for epoch {epoch + 1}...")
        model.train()
        train_one_epoch(train_loader=train_loader,
                        device=device,
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        logging=logging,
                        epoch=epoch,
                        kp_num=args.kp_num,
                        sigma=args.Gauss_sigma,
                        graph = graph)

        logging.info(f"Starting validating for epoch {epoch + 1}...")
        model.eval()
        val_loss = val_one_epoch(test_loader=test_loader,
                                 device=device,
                                 model=model,
                                 criterion=criterion,
                                 logging=logging,
                                 save_dir=work_dir,
                                 epoch=epoch,
                                 sigma=args.Gauss_sigma,
                                 kp_num=args.kp_num,
                                 graph = graph)
        logging.info(f"Validating Loss: {val_loss}")

        save_files = {
            'model': model.state_dict(),
        }
        last_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }
        save_path = os.path.join(work_dir, 'param', f"{epoch + 1}.pth")
        last_path = os.path.join(work_dir, 'last.pth')
        torch.save(save_files, save_path)
        torch.save(last_files, last_path)
        logging.info(f"Save model to {save_path}")
        logging.info('Last params are saved to last.pth')

        scheduler.step()

    logging.info("Training finished.")
    logging.info(f"ALL the logs and param {work_dir}")


if __name__ == '__main__':
    main()

