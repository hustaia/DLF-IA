from utils.regression_trainer import RegTrainer
import argparse
import os
import torch
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='directory to save models')
    parser.add_argument('--dataset', default='', choices=['RGBTCC', 'DroneRGBT'],
                        help='dataset name')
    parser.add_argument('--backbone', default='swin_b', choices=['swin_b'],
                        help='backbone name')
    parser.add_argument('--pretrained-model', default="./swin_base_patch4_window12_384_22k.pth",
                        help='the path of pretrained model')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--device', default='0', help='assign device')

    # default
    parser.add_argument('--bn-eps', type=float, default=1e-3,
                        help='batch normalization epsilon')
    parser.add_argument('--bn-momentum', type=float, default=0.1,
                        help='batch normalization momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')
    parser.add_argument('--save-all-best', type=bool, default=True,
                        help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='default 256')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    parser.add_argument('--w-normalize', type=float, default=0.1,
                        help='weight of normalization term in loss function')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
