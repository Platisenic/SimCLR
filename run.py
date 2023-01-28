import argparse
import logging
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from data_ecg_aug.dataset import TVGHDataset, randomCrop
from models.mayo_simclr import *
from models.xresnet_simclr import *
from simclr import SimCLR
from utils import save_config_file

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--ecgdir', metavar='DIR', default='/ecgdata/libos/filtered_data/',
                    help='path to dataset')
parser.add_argument('--anno', default='/ecgdata/libos/SimCLR/anno/40_9pa_rr/all.csv',
                    help='dataset name')
parser.add_argument('--is3lead', action='store_true',
                    help='Default: 12-lead ECG')
parser.add_argument('--isRaligned', action='store_true',
                    help='Default ECG signal not aligned')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-epochs', default=20, type=int,
                    help='Log every n epochs')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    writer = SummaryWriter(comment='-pretrained')
    save_config_file(writer.log_dir, args.__dict__)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers= [
            logging.FileHandler(os.path.join(writer.log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    train_dataset = TVGHDataset(args.anno, args.ecgdir, is3lead=args.is3lead, isRaligned=args.isRaligned, transform=randomCrop)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = xresnet18SimCLR(out_dim=args.out_dim)
    logging.info(model)
    logging.info(f'Total Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args, writer=writer)
        simclr.train(train_loader)

    writer.close()


if __name__ == "__main__":
    main()
