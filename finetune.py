import os
import argparse
from models.mayo_simclr import *
from models.xresnet_simclr import *
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import logging
from data_ecg_aug.dataset import TVGHDataset
from tqdm import tqdm
from utils import save_config_file, save_checkpoint, t_SNE_analysis, my_confusion_matrix, cal_metrics
from sklearn.utils.class_weight import compute_class_weight

parser = argparse.ArgumentParser(description='PyTorch LVSD Detection')
parser.add_argument('--ecgdir', metavar='DIR', default='/ecgdata/libos/filtered_data/',
                    help='path to dataset')
parser.add_argument('--num-classes', default=2,
                    help='num of classes for prediction')
parser.add_argument('--is3lead', action='store_true',
                    help='Default: 12-lead ECG')
parser.add_argument('--isRaligned', action='store_true',
                    help='Default ECG signal not aligned')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
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
parser.add_argument('--out-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('pretrained-file', type=str)
parser.add_argument('dataset', type=str, help='dataset name')


if __name__ == '__main__':
    args = parser.parse_args()
    writer = SummaryWriter(comment='-finetune')
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

    training_data = TVGHDataset(
        f'anno/{args.dataset}/train.csv', ecg_dir=args.ecgdir, is3lead=args.is3lead, isRaligned=args.isRaligned, split=False)
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    validating_data = TVGHDataset(
        f'anno/{args.dataset}/valid.csv', ecg_dir=args.ecgdir, is3lead=args.is3lead, isRaligned=args.isRaligned, split=False)
    valid_dataloader = DataLoader(
        validating_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    testing_data = TVGHDataset(
        f'anno/{args.dataset}/test.csv', ecg_dir=args.ecgdir, is3lead=args.is3lead, isRaligned=args.isRaligned, split=False)
    test_dataloader = DataLoader(
        testing_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    with torch.cuda.device(args.gpu_index):
        model = xresnet18SimCLR(out_dim=args.out_dim).to(args.device)
        checkpoint = torch.load(args.pretrained_file, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        for name, param in model.named_parameters():
            param.requires_grad = False

        in_dim = model.backbone.fc[0].in_features
        model.backbone.fc = nn.Linear(in_dim, args.num_classes).to(args.device)
        logging.info(model)
        logging.info(f'Total Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(
                training_data.target), y=training_data.target)
        logging.info(f'weights in cross entopy loss: {weights}')

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float, device=args.device)).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        min_loss = 1e10
        early_stop_cnt = 0
        logging.info('--------Start training model----------')
        for epoch_counter in range(args.epochs):
            train_loss = 0
            train_pred = []
            train_target = []
            model.train()
            for X, y in tqdm(train_dataloader, ncols=80):
                X, y = X.to(args.device), y.to(args.device)
                pred = model(X)
                loss = criterion(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_pred.append(pred.argmax(1))
                train_target.append(y)

            test_loss = 0
            test_pred = []
            test_target = []
            model.eval()
            with torch.no_grad():
                for X, y in tqdm(test_dataloader, ncols=80):
                    X, y = X.to(args.device), y.to(args.device)
                    pred = model(X)
                    loss = criterion(pred, y)
                    test_loss += loss.item()
                    test_pred.append(pred.argmax(1))
                    test_target.append(y)

            train_loss /= len(train_dataloader)
            train_pred = torch.cat(train_pred).cpu().numpy()
            train_target = torch.cat(train_target).cpu().numpy()
            train_tn, train_fp, train_fn, train_tp = my_confusion_matrix(train_target, train_pred)
            train_accuracy, train_precision, train_recall, train_specificity, train_f1 = cal_metrics(
                train_tn, train_fp, train_fn, train_tp)
    
            test_loss /= len(test_dataloader)
            test_pred = torch.cat(test_pred).cpu().numpy()
            test_target = torch.cat(test_target).cpu().numpy()
            test_tn, test_fp, test_fn, test_tp = my_confusion_matrix(test_target, test_pred)
            test_accuracy, test_precision, test_recall, test_specificity, test_f1 = cal_metrics(
                test_tn, test_fp, test_fn, test_tp)

            saved = ''        
            if test_recall >= 0.77 and test_specificity >= 0.77:
                saved = ' saved !!'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': model.state_dict(),
                }, is_best=False, filename=os.path.join(writer.log_dir, f'checkpoint_{epoch_counter:04d}.pth.tar'))


            writer.add_scalar('train/acc', train_accuracy, global_step=epoch_counter)
            writer.add_scalar('train/prec', train_precision, global_step=epoch_counter)
            writer.add_scalar('train/recall', train_recall, global_step=epoch_counter)
            writer.add_scalar('train/spec', train_specificity, global_step=epoch_counter)
            writer.add_scalar('train/f1', train_f1, global_step=epoch_counter)
            writer.add_scalar('train/loss', train_loss, global_step=epoch_counter)

            writer.add_scalar('test/acc', test_accuracy, global_step=epoch_counter)
            writer.add_scalar('test/prec', test_precision, global_step=epoch_counter)
            writer.add_scalar('test/recall', test_recall, global_step=epoch_counter)
            writer.add_scalar('test/spec', test_specificity, global_step=epoch_counter)
            writer.add_scalar('test/f1', test_f1, global_step=epoch_counter)
            writer.add_scalar('test/loss', test_loss, global_step=epoch_counter)

            logging.info(f'[{epoch_counter:04d}/{args.epochs:04d}]Train Acc:{train_accuracy:.3f} Prec: {train_precision:.3f} Recall:{train_recall:.3f} Spec:{train_specificity:.3f} f1:{train_f1:.3f} loss:{train_loss:.3f}'
                        f' | Val Acc:{test_accuracy:.3f} Prec: {test_precision:.3f} Recall:{test_recall:.3f} Spec:{test_specificity:.3f} f1:{test_f1:.3f} loss:{test_loss:.3f}{saved}')
            
            if test_loss < min_loss:
                early_stop_cnt = 0
                min_loss = test_loss
            else:
                early_stop_cnt += 1

            if early_stop_cnt >= args.early_stop_epoch:
                break

    logging.info('---end of training----')