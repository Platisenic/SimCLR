import os
import shutil
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def t_SNE_analysis(x: np.ndarray, filename: str): # [samples, features]
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5).fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1])
    plt.xticks()
    plt.yticks()
    plt.savefig(filename)
    plt.close(fig)

def my_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype('bool')
    y_pred = y_pred.astype('bool')
    tn = (~y_true & ~y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    tp = (y_true & y_pred).sum()

    return tn, fp, fn, tp

def cal_metrics(tn, fp, fn, tp):
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    if tp + fp == 0:
        precision = tp / (tp + fp + 1)
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = tp / (tp + fn + 1)
    else:
        recall = tp / (tp + fn)
    if fp + tn == 0:
        specificity = tn / (fp + tn + 1)
    else:
        specificity = tn / (fp + tn)
    if precision + recall == 0:
        f1 = 2 * precision * recall / (precision + recall + 1)
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, specificity, f1