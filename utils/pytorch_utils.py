import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional
from torch.nn import Parameter
from torch.utils.data import Dataset, TensorDataset
import argparse
import torch
import random
import os


def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)


def bca_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def standard_normalize(x_train, x_test, clip_range=None):
    mean, std = np.mean(x_train), np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    if clip_range is not None:
        x = np.clip(x_train, a_min=clip_range[0], a_max=clip_range[1])
        x = np.clip(x_test, a_min=clip_range[0], a_max=clip_range[1])
    return x_train, x_test


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer: nn.Module, epoch: int,
                         learning_rate: float):
    """decrease the learning rate"""
    lr = learning_rate
    if epoch >= 50:
        lr = learning_rate * 0.1
    if epoch >= 100:
        lr = learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_for_balanced_classes(y: torch.Tensor):
    count = [0.0] * len(np.unique(y.numpy()))
    for label in y:
        count[label] += 1.0
    count = [len(y) / x for x in count]
    weight = [0.0] * len(y)
    for idx, label in enumerate(y):
        weight[idx] = count[label]

    return weight


class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}

    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_feat(modelF, modelC, dataset, device, three=False, four=False):
    modelF.eval()
    modelC.eval()
    start = True
    with torch.no_grad():
        iter_data = iter(dataset)
        for i in range(len(dataset)):
            if four:
                (input, _), label, _ = iter_data.next()
            elif three:
                (input, _), label = iter_data.next()
            else:
                input, label = iter_data.next()
            input = input.to(device)
            feature = modelF(input)
            output = modelC(feature)
            if start:
                outputs = output.float().cpu()
                labels = label.float().cpu()
                features = feature.float().cpu()
                start = False
            else:
                outputs = torch.cat((outputs, output.float().cpu()), 0)
                labels = torch.cat((labels, label.float().cpu()), 0)
                features = torch.cat((features, feature.float().cpu()), 0)
    feat = features.detach().numpy()
    y = labels.detach().numpy()
    logits = outputs.detach().numpy()
    return feat, y, logits


def class_centroid(number_class, F_lu, Y_lu):
    res = torch.zeros([number_class, F_lu.size(1)])
    ls_class = torch.unique(Y_lu)
    ls_class = torch.sort(ls_class)[0]
    for ite in ls_class:
        res[ite,:] = F_lu[Y_lu==ite].mean(0)
    return res