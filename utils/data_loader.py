import torch
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('.')
from utils.augment import Transform


label_dict_1 = {
    'left_hand ': 0,
    'right_hand': 1
}

label_dict_3 = {
    'right_hand': 0,
    'feet      ': 1
}


class SSL_dataset(Dataset):
    def __init__(self,
                 data: torch.Tensor,
                 targets: torch.Tensor,
                 transform=None,
                 strong_transform=None,
                 idx=False,
                 ori=False):
        assert data.size(0) == targets.size(0), "Size mismatch between tensors"
        self.data = data
        self.targets = targets.long()
        self.transform = transform
        self.strong_transform = strong_transform
        self.idx = idx
        self.ori = ori
        self.num_classes = len(torch.unique(targets))

        if self.strong_transform is not None:
            assert self.transform is not None, "Weak augmentation needed"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is None:
            if self.idx:
                return img, target, index
            return img, target

        # set augmented images
        img1 = img
        img2 = img
        if self.transform is not None:
            img1 = self.transform(img)
            if self.strong_transform is not None:
                img2 = self.strong_transform(img)
            else:
                img2 = self.transform(img)

        if self.idx:
            if self.ori:
                return (img, img1, img2), target, index
            else:
                return (img1, img2), target, index

        if self.ori:
            return (img, img1, img2), target
        else:
            return (img1, img2), target

    def __len__(self):
        return self.data.size(0)


def split(x, y, num_class, ratio):
    lb_idx = []
    for c in range(num_class):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, int(np.ceil(len(idx) * ratio)), False)
        lb_idx.extend(idx)
    ulb_idx = np.array(sorted(list(set(range(len(x))) - set(lb_idx))))
    return lb_idx, ulb_idx


def Trans(alg, augment, weak, strong):
    if alg == 'fullysupervised':
        transform = None
        strong_transform = None
    elif alg == 'fixmatch' or alg == 'simmatch' or alg == 'ds3tl':
        transform = Transform(augment, gs_am=weak)
        strong_transform = Transform(augment, gs_am=strong)
    elif alg == 'meanteacher' \
            or alg == 'mixmatch' \
            or alg == 'pimodel' \
            or alg == 'pseudolabel' \
            or alg == 'temporal' \
            or alg == 'vat' \
            or alg == 'ict':
        transform = Transform(augment, gs_am=weak)
        strong_transform = None
    else:
        raise Exception(f'Need to define the {alg} corresponding transform')
    return transform, strong_transform


def Load_ssl_cross(data_path,
                   test_id,
                   num_sub,
                   num_class,
                   alg='fullysupervised',
                   lb_ratio=0.05,
                   label_exclude=False,
                   augment='noise',
                   weak=0.1,
                   strong=0.2):
    label_dict = {
        'left_hand ': 0,
        'right_hand': 1,
        'feet      ': 2,
        'tongue    ': 3
    }
    transform, strong_transform = Trans(alg, augment, weak, strong)

    x_lb, y_lb, x_ulb, y_ulb, = [], [], [], []
    for i in range(num_sub):
        data = np.load(data_path + f'/tangent/s{i}.npz')
        x, y = data['tangent'], data['y']
        y = np.array([label_dict[yi] for yi in y])

        if i == test_id:
            x_test, y_test = x, y
        else:
            lb_idx, ulb_idx = split(x, y, num_class, lb_ratio)
            if lb_ratio == 1:
                lb_idx, ulb_idx = np.arange(len(x)), np.arange(len(x))

            x_lb.extend(x[lb_idx])
            y_lb.extend(y[lb_idx])
            x_ulb.extend(x[ulb_idx])
            y_ulb.extend(y[ulb_idx])

    train_labeled_dataset = SSL_dataset(torch.Tensor(np.array(x_lb)), torch.Tensor(np.array(y_lb)).squeeze())

    if not label_exclude and lb_ratio != 1:
        x_ulb = np.concatenate((x_ulb, x_lb), axis=0)
        y_ulb = np.concatenate((y_ulb, y_lb))

    if 'temporal' in alg or 'pseudolabel' in alg or 'fixmatch' in alg or 'simmatch' in alg or 'ds3tl' in alg:
        ori = False
        if 'simmatch' in alg:
            ori = True
        train_unlabeled_dataset = SSL_dataset(torch.Tensor(np.array(x_ulb)),
                                              torch.Tensor(np.array(y_ulb)).squeeze(),
                                              transform=transform,
                                              strong_transform=strong_transform,
                                              idx=True,
                                              ori=ori)
    else:
        train_unlabeled_dataset = SSL_dataset(torch.Tensor(np.array(x_ulb)),
                                              torch.Tensor(np.array(y_ulb)).squeeze(),
                                              transform=transform,
                                              strong_transform=strong_transform)

    test_dataset = SSL_dataset(torch.Tensor(np.array(x_test)), torch.Tensor(np.array(y_test)).squeeze())

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
