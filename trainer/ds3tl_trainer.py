import torch
import logging
from itertools import cycle
from sklearn.metrics import *
from torch.nn import functional as F

from utils.ramp import *
from utils.pytorch_utils import AverageMeter
from utils.loss import ContrastLoss, WeightedInformationMaximization


class Prototype(object):
    def __init__(self, n_class, n_feature, device):
        self.n_class = n_class
        self.prototype_counts = torch.zeros([n_class]).to(device)
        self.prototypes = torch.zeros([n_class, n_feature]).to(device)

    def update(self, features, labels):
        for ite in range(self.n_class):
            class_temp = features[labels == ite]
            if len(class_temp):
                self.prototypes[ite, :] = (self.prototypes[ite, :] * self.prototype_counts[ite] + class_temp.sum(0))
                self.prototype_counts[ite] += len(class_temp)
                self.prototypes[ite, :] /= self.prototype_counts[ite]

    def __call__(self):
        return self.prototypes


class Trainer:
    def __init__(self, modelF, modelC, modelH, optimizer, args):
        self.modelF     = modelF
        self.modelC     = modelC
        self.modelH     = modelH
        self.feature_dim= args.feature_dim
        self.optimizer  = optimizer
        self.rampup     = exp_rampup(args.rampuplength)
        self.device     = args.device
        self.ulb_batch_size = args.ulb_batch_size
        self.epoch      = 0
        self.sp_loss    = torch.nn.CrossEntropyLoss()
        self.ctr_loss   = ContrastLoss()
        # bool
        self.L_u        = args.unlab_loss
        self.L_contras  = args.ctr_loss
        self.L_ent      = args.entmin_loss
        # hyperparameter
        self.usp_weight = args.ulb_weight
        self.ctr_weight = args.ctr_weight
        self.entmin_weight = args.entmin_weight
        self.T          = args.T
        self.threshold  = args.threshold
        self.threshold_push = args.threshold_push
        self.num_neighbors  = args.topm

    def mine_nn_batch(self, memory_bank_unlabeled, num_neighbors=16):
        """
        memory_bank_unlabeled: per_batch features
        num_neighbors: should < num_batch / num_class
        """
        _, idx = memory_bank_unlabeled.topk(num_neighbors, 0, True, True)
        idx = idx.t()
        topk_index = []
        topk_labels = []
        topk_probs = []
        for labels, index in enumerate(idx):
            topk_index.extend(index)
            topk_labels.extend([labels] * num_neighbors)
            topk_probs.extend(memory_bank_unlabeled[index])
        return torch.Tensor(topk_index).type(torch.LongTensor), torch.Tensor(topk_labels).type(
            torch.LongTensor), torch.cat(topk_probs).reshape(len(topk_index), -1)

    def train(self, label_loader, unlab_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        self.modelH.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_ctr = AverageMeter()
        losses_entmin = AverageMeter()

        bank = self.modelH(self.modelF(self.bank.to(self.device))).detach()
        labels = self.labels.to(self.device)
        label_prototype = Prototype(self.num_classes, self.feature_dim, self.device)
        label_prototype.update(bank, labels)

        batch_idx = 0
        for (label_x, label_y), (targ_x, targ_y), ((unlab_weak, unlab_strong), unlab_y, idx) in zip(cycle(label_loader), cycle(test_loader), unlab_loader):
            batch_idx += 1
            batch_size = label_x.shape[0]
            inputs, targets_x = torch.cat((label_x, unlab_weak, unlab_strong)).to(self.device), label_y.to(self.device)
            feat = self.modelF(inputs)
            logits = self.modelC(feat)
            head = self.modelH(feat)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            head_u_w, head_u_s = head[batch_size:].chunk(2)
            feat_u_w, _ = feat[batch_size:].chunk(2)
            del logits
            del head
            del feat
            logits_targ = self.modelC(self.modelF(targ_x.to(self.device)))

            Lx = self.sp_loss(logits_x, targets_x)

            with torch.no_grad():
                sim_weak = head_u_w @ label_prototype().t()  # similarity between weak features and prototypes
                soft_target = F.softmax(sim_weak / self.T, dim=1)

                # scaling
                pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
                pseudo_label = pseudo_label * soft_target
                pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold)

            loss = Lx
            losses_x.update(Lx.item())

            if self.L_u:
                Lu = (mask.float() * F.cross_entropy(logits_u_s, targets_u, reduction='none')).mean()
                loss += self.usp_weight * Lu
                losses_u.update(Lu.item())

            if self.L_contras:
                topk_index, topk_labels, probs = self.mine_nn_batch(soft_target, num_neighbors=self.num_neighbors)
                head = head_u_w[topk_index]
                label_prototype.update(head.detach(), topk_labels)
                L_ctr = self.ctr_loss(torch.cat([head_u_w.unsqueeze(1), head_u_s.unsqueeze(1)], dim=1),
                                      pseudo_label,
                                      select_thresh=self.threshold_push)
                losses_ctr.update(L_ctr.item())
                loss += self.ctr_weight * L_ctr
            if self.L_ent:
                L_entmin = WeightedInformationMaximization(gent=False)(logits_targ)
                losses_entmin.update(L_entmin.item())
                loss += self.entmin_weight * L_entmin

            losses.update(loss.item())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self, data_loader):
        self.modelF.eval()
        self.modelC.eval()
        self.modelH.eval()
        with torch.no_grad():
            total_loss, total_num = 0.0, 0.0
            y_true, y_pred = [], []
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                num_batch = x.shape[0]
                total_num += num_batch
                logits = self.modelC(self.modelF(x))
                loss = self.sp_loss(logits, y)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                total_loss += loss.cpu().item() * num_batch
            acc = accuracy_score(y_true, y_pred)
            bca = balanced_accuracy_score(y_true, y_pred)
        return total_loss / total_num, acc, bca

    def loop(self, epochs, label_data, unlab_data, test_data):
        best_acc, best_bca = 0, 0
        hists = []
        for ep in range(epochs):
            self.epoch = ep
            # register in each epoch
            self.bank = label_data.dataset.data.clone().detach()
            self.labels = label_data.dataset.targets.clone().detach()
            self.num_classes = label_data.dataset.num_classes
            self.train(label_data, unlab_data, test_data)
            test_loss, test_acc, test_bca = self.test(test_data)
            if test_acc >= best_acc:
                best_acc = test_acc
                best_bca = test_bca

            if (ep + 1) % 1 == 0:
                train_loss, train_acc, train_bca = self.test(label_data)
                logging.info('Test Epoch {}/{}: '
                             'train acc: {:.3f} train bca: {:.3f} | '
                             'best acc: {:.3f} best bca: {:.3f} | '
                             'test acc: {:.3f} test bca: {:.3f}'.format(
                    ep, epochs, train_acc, train_bca, best_acc, best_bca, test_acc, test_bca))
                hists.append(
                    {
                        "epoch": ep + 1,
                        "best_acc": best_acc,
                        "best_bca": best_bca,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_bca": train_bca,
                        "loss": test_loss,
                        "acc": test_acc,
                        "bca": test_bca
                    }
                )
        return hists
