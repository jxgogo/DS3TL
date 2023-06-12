from collections import Counter

import torch
import logging
from itertools import cycle
from sklearn.metrics import *
from torch.nn import functional as F

from utils.ramp import *
from utils.pytorch_utils import AverageMeter
from utils.loss import WeightedInformationMaximization
from utils.augment import Transform


class Trainer:
    def __init__(self, modelF, modelC, modelH, optimizer, args):
        self.modelF = modelF
        self.modelC = modelC
        self.modelH = modelH
        self.optimizer = optimizer
        self.device = args.device
        self.sp_loss = torch.nn.CrossEntropyLoss()
        self.usp_loss = self.sp_loss
        self.usp_weight = args.ulb_weight
        self.rampup     = exp_rampup(args.rampuplength)
        self.epoch = 0
        self.T = args.T
        self.threshold = args.threshold

        self.lamda_in = args.lamda_in
        self.T_proto = 0.1
        self.c_smooth = 0.9

    def train(self, label_loader, unlab_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        self.modelH.train()
        batch_idx = 0
        for (label_x, label_y), (targ_x, _), ((unlab_x, unlab_weak, unlab_strong), unlab_y, idx) in \
                zip(cycle(label_loader), cycle(test_loader), unlab_loader):
            bank = self.modelH(self.modelF(self.bank.to(self.device))).detach()
            labels = self.labels.to(self.device)
            batch_idx += 1
            batch_size = label_x.shape[0]
            batch_u = unlab_weak.shape[0]

            inputs = torch.cat((label_x, unlab_weak, unlab_strong)).to(self.device)
            targets_x = label_y.to(self.device)

            feature = self.modelF(inputs)
            logits = self.modelC(feature)
            head = self.modelH(feature)

            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            head_u_w, head_u_s = head[batch_size:].chunk(2)
            feat_u_w, _ = feature[batch_size:].chunk(2)
            del feature
            del logits
            del head

            # loss_in
            with torch.no_grad():
                prob_ku_orig = F.softmax(logits_u_w, dim=-1)
                teacher_logits = head_u_w @ bank.t()  # cosine similarity
                teacher_prob_orig = F.softmax(teacher_logits / self.T_proto, dim=1)  # softmax
                factor = prob_ku_orig.gather(1, labels.expand([batch_u, -1]))  # p_unfold
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)   # new qw used in loss_in

                bs = teacher_prob_orig.size(0)
                aggregated_prob = torch.zeros([bs, self.num_classes]).to(self.device)
                aggregated_prob = aggregated_prob.scatter_add(1, labels.expand([bs, -1]), teacher_prob_orig)  # q_agg
                prob_ku = prob_ku_orig * self.c_smooth + aggregated_prob * (1 - self.c_smooth)  # new pw used in loss_u
                max_probs, targets_u = torch.max(prob_ku, dim=-1)
                mask = max_probs.ge(self.threshold)
            student_logits = head_u_s @ bank.t()
            student_prob = F.softmax(student_logits / self.T_proto, dim=1)
            loss_in = (torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1)).mean()

            # loss_x
            Lx = self.sp_loss(logits_x, targets_x)
            # loss_u
            Lu = (torch.sum(-F.log_softmax(logits_u_s, dim=1) * prob_ku.detach(), dim=1) * mask.float()).mean()
            # loss all
            loss = Lx + self.usp_weight * Lu + self.lamda_in * loss_in

            # backwark
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
        self.num_classes = label_data.dataset.num_classes
        best_acc, best_bca = 0, 0
        hists = []
        for ep in range(epochs):
            self.epoch = ep
            # register in each epoch
            self.bank = label_data.dataset.data.clone().detach()
            self.labels = label_data.dataset.targets.clone().detach()

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

