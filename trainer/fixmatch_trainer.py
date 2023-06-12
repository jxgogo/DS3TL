import torch
import logging
from itertools import cycle
from sklearn.metrics import *
from torch.nn import functional as F

from utils.ramp import *
from utils.pytorch_utils import AverageMeter


class Trainer:
    def __init__(self, modelF, modelC, optimizer, args):
        self.modelF     = modelF
        self.modelC     = modelC
        self.optimizer  = optimizer
        self.device     = args.device
        self.sp_loss    = torch.nn.CrossEntropyLoss()
        self.usp_loss   = self.sp_loss
        self.usp_weight = args.ulb_weight
        self.rampup     = exp_rampup(args.rampuplength)
        self.epoch      = 0
        self.T          = args.T
        self.threshold  = args.threshold

    def train(self, label_loader, unlab_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        batch_idx = 0
        for (label_x, label_y), ((unlab_weak, unlab_strong), unlab_y, _) in zip(cycle(label_loader), unlab_loader):
            batch_idx += 1
            batch_size = label_x.shape[0]
            inputs, targets_x = torch.cat((label_x, unlab_weak, unlab_strong)).to(self.device), label_y.to(self.device)
            feat = self.modelF(inputs)
            logits = self.modelC(feat)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            feat_u_w, _ = feat[batch_size:].chunk(2)
            del logits
            del feat

            pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold)

            Lx = self.sp_loss(logits_x, targets_x)
            Lu = (mask.float() * F.cross_entropy(logits_u_s, targets_u, reduction='none')).mean()
            loss = Lx + self.usp_weight * Lu

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Train Epoch: {:}. batch: {:}. Loss: {:.4f}. Loss_x: {:.4f}. Loss_u: {:.4f}.".format(
            self.epoch+1,
            batch_idx + 1,
            losses.avg,
            losses_x.avg,
            losses_u.avg
        ))

    def test(self, data_loader):
        self.modelF.eval()
        self.modelC.eval()
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
