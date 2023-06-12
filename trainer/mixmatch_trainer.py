import torch
import logging
from itertools import cycle
from sklearn.metrics import *
from torch.nn import functional as F

from utils.loss import one_hot
from utils.ramp import *
from utils.mixup import mixup_one_target
from utils.pytorch_utils import AverageMeter


class Trainer:
    def __init__(self, modelF, modelC, optimizer, args):
        self.modelF = modelF
        self.modelC = modelC
        self.optimizer  = optimizer
        self.device     = args.device
        self.sp_loss    = torch.nn.CrossEntropyLoss()
        self.usp_loss   = self.sp_loss
        self.usp_weight = args.ulb_weight
        self.rampup     = exp_rampup(args.rampuplength)
        self.epoch      = 0
        self.T          = args.T
        self.alpha      = 1.
        self.threshold = args.threshold

    def train(self, label_loader, unlab_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        with torch.enable_grad():
            batch_idx = 0
            for (label_x, label_y), (targ_x, _), ((unlab_x1, unlab_x2), _) in zip(cycle(label_loader), cycle(test_loader), unlab_loader):
                batch_idx += 1
                label_x, label_y = label_x.to(self.device), label_y.to(self.device)
                unlab_x1, unlab_x2 = unlab_x1.to(self.device), unlab_x2.to(self.device)
                lbs = label_x.size(0)

                # label guessing
                with torch.no_grad():
                    outputs1 = self.modelC(self.modelF(unlab_x1))
                    outputs2 = self.modelC(self.modelF(unlab_x2))
                    # avg
                    avg_outputs = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
                    avg_outputs = (avg_outputs / avg_outputs.sum(dim=-1, keepdim=True))
                    # sharpening
                    sharpen_pslab = avg_outputs ** (1 / self.T)
                    sharpen_pslab = sharpen_pslab / sharpen_pslab.sum(dim=1, keepdim=True)
                    sharpen_pslab = sharpen_pslab.detach()
                    labeled_label = one_hot(label_y, outputs1.size(1))
                    # labeled_label = torch.softmax(self.modelC(self.modelF(label_x)) / self.T, dim=-1)
                # concat data
                input_x = torch.cat([label_x, unlab_x1, unlab_x2], dim=0)
                input_y = torch.cat([labeled_label, sharpen_pslab, sharpen_pslab], dim=0)
                # forward
                mixed_x, mixed_y, _ = mixup_one_target(input_x, input_y,
                                                       self.alpha,
                                                       self.device,
                                                       is_bias=False)
                mixed_outputs = self.modelC(self.modelF(mixed_x))
                # loss for labeled samples
                Lx = -torch.mean(torch.sum(mixed_y[:lbs] * F.log_softmax(mixed_outputs[:lbs], dim=1), dim=1))
                losses_x.update(Lx.item())
                # loss for unlabeled samples
                cons_loss = F.mse_loss(torch.softmax(mixed_outputs[lbs:], dim=1), mixed_y[lbs:])
                losses_u.update(cons_loss.item())

                loss = Lx + self.usp_weight * cons_loss
                losses.update(loss.item())

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
        hists = []
        best_acc, best_bca = 0, 0
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

