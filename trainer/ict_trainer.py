import copy

import torch
import logging
from itertools import cycle
from sklearn.metrics import *

from utils.loss import mse_with_softmax
from utils.ramp import *
from utils.mixup import mixup_one_target
from utils.pytorch_utils import AverageMeter


class Trainer:
    def __init__(self, modelF, modelC, optimizer, args):
        self.modelF = modelF
        self.ema_modelF = copy.deepcopy(modelF)
        self.modelC = modelC
        self.ema_modelC = copy.deepcopy(modelC)
        self.optimizer  = optimizer
        self.device     = args.device
        self.sp_loss    = torch.nn.CrossEntropyLoss()
        self.usp_loss   = self.sp_loss
        self.usp_weight = args.ulb_weight
        self.rampup     = exp_rampup(args.rampuplength)
        self.epoch      = 0
        self.alpha      = 1.
        self.global_step = 0
        self.ema_decay   = args.ema_decay
        self.threshold   = args.threshold
        self.T = args.T

    def update_ema(self, model_list, ema_model_list, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_model, model in zip(ema_model_list, model_list):
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                # ema_param.data.mul_(alpha).add_(1-alpha, param.data)
                ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))

    def train(self, label_loader, unlab_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        self.ema_modelF.train()
        self.ema_modelC.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        with torch.enable_grad():
            batch_idx = 0
            for (label_x, label_y), (targ_x, _), ((unlab_x1, _), _) in zip(cycle(label_loader), cycle(test_loader), unlab_loader):
                batch_idx += 1
                self.global_step += 1
                label_x, label_y = label_x.to(self.device), label_y.to(self.device)
                unlab_x1 = unlab_x1.to(self.device)

                # === forward ===
                Lx = self.sp_loss(self.modelC(self.modelF(label_x)), label_y)
                losses_x.update(Lx.item())

                # === Semi-supervised Training ===
                with torch.no_grad():
                    ema_outputs_u = self.ema_modelC(self.ema_modelF(unlab_x1))

                # mixup-consistency loss
                mixed_ux, mixed_uy, lam = mixup_one_target(unlab_x1, ema_outputs_u, self.alpha, self.device)
                mixed_outputs_u = self.modelC(self.modelF(mixed_ux))
                cons_loss = mse_with_softmax(mixed_outputs_u, mixed_uy)
                losses_u.update(cons_loss.item())

                loss = Lx + self.usp_weight * cons_loss
                losses.update(loss.item())

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update mean-teacher
                self.update_ema([self.modelF, self.modelC], [self.ema_modelF, self.ema_modelC], self.ema_decay, self.global_step)

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

