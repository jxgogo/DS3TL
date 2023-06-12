import torch
import logging
from itertools import cycle
from sklearn.metrics import *

from utils.loss import mse_with_softmax
from utils.ramp import *
from utils.pytorch_utils import Bn_Controller
from utils import attack_lib
from utils.pytorch_utils import AverageMeter


class Trainer:
    def __init__(self, modelF, modelC, optimizer, args):
        self.modelF = modelF
        self.modelC = modelC
        self.optimizer  = optimizer
        self.device     = args.device
        self.sp_loss    = torch.nn.CrossEntropyLoss()
        self.usp_loss   = mse_with_softmax
        self.usp_weight = args.ulb_weight
        self.rampup     = exp_rampup(args.rampuplength)
        self.epoch      = 0
        self.usp_use_bn = True
        self.bn_controller = Bn_Controller()
        self.eps        = args.eps
        self.threshold = args.threshold
        self.T = args.T

    def train(self, label_loader, unlab_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        with torch.enable_grad():
            batch_idx = 0
            for (label_x, label_y), (targ_x, _), ((unlab_x1, _), _) in zip(cycle(label_loader), cycle(test_loader), unlab_loader):
                batch_idx += 1
                label_x, label_y = label_x.to(self.device), label_y.to(self.device)
                unlab_x1 = unlab_x1.to(self.device)

                feat_u_w = self.modelF(unlab_x1)
                unlab_outputs = self.modelC(feat_u_w)

                Lx = self.sp_loss(self.modelC(self.modelF(label_x)), label_y)
                losses_x.update(Lx.item())

                # === Semi-supervised Training Phase ===
                # consistency loss
                with torch.no_grad():
                    vlogits = unlab_outputs.clone().detach()
                if self.usp_use_bn:
                    self.bn_controller.freeze_bn(self.modelF)
                unlab_adv = attack_lib.FGSM(self.modelF, self.modelC,
                                            unlab_x1,
                                            y=torch.max(vlogits, dim=-1)[1],
                                            eps=self.eps)
                adv_outputs = self.modelC(self.modelF(unlab_adv))
                if self.usp_use_bn:
                    self.bn_controller.unfreeze_bn(self.modelF)
                cons_loss = self.usp_loss(adv_outputs, vlogits)
                losses_u.update(cons_loss.item())

                loss = Lx + self.usp_weight * cons_loss
                losses.update(loss.item())

                # backwark
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

