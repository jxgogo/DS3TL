import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


def mse_with_softmax(logit1, logit2):
    logit2 = logit2.detach()
    assert logit1.size()==logit2.size()
    return F.mse_loss(F.softmax(logit1, dim=-1), F.softmax(logit2, dim=-1))


def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).to(targets.device)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -torch.mean((x.softmax(1) * torch.log(x.softmax(1) + 1e-8)).sum(dim=1))  # H


def softmax_div_entropy(x: torch.Tensor) -> torch.Tensor:
    """category-diversity loss"""
    pb_pred_all = x.softmax(1).sum(dim=0)
    pb_pred_tgt = 1.0 / pb_pred_all.sum() * pb_pred_all   # normalizatoin
    return - (- torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-8))))  # -H


class LabelSmoothLoss(nn.Module):
    def __init__(self, alpha) -> None:
        super(LabelSmoothLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        n_class = pred.shape[-1]
        lb_pos = 1.0 - self.alpha
        lb_neg = self.alpha / (n_class - 1)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            label_onehot = torch.empty_like(pred).fill_(lb_neg).scatter_(
                dim=1, index=target.data.unsqueeze(1), value=lb_pos)
        return torch.mean(-torch.sum(label_onehot * pred, dim=-1))


def Entropy(input:torch.Tensor)->torch.Tensor:
    """
    Compute the entropy
    :param input: the softmax output
    :return: entropy
    """
    entropy=-input*torch.log(input+1e-5)
    entropy=torch.sum(entropy,dim=1)
    return entropy


class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """
    def __init__(self,
                 t:Optional[float]=2.0):
        super(ClassConfusionLoss,self).__init__()
        self.t=t

    def forward(self,
                output:torch.Tensor)->torch.Tensor:
        n_sample,n_class=output.shape
        softmax_out=nn.Softmax(dim=1)(output/self.t)
        entropy_weight=Entropy(softmax_out).detach()
        entropy_weight=1+torch.exp(-entropy_weight)
        entropy_weight=(n_sample*entropy_weight/torch.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix=torch.mm((softmax_out*entropy_weight).transpose(1,0),softmax_out)
        class_confusion_matrix=class_confusion_matrix/torch.sum(class_confusion_matrix,dim=1)
        mcc_loss=(torch.sum(class_confusion_matrix)-torch.trace(class_confusion_matrix))/n_class
        return mcc_loss


def entropy_weight(output, T=2.0):
    n_sample = output.shape[0]
    softmax_out = nn.Softmax(dim=1)(output / T)
    entropy_weight = Entropy(softmax_out).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight))  # Normalization

    return entropy_weight

def kl_weight(output1, output2):
    n_sample = output1.shape[0]
    V = F.kl_div(F.log_softmax(output1, dim=-1), F.softmax(output2, dim=-1), reduction='none').mean(dim=-1)
    kl_weight = 1 + torch.exp(-V.detach())
    kl_weight = (n_sample * kl_weight / torch.sum(kl_weight))  # Normalization

    return kl_weight, V


class WeightedInformationMaximization(nn.Module):
    """
    Sample weighted information maximization
    """
    def __init__(self,
                 gent:bool,
                 t:Optional[float]=2.0):
        super(WeightedInformationMaximization,self).__init__()
        self.gent=gent
        self.t = t

    def forward(self,
                output:torch.Tensor)->torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        entropy_weight = entropy_weight.view(-1)
        entropy_loss=torch.mean(Entropy(softmax_out) * entropy_weight)
        if self.gent:
            msoftmax=softmax_out.mean(dim=0)
            gentropy_loss=torch.sum(-msoftmax*torch.log(msoftmax+1e-8))
            entropy_loss-=gentropy_loss
        return entropy_loss


class ContrastLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, pseudo_label, select_thresh):
        device = features.device
        batch_size = features.shape[0]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        select_matrix = contrast_select(thresh=select_thresh, max_probs=max_probs)
        # weight according to predict probability
        max_probs = max_probs.contiguous().view(-1, 1)
        targets_u = targets_u.contiguous().view(-1, 1)
        mask = torch.eq(targets_u, targets_u.T).float().to(device)
        score_mask = torch.matmul(max_probs, max_probs.T)  # shape: batch_size x batch_size
        mask = mask.mul(score_mask)
        mask = mask * select_matrix

        # contast loss
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0, :]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Uself.contrast_mode')

        # feature similarity
        anchor_similarity = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                      self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_similarity, dim=1, keepdim=True)
        logits = anchor_similarity - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # mask out self contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), dim=1,
                                    index=torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                    value=0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute over positive
        mask += 1e-8
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob

        return loss.mean()


def contrast_select(thresh, max_probs):
    contrast_mask = max_probs.ge(thresh).float()
    contrast_mask2 = torch.clone(contrast_mask)
    contrast_mask2[contrast_mask == 0] = -1
    select_elements = torch.eq(contrast_mask2.reshape([-1, 1]),
                               contrast_mask.reshape([-1, 1]).T).float()
    select_elements += torch.eye(contrast_mask.shape[0]).to(max_probs.device)
    select_elements[select_elements > 1] = 1
    select_matrix = torch.ones(contrast_mask.shape[0]).to(max_probs.device) * select_elements
    return select_matrix

