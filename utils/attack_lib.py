import torch
import torch.nn as nn
from typing import Optional


def FGSM(modelF: nn.Module,
         modelC: nn.Module,
         x: torch.Tensor,
         y: torch.Tensor,
         eps: Optional[float] = 0.01):
    """ FGSM attack """
    device = next(modelF.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x.requires_grad = True

    with torch.enable_grad():
        loss = criterion(modelC(modelF(x)), y)
    grad = torch.autograd.grad(loss, x, retain_graph=False,
                               create_graph=False)[0]

    adv_x = x.detach() + eps * grad.detach().sign()

    return adv_x


def PGD(model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: Optional[float] = 0.01,
        alpha: Optional[float] = 0.001,
        steps: Optional[int] = 20):
    """ PGD attack """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)

    model.eval()

    # craft adversarial examples
    adv_x = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        adv_x.requires_grad = True
        with torch.enable_grad():
            loss = criterion(model(adv_x), y)
        grad = torch.autograd.grad(loss,
                                   adv_x,
                                   retain_graph=False,
                                   create_graph=False)[0]
        adv_x = adv_x.detach() + alpha * grad.detach().sign()
        # projection
        delta = torch.clamp(adv_x - x, min=-eps, max=eps)
        adv_x = (x + delta).detach()

    return adv_x
