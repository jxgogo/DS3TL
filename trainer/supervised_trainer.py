import torch.nn as nn
import torch
import logging
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def eval(modelF, modelC, data_loader):
    device = next(modelF.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        total_loss, total_num = 0.0, 0.0
        y_true, y_pred = [], []
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = modelC(modelF(x))
            loss = criterion(logits, y)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            total_loss += loss.cpu().item() * num_batch
        acc = accuracy_score(y_true, y_pred)
        bca = balanced_accuracy_score(y_true, y_pred)
    return total_loss / total_num, acc, bca


def train(modelF, modelC, train_loader, test_loader, optimizer, epochs):
    device = next(modelF.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)
    hists = []
    for epoch in range(epochs):
        modelF.train()
        modelC.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = modelC(modelF(batch_x))
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            modelF.eval()
            modelC.eval()
            train_loss, train_acc, train_bca = eval(modelF, modelC, train_loader)
            test_loss, test_acc, test_bca = eval(modelF, modelC, test_loader)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} | test loss: {:.4f} test acc: {:.2f} test bca: {:.2f}'
                .format(epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc, test_bca))

            hists.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_bca": train_bca,
                    "loss": test_loss,
                    "acc": test_acc,
                    "bca": test_bca
                }
            )

    return hists, modelF, modelC
