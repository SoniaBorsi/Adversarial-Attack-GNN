# utils.py

import torch
import torch.nn.functional as F
from metrics import accuracy

train_losses = []
train_accuracies = []

def train(model, optimizer, features, adj, labels, idx_train, epoch, use_cuda):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    train_losses.append(loss_train.item())
    train_accuracies.append(acc_train.item())

    print(f"Epoch {epoch:03d} | Loss: {loss_train:.4f} | Acc: {acc_train:.4f}")

def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f"Test set results: Loss = {loss_test:.4f} | Accuracy = {acc_test:.4f}")
    return loss_test.item(), acc_test.item()

