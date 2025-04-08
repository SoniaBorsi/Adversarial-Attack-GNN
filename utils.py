# utils.py
import wandb
import torch.nn.functional as F
from src.metrics import compute_f1_score, compute_precision, compute_recall, accuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



def train(model, optimizer, features, adj, labels, idx_train, epoch, use_cuda):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Gradient norm 
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Learning rate
    lr = optimizer.param_groups[0]['lr']

    wandb.log({
        "epoch": epoch,
        "train_loss": loss_train.item(),
        "train_accuracy": acc_train.item(),
        "grad_norm": total_norm,
        "learning_rate": lr
    })

    print(f"Epoch {epoch:03d} | Loss: {loss_train:.4f} | Acc: {acc_train:.4f}")
    return loss_train.item(), acc_train



def test(model, features, adj, labels, idx_test, epoch=None):
    model.eval()
    output = model(features, adj)
    
    # Compute loss and accuracy
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    # Additional metrics
    f1 = compute_f1_score(output[idx_test], labels[idx_test])
    prec = compute_precision(output[idx_test], labels[idx_test])
    rec = compute_recall(output[idx_test], labels[idx_test])

    preds = output[idx_test].max(1)[1].cpu().numpy()
    true_labels = labels[idx_test].cpu().numpy()
    cm = confusion_matrix(true_labels, preds)
    
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Prepare a dict of metrics to log
    log_dict = {
        "test_loss": loss_test.item(),
        "test_accuracy": acc_test.item(),
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec
    }
    # If epoch is provided, include it in the logs
    if epoch is not None:
        log_dict["epoch"] = epoch

    # Log to Weights & Biases
    wandb.log(log_dict)
    
    print(f"Test Results - Loss: {loss_test:.4f} | Accuracy: {acc_test:.4f} | "
          f"F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    
    return loss_test.item(), acc_test.item()
