import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from model import GCN
from data import load_data
from config import config
from utils import train, test

def main():
    # Initialize wandb
    wandb.init(
        project="gnn",
        config=config,
        name=f"{config['active_dataset']}_run_{int(time.time())}",
    )
    run_config = wandb.config

    # Select dataset folder
    dataset_path = run_config["datasets"][run_config["active_dataset"]]

    # Check if the user wants to load the perturbed dataset
    if run_config["use_perturbed"]:
        perturbed_path = dataset_path.rstrip("/") + "/perturbed/"
        print("→ Loading perturbed dataset from:", perturbed_path)
        dataset_path = perturbed_path
    else:
        print("→ Loading original dataset from:", dataset_path)

    # Load dataset
    adj, features, labels, idx_train, idx_test = load_data(dataset_path)

    # Set up the GCN model
    model = GCN(
        nfeat=features.shape[1],
        nhid=run_config["hidden"],
        nclass=labels.max().item() + 1,
        dropout=run_config["dropout"],
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=run_config["lr"],
        weight_decay=run_config["weight_decay"]
    )

    use_cuda = run_config["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
        idx_train, idx_test = idx_train.cuda(), idx_test.cuda()

    best_accuracy = 0.0
    epochs_without_improvement = 0

    # Training loop with early stopping
    for epoch in range(run_config["epochs"]):
        loss, acc = train(model, optimizer, features, adj, labels, idx_train, epoch, use_cuda)
        loss_test, acc_test = test(model, features, adj, labels, idx_test, epoch)

        if acc_test > best_accuracy:
            best_accuracy = acc_test
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= run_config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Training completed.")

if __name__ == "__main__":
    main()
