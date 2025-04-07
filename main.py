import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import GCN
from data import load_data
from config import config
from utils import train, test
#from attacks import apply_nettack



def main():
    # Config
    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    hidden = config["hidden"]
    dropout = config["dropout"]
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    apply_attack = config.get("apply_attack", False)
    dataset_path = config["datasets"][config["active_dataset"]]

    # Load data
    adj, features, labels, idx_train, idx_test = load_data(dataset_path)

    # Model
    model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_cuda:
        model.cuda()
        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
        idx_train, idx_test = idx_train.cuda(), idx_test.cuda()

    # Training loop
    t_total = time.time()
    for epoch in range(epochs):
        train(model, optimizer, features, adj, labels, idx_train, epoch, use_cuda)
    print("Training finished in {:.4f}s".format(time.time() - t_total))

    # Test on clean graph
    print("\n[ Clean Evaluation ]")
    test(model, features, adj, labels, idx_test)

    # adversarial attack
    # if apply_attack:
    #     target_node = idx_test[0].item()
    #     n_perturbations = 3

    #     print(f"\n[ Nettack ] Targeting node {target_node} with {n_perturbations} perturbations...")
    #     perturbed_adj, perturbed_features = apply_nettack(model, adj, features, labels, target_node, n_perturbations)

    #     model.eval()
    #     output = model(perturbed_features, perturbed_adj)
    #     pred = output[target_node].argmax().item()
    #     true = labels[target_node].item()
    #     print(f"After attack: Target node {target_node} - True label: {true}, Predicted: {pred}")
    # else:
    #     print("\n[ Nettack Skipped ] Attack not applied based on config setting.")

if __name__ == "__main__":
    main()
