import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from model import GCN
from data import load_data
from config import config
from utils import train, test
#from attacks import apply_nettack

def main():
    # Initialize wandb
    wandb.init(
        project="gnn",
        config=config,
        name=f"{config['active_dataset']}_run_{int(time.time())}",
    )
    run_config = wandb.config

    # Load data
    dataset_path = run_config["datasets"][run_config["active_dataset"]]
    adj, features, labels, idx_train, idx_test = load_data(dataset_path)

    # Setup model
    model = GCN(
        nfeat=features.shape[1],
        nhid=run_config["hidden"],
        nclass=labels.max().item() + 1,
        dropout=run_config["dropout"],
    )
    optimizer = optim.Adam(model.parameters(),
                           lr=run_config["lr"],
                           weight_decay=run_config["weight_decay"])

    use_cuda = run_config["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
        idx_train, idx_test = idx_train.cuda(), idx_test.cuda()

    # Train for several epochs, and test every epoch
    for epoch in range(run_config["epochs"]):
        # 1) One epoch of training
        loss, acc = train(model, optimizer, features, adj, labels, idx_train, epoch, use_cuda)

        # 2) Evaluate on test set at end of this epoch
        loss_test, acc_test = test(model, features, adj, labels, idx_test, epoch)

    print("Training completed.")

    # final test 
    print("\n[Evaluation ]")
    final_loss_test, final_acc_test = test(model, features, adj, labels, idx_test)
    wandb.log({"final_test_loss": final_loss_test, "final_test_accuracy": final_acc_test})

if __name__ == "__main__":
    main()





    # Optional: Apply Attack
    # if run_config.get("apply_attack", False):
    #     target_node = idx_test[0].item()
    #     n_perturbations = run_config.get("n_perturbations", 3)

    #     print(f"\n[ Nettack ] Targeting node {target_node} with {n_perturbations} perturbations...")
    #     perturbed_adj, perturbed_features = apply_nettack(model, adj, features, labels, target_node, n_perturbations)

    #     model.eval()
    #     output = model(perturbed_features, perturbed_adj)
    #     pred = output[target_node].argmax().item()
    #     true = labels[target_node].item()

    #     print(f"After attack: Target node {target_node} - True: {true}, Pred: {pred}")
    #     wandb.log({"attack_target_node": target_node, "true_label": true, "predicted_label": pred})