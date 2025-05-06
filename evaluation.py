import torch
import os
import constants
import numpy as np

from torch_geometric.utils import to_torch_coo_tensor
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_model(model, data, perturbed_adj, perturbed_features):

    if not perturbed_adj.is_sparse:
        # Ensure the adjacency matrix is in sparse COO format
        perturbed_adj = perturbed_adj.to_sparse()

    # Get edge_index directly from the sparse tensor (COO format)
    edge_index = perturbed_adj._indices()

    if perturbed_features is None:
        # Use original features if perturbed ones are not returned
        perturbed_x = data.x
    else:
        # Ensure features are in the correct format
        if hasattr(perturbed_features, "toarray"):
            perturbed_x = torch.tensor(perturbed_features.toarray(), dtype=torch.float)
        else:
            perturbed_x = perturbed_features

    # Forward pass through the model using the perturbed graph and features
    model.eval()
    with torch.no_grad():
        out = model(perturbed_x, edge_index)

    if data.test_mask.shape[0] != data.num_nodes:
        # Ensure test_mask is correctly sized and boolean, truncate or adjust the size
        data.test_mask = data.test_mask[:data.num_nodes]

    if len(data.test_mask.shape) > 1:
        # Flatten test_mask if it's not 1D
        data.test_mask = data.test_mask.flatten()

    # Ensure data.test_mask has the correct size and is boolean
    data.test_mask = data.test_mask[:data.num_nodes].to(torch.bool)

    # EVALUATION METRICS
    pred = out.argmax(dim=1)
    true = data.y[data.test_mask].cpu().numpy()
    pred_labels = pred[data.test_mask].cpu().numpy()

    # Accuracy
    correct = (pred_labels == true).sum().item()
    acc = correct / data.test_mask.sum().item()

    # Precision, recall, F1 score
    precision = precision_score(true, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true, pred_labels, average='macro', zero_division=0)

    return acc, precision, recall, f1


def before_attack(model, data, dataset_name):

    print("-" * 100)
    print(f"Evaluating on clean (original) graph for dataset: {dataset_name}")

    # Create PyTorch sparse adjacency matrix
    original_adj = to_torch_coo_tensor(
        data.edge_index, 
        edge_attr=None, #=torch.ones(data.edge_index.size(1)),
        size=(data.num_nodes, data.num_nodes)
    )

    acc, precision, recall, f1 = evaluate_model(model, data, original_adj, data.x)

    with open(constants.RES_PATH, "a") as f:  
        f.write("-" * 100 + "\n")
        f.write(f"Evaluation Metrics on clean {dataset_name}:\n")
        f.write(f"    Accuracy:   {acc:.4f}\n")
        f.write(f"    Precision:  {precision:.4f}\n")
        f.write(f"    Recall:     {recall:.4f}\n")
        f.write(f"    F1 Score:   {f1:.4f}\n\n")  

    print(f"    Saved clean evaluation metrics to {constants.RES_PATH}\n")

    return acc, precision, recall, f1


def after_attack(model, data, dataset_name, perturbed_adj, perturbed_features):

    acc, precision, recall, f1 = evaluate_model(model, data, perturbed_adj, perturbed_features)

    with open(constants.RES_PATH, "a") as f: 
        f.write(f"Evaluation Metrics on poisoned {dataset_name}:\n")
        f.write(f"    Accuracy:   {acc:.4f}\n")
        f.write(f"    Precision:  {precision:.4f}\n")
        f.write(f"    Recall:     {recall:.4f}\n")
        f.write(f"    F1 Score:   {f1:.4f}\n\n")  

    print(f"    Saved poisoned evaluation metrics to {constants.RES_PATH}\n")

    return acc, precision, recall, f1


def compare_results(dataset_name, acc_clean, prec_clean, rec_clean, f1_clean,
                    acc_poisoned, prec_poisoned, rec_poisoned, f1_poisoned):
    
    acc_drop = acc_clean - acc_poisoned
    asr = acc_drop / acc_clean if acc_clean > 0 else 0

    with open(constants.RES_PATH, "a") as f:  
        f.write(f"Metric Comparison: {dataset_name}:\n")
        f.write(f"    Accuracy Drop:        {acc_drop:.4f}\n")
        f.write(f"    Attack Success Rate:  {asr:.4f}\n\n")
  

    print(f"    Compared evaluation metrics to {constants.RES_PATH}\n")


def avg_std(acc_poisoned_list, prec_poisoned_list, rec_poisoned_list, f1_poisoned_list, dataset_name):
    acc_poisoned_avg = np.mean(acc_poisoned_list)
    acc_poisoned_std = np.std(acc_poisoned_list)

    prec_poisoned_avg = np.mean(prec_poisoned_list)
    prec_poisoned_std = np.std(prec_poisoned_list)

    rec_poisoned_avg = np.mean(rec_poisoned_list)
    rec_poisoned_std = np.std(rec_poisoned_list)

    f1_poisoned_avg = np.mean(f1_poisoned_list)
    f1_poisoned_std = np.std(f1_poisoned_list)

    with open(constants.RES_PATH, "a") as f:  
        f.write(f"Average and Std for {dataset_name}:\n")
        f.write(f"    Accuracy:   {acc_poisoned_avg:.4f} ± {acc_poisoned_std:.4f}\n")
        f.write(f"    Precision:  {prec_poisoned_avg:.4f} ± {prec_poisoned_std:.4f}\n")
        f.write(f"    Recall:     {rec_poisoned_avg:.4f} ± {rec_poisoned_std:.4f}\n")
        f.write(f"    F1 Score:   {f1_poisoned_avg:.4f} ± {f1_poisoned_std:.4f}\n\n")  

    print(f"    Saved average and std to {constants.RES_PATH}\n")
