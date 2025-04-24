import torch 
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import Metattack
from deeprobust.graph.utils import preprocess, accuracy
from deeprobust.graph.global_attack import GCN
from torch_geometric.datasets import Planetoid, WebKB, OGB_MAG
from torch_geometric.utils import to_scipy_sparse_matrix


# Load datasets
def load_dataset(name):
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=f"data/{name}", name=name)
    elif name in ['Texas', 'Polblogs']:
        dataset = WebKB(root=f"data/{name}", name=name)
    elif name == 'ogbn-proteins':
        dataset = OGB_MAG(root="data/ogbn-proteins", name=name)
    else:
        raise ValueError(f"Dataset {name} not supported.")
    return dataset[0]

# Prepare data for Metattack
def prepare_data(data):
    adj = to_scipy_sparse_matrix(data.edge_index).tocoo()
    features = data.x.numpy()
    labels = data.y.numpy()
    idx_train = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    idx_val = data.val_mask.nonzero(as_tuple=True)[0].numpy()
    idx_test = data.test_mask.nonzero(as_tuple=True)[0].numpy()
    return adj, features, labels, idx_train, idx_val, idx_test

# Perform Metattack
def run_metattack(dataset_name):
    print(f"Running Metattack on {dataset_name}...")
    data = load_dataset(dataset_name)
    adj, features, labels, idx_train, idx_val, idx_test = prepare_data(data)

    # Preprocess data
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # Initialize Metattack
    model = Metattack(model=None, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device='cpu')
    model = model.to('cpu')

    # Train Metattack
    model.attack(features, adj, labels, idx_train, idx_val, perturbations=100)

    # Get perturbed adjacency matrix
    modified_adj = model.modified_adj

    print("Metattack completed.")
    return modified_adj

# Example usage
datasets = ['Cora', 'Citeseer', 'Pubmed', 'Polblogs', 'Texas', 'ogbn-proteins']
for dataset_name in datasets:
    modified_adj = run_metattack(dataset_name)
    print(f"Modified adjacency matrix for {dataset_name} obtained.")