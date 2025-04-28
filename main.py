import torch
import torch_geometric
import deeprobust
import os

import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import torch.nn.functional as F
import numpy as np

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph, to_torch_coo_tensor
from torch_geometric.data import Data
from torch_sparse import from_scipy

# Import the other files
from datasets import load_dataset, split_masks, patch_data
from model import GCN, train_model
from metattack import apply_metattack
from evaluation import evaluate_model, before_attack, after_attack
from visualization import visualize_graph, perturbed_graph

def run_experiment(dataset_name):
    """
    Run the experiment for a given dataset.
    Args:
        dataset_name (str): Name of the dataset to run the experiment on.
    """

    print("*" * 100) # Separate the dataset run 
    dataset = load_dataset(dataset_name)
    data = patch_data(dataset, dataset_name)

    num_classes = len(torch.unique(data.y))

    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"    Number of nodes: {data.num_nodes}")
    print(f"    Number of features: {data.num_features}")
    print(f"    Number of classes: {num_classes}\n")

    model = GCN(data.num_features, 16, num_classes)

    # Save the trained model path
    os.makedirs("trained_models", exist_ok=True)  # Make sure the folder exists
    model_save_path = os.path.join("trained_models", f"{dataset_name}_model.pt")
    
    if os.path.exists(model_save_path):
        # If the model already exists, load it
        print(f"Model already exists at {model_save_path}. Loading the model.")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
    else:
        # If the model does not exist, train it
        print(f"Saving mosdel to {model_save_path}")
        model = train_model(model, data)
        torch.save(model.state_dict(), model_save_path)
        print(f"Trained new model and saved to {model_save_path}\n")

    # Evaluate the model on the original graph
    acc_clean, precision_clean, recall_clean, f1_clean = before_attack(model, data, dataset_name)
    
    if data.num_nodes <= 5000:
        # Visualize the original graph if it is small enough
        print("-" * 100)
        print("Original Graph")
        visualize_graph(data.edge_index, title=f"{dataset_name}: Before Metattack", save_dir="visuals")

    # TODO: increase num_perturbations for larger datasets
    # For larger datasets, use a smaller percentage of edges to perturb
    # num_perturbations = int(data.edge_index.size(1) * 0.01)
    num_perturbations = 5 # Base code for now
    print(f"    Running Metattack on {num_perturbations} perturbations")

    # Apply Metattack
    perturbed_adj, perturbed_features = apply_metattack(model, data, num_perturbations=num_perturbations)
    edge_index_perturbed = perturbed_adj.nonzero().t()

    # Save the perturbed dataset
    perturbed_graph(perturbed_adj, perturbed_features, data, dataset_name)

    # Perturbed stuff
    if data.num_nodes <= 5000:
        # Visualize the perturbed graph if it is small enough
        print("-" * 100)
        print("Perturbed Graph")
        visualize_graph(edge_index_perturbed, title=f"{dataset_name}: After Metattack", save_dir="visuals")
    
    # Evaluate the model on the perturbed graph
    after_attack(model, data, dataset_name, perturbed_adj, perturbed_features)


if __name__ == "__main__": 
    """
    Run the experiment for all datasets.
    Uncomment the datasets you want to run.
    If you want to run all datasets, uncomment the line below.
    Note: Some datasets may take a long time to run or may require more memory.
    """
    # all datasets, uncomment when debugged all
    #datasets = ["Cora", "Citeseer", "PolBlogs", "Texas", "Flickr", "PubMed", "ogbn-proteins"]

    # For now, core datasets 
    datasets = ["Cora", "Citeseer", "PolBlogs", "Texas"]

    # TODO: debug these datasets
    # These datasets crash - reason: too big, use all available RAM
    #datasets = [ "ogbn-proteins", "Flickr", "PubMed"]

    # For debugging purposes, uncomment the dataset you want to run
    #datasets = ["Cora"]
    #datasets = ["Citeseer"]
    #datasets = ["PolBlogs"]
    #datasets = ["Texas"]
    #datasets = ["Flickr"]
    #datasets = ["PubMed"]
    #datasets = ["ogbn-proteins"]

    for dataset_name in datasets:
        run_experiment(dataset_name)