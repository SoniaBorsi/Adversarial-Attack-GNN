import torch
import torch_geometric
import deeprobust

import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import torch.nn.functional as F
import numpy as np

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from torch_geometric.data import Data
from torch_sparse import from_scipy

# Import the other files
from datasets import load_dataset, split_masks
from model import GCN, train_model
from metattack import apply_metattack
from evaluation import evaluate_model
from visualization import visualize_graph

'''
HOW TO RUN:
1. ssh into glogin: 
    ssh <your_pid>@glogin.cs.vt.edu

2. Download and create a conda environment: 
    wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
    chmod +x Anaconda3-2023.03-Linux-x86_64.sh
    ./Anaconda3-2023.03-Linux-x86_64.sh
    source ~/.bashrc
    conda --version
    conda create --name huggingface_env python=3.8
    source activate huggingface_env
    pip install torch diffusers transformers datasets accelerate

3. Install the required packages: 

    pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
    pip install deeprobust
    pip install networkx matplotlib
    pip install ogb

4. Run the script: 

    python main.py

This will run the experiment on the specified datasets and print the evaluation metrics after applying Metattack.

NOTE: 
    After you run the script, there should be a folder called "data" will contain the Cora, Citeseer, 
    and Texas datasets.
    Another folder called "tmp" will contain the PolBlogs dataset.
'''

'''
To check which specs you have:

    python -c "import torch; print(torch.__version__)"
    conda --version

my specs: 
    2.4.1+cu121
    conda 23.1.0
'''

def run_experiment(dataset_name):
    """
    Run the experiment for a given dataset.
    Args:
        dataset_name (str): Name of the dataset to run the experiment on.
    """
    # Separate the dataset run 
    print("\n*\n" * 100)
    dataset = load_dataset(dataset_name)

    if isinstance(dataset, Data):
        # Already a Data object
        data = dataset
    else:
        # Dataset is a list-like object (like Planetoid)
        data = dataset[0]

    if data.x is None:
        # Patching missing features (for PolBlogs, speficially)
        # All zero-features carry no useful info -> poor performance,
        # so each node has a unique identifier (one-hot encoding). 
        # GNN can differentiate them based on graph structure + node identity
        print(f"{dataset_name} patching missing features...")       #DEBUG
        data.x = torch.eye(data.num_nodes)

    # Check the shape of the masks and labels BEFORE: debug
    #print(f"train_mask shape: {data.train_mask.shape}")        # DEBUG
    #print(f"data.y shape: {data.y.shape}")     # DEBUG

    if not hasattr(data, 'train_mask') or data.train_mask.shape[0] != data.num_nodes:
        # dataset that doesnt come with masks ie. built-in train/test/val splits
        print(f"{dataset_name} making masks...")        #DEBUG
        data = split_masks(data)
    else:
        if len(data.train_mask.shape) > 1:
            # If the train_mask exists but is not 1D, fix it to be
            print(f"Fixing train_mask_shape for {dataset_name}...")     #DEBUG
            data.train_mask = data.train_mask.view(-1)

    # check train_mask is correct (Texas)
    if data.train_mask.shape[0] != data.num_nodes:
        # Truncate or adjust the size
        print(f"Adjusting train_masks size for {dataset_name}...")      #DEBUG
        data.train_mask = data.train_mask[:data.num_nodes]

    if len(data.train_mask.shape) == 1:
        # Ensure it is boolean
        data.train_mask = data.train_mask.to(torch.bool)

    # Check the shape of the masks and labels AFTER: debug
    #print(f"train_mask shape: {data.train_mask.shape}")        # DEBUG
    #print(f"data.y shape: {data.y.shape}")     # DEBUG

    num_classes = len(torch.unique(data.y))

    # Get the information about the dataset
    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"    Number of nodes: {data.num_nodes}")
    print(f"    Number of features: {data.num_features}")
    print(f"    Number of classes: {num_classes}\n")

    model = GCN(data.num_features, 16, num_classes)
    model = train_model(model, data)

    if data.num_nodes <= 5000:
        print("-" * 100)
        print("Original Graph")
        visualize_graph(data.edge_index, title=f"{dataset_name}: Before Metattack")

    # TODO: increase num_perturbations for larger datasets
    # For larger datasets, use a smaller percentage of edges to perturb
    # num_perturbations = int(data.edge_index.size(1) * 0.01)
    num_perturbations = 5 # Base code for now
    print(f"Running Metattack on {num_perturbations} perturbations")

    # Apply Metattack
    perturbed_adj, perturbed_features = apply_metattack(model, data, num_perturbations=num_perturbations)
    edge_index_perturbed = perturbed_adj.nonzero().t()

    # Perturbed stuff
    if data.num_nodes <= 5000:
        print("-" * 100)
        print("Perturbed Graph")
        visualize_graph(edge_index_perturbed, title=f"{dataset_name}: After Metattack")

    # Print metrics
    acc, precision, recall, f1 = evaluate_model(model, data, perturbed_adj, perturbed_features)
    print("-" * 100)
    print(f"Evaluation Metrics after Metattack on {dataset_name}:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall   : {recall:.4f}")
    print(f"    F1 Score : {f1:.4f}")


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