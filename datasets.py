import torch
import constants
import os

from torch_geometric.datasets import Planetoid, WebKB, PolBlogs, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def load_dataset(name):
    """
    Load a dataset based on the provided name.
    Args:
        name (str): Name of the dataset to load.
    Returns:
        Data: Loaded dataset.
    Raises:
        ValueError: If the dataset name is not recognized.
    """

    if name in [constants.CORA, constants.CITESEER, constants.PUBMED]:
        data = Planetoid(root=f"./data/{name}", name=name)[0]
    elif name == constants.TEXAS:
        data = WebKB(root=f"./data/{name}", name=name)[0]
    elif name == constants.POLBLOGS:
        data = PolBlogs(root=f"./tmp/polblogs")[0]
    elif name == constants.OGBN_PROTEINS:
        data = PygNodePropPredDataset(root="./data/ogb", name="ogbn-proteins")[0]
    elif name == constants.FLICKR:
        data = Flickr(root=f"./data/Flickr")[0]
    else:
        raise ValueError("Dataset not found")

    # After loading, optionally apply subgraph extraction
    if name in [constants.FLICKR, constants.OGBN_PROTEINS, constants.PUBMED]:
        print(f"Extracting subgraph for {name}...")
        data = extract_subgraph(data, 
            num_hops=2, 
            min_num_nodes=3000, 
            num_center_nodes=10,
            max_attempts=10
        )

    return data


def split_masks(data, train_ratio=0.6, val_ratio=0.2):
    """
    Split the dataset into train, validation, and test masks.
    Args:
        data (Data): The input data containing node features and edge indices.
        train_ratio (float): Ratio of nodes to be used for training.
        val_ratio (float): Ratio of nodes to be used for validation.
    Returns:
        Data: The input data with updated train, val, and test masks.
    """
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size + val_size]] = True
    data.test_mask[indices[train_size + val_size:]] = True

    return data

def patch_data(dataset, dataset_name):
    """
    Patch the dataset if needed.
    Args:
        dataset (Data): The input data containing node features and edge indices.
        dataset_name (str): Name of the dataset to patch.
    Returns:
        Data: The patched dataset.
    """
    
    if isinstance(dataset, Data):
        data = dataset  # Already a Data object
    else:
        data = dataset[0]  # Dataset is a list-like object (like Planetoid)

    if data.x is None:
        # Patching missing features (for PolBlogs, speficially)
        # All zero-features carry no useful info -> poor performance,
        # so each node has a unique identifier (one-hot encoding). 
        # GNN can differentiate them based on graph structure + node identity
        data.x = torch.eye(data.num_nodes)

    if not hasattr(data, 'train_mask') or data.train_mask.shape[0] != data.num_nodes:
        # dataset that doesnt come with masks ie. built-in train/test/val splits
        data = split_masks(data)
    else:
        if len(data.train_mask.shape) > 1:
            # If the train_mask exists but is not 1D, fix it to be
            data.train_mask = data.train_mask.view(-1)

    if data.train_mask.shape[0] != data.num_nodes:
        # check train_mask is correct (Texas)
        # Truncate or adjust the size
        data.train_mask = data.train_mask[:data.num_nodes]

    if len(data.train_mask.shape) == 1:
        # Ensure it is boolean
        data.train_mask = data.train_mask.to(torch.bool)

    print(f"Data.x shape: {data.x.shape}, Data.train_mask shape: {data.train_mask.shape}")

    return data

def extract_subgraph(data, num_hops, min_num_nodes, num_center_nodes, max_attempts):
    """
    Extract a subgraph from the dataset.
    Args:
        data (Data): The input data containing node features and edge indices.
        num_hops (int): Number of hops to consider for subgraph extraction.
        min_num_nodes (int): Minimum number of nodes required in the subgraph.
        num_center_nodes (int): Number of center nodes to sample for subgraph extraction.
        max_attempts (int): Maximum number of attempts to find a valid subgraph.
    Returns:
        Data: The extracted subgraph.
    """
    
    device = data.x.device
    attempt = 0

    while attempt < max_attempts:
        center_nodes = torch.randint(0, data.num_nodes, (num_center_nodes,), device=device)

        subset_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_nodes,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True
        )

        print(f"Attempt {attempt + 1}: Found {subset_nodes.size(0)} nodes.")  # DEBUG

        if subset_nodes.size(0) >= min_num_nodes:
            break  # Good enough!
        
        attempt += 1
        num_center_nodes = int(num_center_nodes * 1.5)  # Try sampling more centers next round

    # No slicing! Keep all nodes and edges found.
    sub_x = data.x[subset_nodes]
    sub_y = data.y[subset_nodes]

    # Relabel labels to be contiguous
    unique_labels = sub_y.unique()
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    sub_y = torch.tensor([label_map[label.item()] for label in sub_y], device=sub_y.device)

    subgraph = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        y=sub_y
    )

    return subgraph

def perturbed_dataset(perturbed_adj, perturbed_features, data, dataset_name, num_perturbations):
    """
    Save the perturbed dataset.
    Args:
        perturbed_adj (torch.Tensor): The perturbed adjacency matrix.
        perturbed_features (torch.Tensor): The perturbed node features.
        data (Data): The original data object containing node features and edge indices.
        dataset_name (str): Name of the dataset to save the perturbed data for.
    """
    # Save the perturbed dataset
    os.makedirs("perturbed_data", exist_ok=True)  # create a folder if it doesn't exist
    save_path = os.path.join("perturbed_data", f"{dataset_name}_perturbed{num_perturbations}.pt")

    # Save perturbed adj and features together
    torch.save({
        "perturbed_adj": perturbed_adj,
        "perturbed_features": perturbed_features,
        "labels": data.y,  # saving labels in case you want to re-train/eval later
        "train_mask": data.train_mask,
        "val_mask": data.val_mask if hasattr(data, 'val_mask') else None,
        "test_mask": data.test_mask if hasattr(data, 'test_mask') else None
    }, save_path)

    print(f"\nSaved perturbed dataset to {save_path}\n")