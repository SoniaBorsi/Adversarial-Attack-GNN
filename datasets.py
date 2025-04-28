import torch

from torch_geometric.datasets import Planetoid, WebKB, PolBlogs, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data

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

    if name in ["Cora", "Citeseer", "PubMed"]:
        return Planetoid(root=f"./data/{name}", name=name)[0]
    elif name == "Texas":
        return WebKB(root=f"./data/{name}", name=name)[0]
    elif name == "PolBlogs":
        return PolBlogs(root=f"./tmp/polblogs")[0]
    elif name == "ogbn-proteins":
        return PygNodePropPredDataset(root="./data/ogb", name="ogbn-proteins")[0]
    elif name == "Flickr":
        return Flickr(root=f"./data/Flickr")[0]
    else:
        raise ValueError("Dataset not found")


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
        print(f"    {dataset_name} patching missing features...")       #DEBUG
        data.x = torch.eye(data.num_nodes)

    # Check the shape of the masks and labels BEFORE: debug
    #print(f"train_mask shape: {data.train_mask.shape}")        # DEBUG
    #print(f"data.y shape: {data.y.shape}")     # DEBUG

    if not hasattr(data, 'train_mask') or data.train_mask.shape[0] != data.num_nodes:
        # dataset that doesnt come with masks ie. built-in train/test/val splits
        print(f"    {dataset_name} making masks...")        #DEBUG
        data = split_masks(data)
    else:
        if len(data.train_mask.shape) > 1:
            # If the train_mask exists but is not 1D, fix it to be
            print(f"    Fixing train_mask_shape for {dataset_name}...")     #DEBUG
            data.train_mask = data.train_mask.view(-1)

    # check train_mask is correct (Texas)
    if data.train_mask.shape[0] != data.num_nodes:
        # Truncate or adjust the size
        print(f"    Adjusting train_masks size for {dataset_name}...")      #DEBUG
        data.train_mask = data.train_mask[:data.num_nodes]

    if len(data.train_mask.shape) == 1:
        # Ensure it is boolean
        data.train_mask = data.train_mask.to(torch.bool)

    # Check the shape of the masks and labels AFTER: debug
    #print(f"train_mask shape: {data.train_mask.shape}")        # DEBUG
    #print(f"data.y shape: {data.y.shape}")     # DEBUG

    return data

