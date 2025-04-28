import torch
from torch_geometric.datasets import Planetoid, WebKB, PolBlogs, Flickr
from ogb.nodeproppred import PygNodePropPredDataset

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