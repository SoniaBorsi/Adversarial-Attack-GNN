import torch
import scipy.sparse as sparse
from deeprobust.graph.global_attack import Metattack
from torch_geometric.utils import to_scipy_sparse_matrix


def apply_metattack(model, data, num_perturbations):
    """
    Apply Metattack to the given model and data.
    Args:
        model (torch.nn.Module): The trained model.
        data (Data): The input data containing node features and edge indices.
        num_perturbations (int): Number of perturbations to apply.
    Returns:
        tuple: Modified adjacency matrix and features.
    """
    
    # Convert adjacency matrix and features to sparse format
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features = sparse.csr_matrix(data.x.numpy())
    labels = data.y.numpy()

    # Get train and unlabeled indices
    idx_train = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    idx_unlabeled = (~data.train_mask).nonzero(as_tuple=True)[0].cpu().numpy()

    # Initialize Metattack
    attacker = Metattack(
        model=model,
        nnodes=data.num_nodes,
        attack_structure=True,
        attack_features=False,
        device='cpu'
    )

    # Perform attack
    attacker.attack(
        features,
        adj,
        labels,
        idx_train,
        idx_unlabeled,
        n_perturbations=num_perturbations,
        ll_constraint=False  # This avoids the sparse tensor issue
    )

    # Ensure `modified_adj` and `modified_features` exist and return them
    modified_adj = attacker.modified_adj
    modified_features = attacker.modified_features

    # If Metattack doesn't return features, use the original
    if modified_features is None:
        modified_features = data.x

    return attacker.modified_adj, attacker.modified_features
