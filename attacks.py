from deeprobust.graph.global_attack import Nettack
from deeprobust.graph.utils import preprocess
import torch
import warnings

def nettack(model, adj, features, labels, idx_target, n_perturbations):
    # Preprocess the graph
    adj, features = preprocess(adj, features)

    # Clone model
    model.eval()
    surrogate = model

    # Setup attack
    attacker = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu')
    attacker.attack(features, adj, labels, idx_target, n_perturbations)

    perturbed_adj = attacker.adj
    perturbed_features = attacker.modified_features

    return perturbed_adj, perturbed_features
