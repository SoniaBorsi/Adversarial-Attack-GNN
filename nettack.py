import torch
import numpy as np
from scipy.sparse import csr_matrix
from deeprobust.graph.targeted_attack.nettack import Nettack
from torch_geometric.utils import to_scipy_sparse_matrix

def apply_nettack(model, data, target_node, n_perturbations,
                  attack_structure=True, attack_features=False,
                  device='cpu'):
    """
    Apply Nettack to one node and return a new Data with perturbations.
    """

    # 1) Build SciPy-sparse graph & features
    adj      = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    features = csr_matrix(data.x.numpy())   # sparse features
    labels   = data.y.numpy()

    # 2) Monkey‑patch so Deeprobust finds weight matrices with the right dims:
    #    A should be (in_feats, hidden); B should be (hidden, out_feats)
    if hasattr(model, 'gc1') and hasattr(model.gc1, 'lin'):
        model.gc1.weight = model.gc1.lin.weight.T    # (in_feats, hidden)
    if hasattr(model, 'gc2') and hasattr(model.gc2, 'lin'):
        model.gc2.weight = model.gc2.lin.weight.T    # (hidden, out_feats)

    # 3) (Re)train the surrogate on the clean graph so the weights aren't random
    #    ─ important for the attack to be meaningful:
    model = model.to(device)
    model.eval()  # assume you already trained before calling this

    # 4) Initialize and run the Nettack attack
    attacker = Nettack(
        model,
        nnodes=data.num_nodes,
        attack_structure=attack_structure,
        attack_features=attack_features,
        device=device
    )
    attacker.attack(
        features,
        adj,
        labels,
        target_node,
        n_perturbations
    )

    # 5) Extract the perturbed graph & features
    mod_adj      = attacker.modified_adj
    mod_features = attacker.modified_features

    # 6) Build a new Data object
    new_data = data.clone()
    coo = mod_adj.tocoo()
    new_data.edge_index = torch.LongTensor([coo.row, coo.col])

    # 7) Dense‑ify features if needed
    if hasattr(mod_features, "toarray"):
        arr = mod_features.toarray()
    else:
        arr = mod_features
    new_data.x = torch.FloatTensor(arr)

    return new_data
