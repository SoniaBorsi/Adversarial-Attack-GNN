from deeprobust.graph.global_attack import Nettack
from deeprobust.graph.utils import preprocess
import copy


def apply_nettack(model, adj, features, labels, idx_target, n_perturbations):
    """
    Apply Nettack to a given graph targeting a specific node.
    
    Parameters:
      model: A trained GNN model (e.g., a GCN) to be used as the surrogate.
      adj: The adjacency matrix (scipy sparse matrix).
      features: The feature matrix (scipy sparse or NumPy array).
      labels: The node labels (numpy array or torch tensor).
      idx_target: The target node index (or list of indices) to attack.
      n_perturbations: An integer, the number of perturbations allowed.
      
    Returns:
      perturbed_adj: The perturbed (adversarial) adjacency matrix (after attack).
      perturbed_features: Modified features (if attack_features=True; otherwise, likely unchanged).
    """
    # Preprocess the graph (usually converts features to dense and normalizes adj, etc.)
    # Note: preprocess here should output the expected format for Nettack.
    adj_norm, features_norm = preprocess(adj, features)
    
    # Ensure the model is in evaluation mode and make a copy to serve as a surrogate.
    model.eval()
    surrogate = copy.deepcopy(model)
    
    # Set device – ensure both your model and inputs are on the same device.
    device = 'cpu'  # or 'cuda' if you are using GPU. 
    surrogate.to(device)
    
    # Determine the number of nodes from the preprocessed adjacency matrix.
    nnodes = adj_norm.shape[0]
    
    # Initialize Nettack: here, attack_structure is True, attack_features is False.
    attacker = Nettack(surrogate, nnodes=nnodes, attack_structure=True, 
                         attack_features=False, device=device)
    
    # Run the attack.
    # Note: idx_target should be a single node (or list with one element).
    attacker.attack(features_norm, adj_norm, labels, idx_target, n_perturbations)
    
    # After the attack, obtain the modified adjacency and features.
    perturbed_adj = attacker.adj
    perturbed_features = attacker.modified_features  # Will be same as input if attack_features is False.
    
    return perturbed_adj, perturbed_features





