from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def evaluate_model(model, data, perturbed_adj, perturbed_features):
    """
    Evaluate the model on the perturbed graph and features.
    Args:
        model (torch.nn.Module): The trained model.
        data (Data): The input data containing node features and edge indices.
        perturbed_adj (torch.Tensor): Perturbed adjacency matrix.
        perturbed_features (torch.Tensor): Perturbed node features.
    Returns:
        tuple: Accuracy, precision, recall, and F1 score.
    """

    # Check the type of perturbed_adj
    #print(f"perturbed_adj type: {type(perturbed_adj)}")        # DEBUG

    # Ensure the adjacency matrix is in sparse COO format
    if not perturbed_adj.is_sparse:
        perturbed_adj = perturbed_adj.to_sparse()

    # Get edge_index directly from the sparse tensor (COO format)
    edge_index = perturbed_adj._indices()

    # Use original features if perturbed ones are not returned
    if perturbed_features is None:
        perturbed_x = data.x
    else:
        if hasattr(perturbed_features, "toarray"):
            perturbed_x = torch.tensor(perturbed_features.toarray(), dtype=torch.float)
        else:
            perturbed_x = perturbed_features

    # Forward pass through the model using the perturbed graph and features
    model.eval()
    with torch.no_grad():
        out = model(perturbed_x, edge_index)

    # Ensure test_mask is correctly sized and boolean
    if data.test_mask.shape[0] != data.num_nodes:
        print(f"Adjusting test_mask size for {data.dataset_name}...")   #DEBUG
        # Truncate or adjust the size
        data.test_mask = data.test_mask[:data.num_nodes]

    # Flatten test_mask if it's not 1D
    if len(data.test_mask.shape) > 1:
        data.test_mask = data.test_mask.flatten()

    # Ensure data.test_mask has the correct size and is boolean
    data.test_mask = data.test_mask[:data.num_nodes].to(torch.bool)

    #if len(data.test_mask.shape) == 1:
        # Ensure it is boolean
        #data.test_mask = data.test_mask.to(torch.bool)

    # Check the shapes of predictions and ground truth labels
    print(f"Prediction shape: {out.shape}")                 # DEBUG
    print(f"test_mask shape: {data.test_mask.shape}")       # DEBUG
    print(f"Ground truth labels shape: {data.y.shape}")     # DEBUG

    # Ensure data.y has the same size as test_mask
    assert data.y.shape[0] == data.num_nodes, "data.y shape mismatch with num_nodes"

    # EVALUATION METRICS
    pred = out.argmax(dim=1)
    true = data.y[data.test_mask].cpu().numpy()
    pred_labels = pred[data.test_mask].cpu().numpy()

    # Accuracy
    correct = (pred_labels == true).sum().item()
    acc = correct / data.test_mask.sum().item()

    # Precision, recall, F1 score
    precision = precision_score(true, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true, pred_labels, average='macro', zero_division=0)

    return acc, precision, recall, f1