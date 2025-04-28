import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    A simple GCN model for node classification.
    Args:
        num_features (int): Number of input features.
        hidden_channels (int): Number of hidden channels.
        num_classes (int): Number of output classes.
    Attributes:
        gc1 (GCNConv): First GCN layer.
        gc2 (GCNConv): Second GCN layer.
        nfeat (int): Number of input features.
        nclass (int): Number of output classes.
        hidden_sizes (list): List of hidden sizes.
        with_relu (bool): Whether to apply ReLU activation.
    Properties:
        gc1_weight (torch.Tensor): Weights of the first GCN layer.
        gc2_weight (torch.Tensor): Weights of the second GCN layer.
        weight (torch.Tensor): Weights of the first GCN layer.
    Methods:
        forward(x, edge_index): Forward pass of the model.
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices of the graph.
        Returns:
            torch.Tensor: Output node features after two GCN layers.
    """
    
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.gc1 = GCNConv(num_features, hidden_channels)
        self.gc2 = GCNConv(hidden_channels, num_classes)

        self.nfeat = num_features
        self.nclass = num_classes
        self.hidden_sizes = [hidden_channels]
        self.with_relu = True

    @property
    def gc1_weight(self):
        return self.gc1.lin.weight

    @property
    def gc2_weight(self):
        return self.gc2.lin.weight

    @property
    def weight(self):
        return self.gc1.lin.weight

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, edge_index)
        self.output = x
        return x

def train_model(model, data, epochs=100, lr=0.01, weight_decay=5e-4):
    """
    Train the GCN model on the given data.
    Args:
        model (GCN): The GCN model to train.
        data (Data): The input data containing node features and edge indices.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
    Returns:
        model (GCN): The trained GCN model.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return model