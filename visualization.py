import networkx as nx
import matplotlib.pyplot as plt
import torch
import os

from torch_geometric.data import Data


def visualize_graph(edge_index, title="Graph", save_dir="visuals"):
    """
    Visualizes a graph given its edge index.
    Args:
        edge_index (torch.Tensor or np.ndarray): Edge index of the graph.
        title (str): Title for the plot.
        save_dir (str): Directory to save the visualization.
    Raises:
        ValueError: If edge_index is not in the expected format.
    """
    # Check if edge_index is a torch tensor and convert to numpy if so
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Check if edge_index is a numpy array and has the correct shape
    if edge_index.shape[0] != 2:
        raise ValueError("Expected edge_index shape [2, num_edges]")

    # Convert to list of edge tuples
    edge_list = list(zip(edge_index[0], edge_index[1]))

    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Make the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=30, edge_color="gray", alpha=0.6, with_labels=False)
    plt.title(title)

    # Sanitize title for filename (optional but good practice)
    safe_title = title.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f"{safe_title}.png")

    plt.savefig(save_path) # Save the figure
    plt.close()  # Close the plot to free memory
    print(f"    Graph saved to {save_path}")


def perturbed_graph(perturbed_adj, perturbed_features, data, dataset_name):
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
    save_path = os.path.join("perturbed_data", f"{dataset_name}_perturbed.pt")

    # Save perturbed adj and features together
    torch.save({
        "perturbed_adj": perturbed_adj,
        "perturbed_features": perturbed_features,
        "labels": data.y,  # saving labels in case you want to re-train/eval later
        "train_mask": data.train_mask,
        "val_mask": data.val_mask if hasattr(data, 'val_mask') else None,
        "test_mask": data.test_mask if hasattr(data, 'test_mask') else None
    }, save_path)

    print(f"    Saved perturbed dataset to {save_path}")