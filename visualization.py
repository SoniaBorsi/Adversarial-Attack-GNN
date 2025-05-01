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
    if isinstance(edge_index, torch.Tensor):
        # Check if edge_index is a torch tensor and convert to numpy if so
        edge_index = edge_index.cpu().numpy()

    if edge_index.shape[0] != 2:
        # Check if edge_index is a numpy array and has the correct shape
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

    # Sanitize title for filename 
    safe_title = title.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f"{safe_title}.png")

    plt.savefig(save_path) # Save the figure
    plt.close()  # Close the plot to free memory
    print(f"    Graph saved to {save_path}\n")