import networkx as nx
import matplotlib.pyplot as plt
import torch

def visualize_graph(edge_index, title="Graph"):
    """
    Visualizes a graph given its edge index.
    Args:
        edge_index (torch.Tensor or np.ndarray): Edge index of the graph.
        title (str): Title for the plot.
    Raises:
        ValueError: If edge_index is not in the expected format.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    if edge_index.shape[0] != 2:
        raise ValueError("Expected edge_index shape [2, num_edges]")

    # Convert to list of edge tuples
    edge_list = list(zip(edge_index[0], edge_index[1]))

    G = nx.Graph()
    G.add_edges_from(edge_list)
    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=30, edge_color="gray", alpha=0.6, with_labels=False)
    plt.title(title)
    plt.show()