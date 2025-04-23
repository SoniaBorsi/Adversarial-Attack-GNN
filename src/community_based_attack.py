import os
import argparse
import random
from pathlib import Path

import networkx as nx
from networkx.algorithms import community
import numpy as np


def load_graph_and_features(datadir):
    base = Path(datadir)
    if (base / "cora").exists():
        folder = base / "cora"
        edge_file = folder / "cora.cites"
        feat_file = folder / "cora.content"
        sep = '\t'
    elif (base / "citeseer").exists():
        folder = base / "citeseer"
        edge_file = folder / "citeseer.cites"
        feat_file = folder / "citeseer.content"
        sep = '\t'
    else:
        raise ValueError("Unsupported dataset")

    G = nx.Graph()
    with open(edge_file) as f:
        for line in f:
            u, v = line.strip().split(sep)[:2]
            G.add_edge(u, v)

    X = {}
    with open(feat_file) as f:
        for line in f:
            parts = line.strip().split(sep)
            X[parts[0]] = np.array(list(map(float, parts[1:-1])))

    return G, X, folder


def detect_communities(G):
    return list(community.louvain_communities(G, seed=42))


def pick_influential_nodes(G, comms):
    influential = []
    for comm in comms:
        subG = G.subgraph(comm)
        centrality = nx.degree_centrality(subG)
        top_node = max(centrality, key=centrality.get)
        influential.append(top_node)
    return influential


def apply_joint_perturbations(G, targets, X, budget):
    Gp = G.copy()
    perturbations = 0
    
    while perturbations < budget:
        target = random.choice(targets)
        neighbors = list(Gp.neighbors(target))
        non_neighbors = list(set(Gp.nodes()) - set(neighbors) - {target})
        
        if random.random() < 0.5 and neighbors:
            # Remove a random neighbor
            u = random.choice(neighbors)
            Gp.remove_edge(target, u)
        elif non_neighbors:
            # Add an edge to a random non-neighbor
            u = random.choice(non_neighbors)
            Gp.add_edge(target, u)
        perturbations += 1

        # Feature perturbation (random noise)
        if target in X:
            X[target] = np.roll(X[target], 1)

    return Gp, X


def save_perturbed(Gp, X, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "perturbed_edges.csv", 'w') as f:
        for u, v in Gp.edges():
            f.write(f"{u},{v}\n")
    with open(outdir / "perturbed_content.csv", 'w') as f:
        for n, feats in X.items():
            feat_str = ','.join(map(str, feats))
            f.write(f"{n},{feat_str}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset folder (cora/citeseer)")
    parser.add_argument("--budget", type=int, default=50, help="Total perturbation budget")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    G, X, folder = load_graph_and_features(args.data)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    comms = detect_communities(G)
    print(f"Detected {len(comms)} communities")

    influential_nodes = pick_influential_nodes(G, comms)
    print("Influential nodes selected:", influential_nodes)

    Gp, Xp = apply_joint_perturbations(G, influential_nodes, X, args.budget)
    print(f"Perturbations applied with budget {args.budget}")

    save_perturbed(Gp, Xp, folder / "perturbed")
    print(f"Perturbed data saved to {folder / 'perturbed'}")


if __name__ == "__main__":
    main()