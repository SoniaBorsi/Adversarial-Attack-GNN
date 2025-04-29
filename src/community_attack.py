#!/usr/bin/env python
"""
Strong‑Unnoticeable Attack (generic version)
• Downloads/organises every dataset under ``<root>/<dataset>/`` just like Cora/Citeseer/PubMed.
• Auto‑creates a sibling ``perturbed/`` folder where the attacked graph gets written.
• Supports: Planetoid (cora, citeseer, pubmed), WebKB‑texas, PolBlogs.
• Handles feature‑less graphs by falling back to an identity matrix.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import argparse, time, urllib.error

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB, PolBlogs
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_geometric.data import download_url

# ────────────────────────────────────────────────────────────────────
# Helper: robust WebKB download (handles occasional GitHub 429)
# ────────────────────────────────────────────────────────────────────

def _safe_webkb(root: str, name: str = "Texas", retries: int = 5):
    """Retry with exponential back‑off if GitHub rate‑limits (HTTP 429)."""
    raw_urls = [
        f"https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/{name.lower()}/out1_node_feature_label.txt",
        f"https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/{name.lower()}/out1_graph_edges.txt",
    ] + [
        f"https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits/{name.lower()}_split_0.6_0.2_{i}.npz"
        for i in range(10)
    ]
    raw_root = Path(root) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for url in raw_urls:
        fname = raw_root / Path(url).name
        if fname.exists():
            continue
        for attempt in range(retries):
            try:
                download_url(url, raw_root)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    wait = 60 * (attempt + 1)
                    print(f"GitHub 429 — sleeping {wait}s then retrying …")
                    time.sleep(wait)
                else:
                    raise
    return WebKB(root, name=name)

# ────────────────────────────────────────────────────────────────────
# Dataset loader (now guarantees  <root>/<dataset>/ hierarchy)
# ────────────────────────────────────────────────────────────────────

def load_graph(root: str, name: str):
    """Return (X, y, E, data) for any supported dataset.

    • Downloads into **root/dataset/** (lower‑case) to keep everything tidy.
    • If a graph has no features (e.g. PolBlogs) → use identity matrix.
    """
    name_l   = name.lower()
    ds_dir   = Path(root) / name_l   # e.g. data/cora , data/polblogs …
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Select + download ----------------------------------------------------
    if name_l in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(str(ds_dir), name=name_l.capitalize(), split="public")
    elif name_l == "texas":
        ds = _safe_webkb(str(ds_dir), name="Texas")
    elif name_l == "polblogs":
        ds = PolBlogs(str(ds_dir))
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = ds[0].cpu()

    # Features -------------------------------------------------------------
    if getattr(data, "x", None) is None or data.x.numel() == 0:
        X_np = np.eye(data.num_nodes, dtype=np.float32)
    else:
        X_np = data.x.numpy()

    # Adjacency ------------------------------------------------------------
    edge_index = to_undirected(data.edge_index)

    return (
        X_np,
        data.y.numpy(),
        edge_index.numpy().T,  # shape (|E|, 2)
        data,
    )

# ────────────────────────────────────────────────────────────────────
# Graph helpers / surrogate
# ────────────────────────────────────────────────────────────────────

def nx_from_edges(n: int, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    return G

class OneLayerGCN(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = GCNConv(d_in, d_out, add_self_loops=True)
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# -------------------------------------------------------------------

def surrogate_loss_delta(X: np.ndarray, y: np.ndarray, G: nx.Graph, swap: Tuple[int, int, int]) -> float:
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)
    model = OneLayerGCN(x.size(1), y_t.max().item() + 1)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train(); opt.zero_grad()
    F.nll_loss(F.log_softmax(model(x, edge_index), dim=1), y_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        base  = model(x, edge_index)
        ei_sw = edge_index.clone()
        mask  = ~(((ei_sw[0]==swap[0])&(ei_sw[1]==swap[1])) | ((ei_sw[0]==swap[1])&(ei_sw[1]==swap[0])))
        ei_sw = torch.cat([ei_sw[:, mask], torch.tensor([[swap[0], swap[1]], [swap[2], swap[0]]], dtype=torch.long)], dim=1)
        delta = F.nll_loss(F.log_softmax(model(x, ei_sw), dim=1), y_t) - F.nll_loss(F.log_softmax(base, dim=1), y_t)
    return float(delta)

# -------------------------------------------------------------------

def attack(
    X: np.ndarray,
    edges: np.ndarray,
    labels: np.ndarray,
    budget: int,
    *,
    p_surrogate: float = 0.3,
    topk_hubs: int = 300,
    flip_features: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng, G = np.random.default_rng(seed), nx_from_edges(len(labels), edges)
    hubs    = sorted(nx.pagerank(G, alpha=0.85), key=lambda v: -nx.pagerank(G)[v])[: min(topk_hubs, len(G))]
    hub_set = set(hubs)

    def is_bridge(u, v, _G=G, _cache={}):
        if _cache.get('ctr', 0) % 200 == 0:
            _cache['B'] = set(nx.bridges(_G))
        _cache['ctr'] = _cache.get('ctr', 0) + 1
        return (u, v) in _cache['B'] or (v, u) in _cache['B']

    steps = 0
    while steps < budget:
        u          = rng.choice(hubs)
        neigh_same = [v for v in G[u] if labels[v] == labels[u]]
        if not neigh_same:
            continue
        v_del = rng.choice(neigh_same)
        cand  = [w for w in hub_set if labels[w] != labels[u] and w not in G[u]]
        if not cand:
            continue
        w_add = rng.choice(cand)
        if is_bridge(u, v_del) or G.has_edge(u, w_add):
            continue
        if len({labels[x] for x in G[w_add]} | {labels[u]}) == len(np.unique(labels)):
            continue
        if rng.random() < p_surrogate and surrogate_loss_delta(X, labels, G, (u, v_del, w_add)) <= 0:
            continue
        G.remove_edge(u, v_del); G.add_edge(u, w_add); steps += 1

    new_edges = np.array(G.edges(), dtype=np.int64)
    if flip_features:
        rare_cols = np.where(X.sum(0) < 10)[0]
        for u in hubs:
            if rare_cols.size == 0:
                break
            X[u, rng.choice(rare_cols)] ^= 1
    return new_edges, X

# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------

def save_perturbed(out_dir: Path, feats: np.ndarray, labels: np.ndarray, edges: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "perturbed_edges.csv", edges, fmt="%d", delimiter=",")
    with open(out_dir / "perturbed_content.csv", "w") as f:
        for idx, (feat, lab) in enumerate(zip(feats, labels)):
            f.write(f"{idx},{','.join(map(str, feat))},{lab}\n")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cora", "citeseer", "pubmed", "texas", "polblogs"], default="cora")
    p.add_argument("--root", default="data")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk_hubs", type=int, default=300)
    p.add_argument("--p_surrogate", type=float, default=0.4)
    args = p.parse_args()

    X, y, E, _ = load_graph(args.root, args.dataset)
    new_E, new_X = attack(
        X=X.copy(), edges=E, labels=y, budget=args.budget, p_surrogate=args.p_surrogate,
        topk_hubs=args.topk_hubs, flip_features=True, seed=args.seed,
    )
    out_path = Path(args.root) / args.dataset / "perturbed"
    save_perturbed(out_path, new_X, y, new_E)
    print(f"Perturbed graph written to {out_path} (|E| {len(E)} → {len(new_E)})")

if __name__ == "__main__":
    main()



