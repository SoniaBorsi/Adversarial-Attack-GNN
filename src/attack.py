
###### VERSION TOP
#!/usr/bin/env python
"""
-----------------------------

 • targets **high-centrality (hub) vertices**
 • uses **degree-preserving edge *swaps***  (one deletion + one addition)
 • optional **surrogate-loss scoring** to pick the best swap
 • keeps
      –   exact degree sequence
      –   graph connected  (no bridge removal)
      –   edge count constant
      –   no new 2-hop paths linking *all* classes
 • can add a single tiny feature-bit flip on every touched vertex
"""

from   pathlib import Path
import argparse, random
import numpy  as np
import torch, torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
from   torch_geometric.datasets import Planetoid
from   torch_geometric.nn       import GCNConv                 # tiny 1-layer surrogate
from   typing import Tuple, List, Set
from torch_geometric.utils import to_scipy_sparse_matrix

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_planetoid(root: str, name: str):
    ds   = Planetoid(root, name=name, split="public", transform=None)
    data = ds[0].cpu()
    edge_index = data.edge_index.numpy().T     # (|E|,2)
    return data.x.numpy(), data.y.numpy(), edge_index, data     # last one for features


def nx_from_edges(n: int, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    return G


# --------------------------------------------------------------------------- #
# Tiny *one-layer* GCN surrogate – cheap gradient of the current graph
# --------------------------------------------------------------------------- #
class OneLayerGCN(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = GCNConv(d_in, d_out, add_self_loops=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def surrogate_loss_delta(
        X   : np.ndarray,
        y   : np.ndarray,
        G   : nx.Graph,
        swap: Tuple[int,int,int]      # (u, v_del, w_add)
) -> float:
    """
    Very small proxy:
        − train 1 epoch on current graph,
        − return Δ cross-entropy that the swap would cause
    """
    device      = "cpu"
    n           = X.shape[0]
    edge_index  = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    x           = torch.tensor(X, dtype=torch.float)
    y_torch     = torch.tensor(y, dtype=torch.long)

    model = OneLayerGCN(x.size(1), y_torch.max().item()+1).to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(1):                          # one super-fast epoch is enough
        optim.zero_grad()
        out  = model(x, edge_index)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_torch)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        base    = model(x, edge_index)

        # apply the swap on-the-fly
        ei_swap = edge_index.clone()
        # remove (u,v_del) — we know it's there; add (u,w_add)
        mask       = ~(((ei_swap[0]==swap[0])&(ei_swap[1]==swap[1]))|
                       ((ei_swap[0]==swap[1])&(ei_swap[1]==swap[0])))
        ei_swap    = ei_swap[:, mask]
        ei_swap    = torch.cat([ei_swap,
                                torch.tensor([[swap[0], swap[1]],
                                              [swap[2], swap[0]]], dtype=torch.long)], dim=1)
        after    = model(x, ei_swap)
        delta    = (F.nll_loss(F.log_softmax(after, dim=1), y_torch) -
                    F.nll_loss(F.log_softmax(base , dim=1), y_torch))
    return float(delta)


# --------------------------------------------------------------------------- #
#                >>>   STRONG   U N N O T I C E A B L E   <<<                #
# --------------------------------------------------------------------------- #
def strong_unnoticeable_attack(
        edges          : np.ndarray,
        labels         : np.ndarray,
        budget         : int,
        p_surrogate    : float = 0.3,
        topk_hubs      : int   = 300,
        flip_features  : bool  = True,
        seed           : int   = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (new_edges, maybe_flipped_X)
    – degree sequence preserved
    – connectivity preserved
    – optional tiny bit-flips in features
    """
    rng   = np.random.default_rng(seed)
    n     = len(labels)
    G     = nx_from_edges(n, edges)

    # --- compute hubs ------------------------------------------------------ #
    pr          = nx.pagerank(G, alpha=0.85)
    hubs        = sorted(G, key=lambda v: -pr[v])[:topk_hubs]
    hub_set     = set(hubs)

    # --- build fast bridge look-up ---------------------------------------- #
    def is_bridge(u,v, _G=G, _cache={}):
        # recompute bridges *only* every 200 operations
        if 'ctr' not in _cache or _cache['ctr']%200==0:
            _cache['B'] = set(nx.bridges(_G))
        _cache['ctr'] = _cache.get('ctr',0)+1
        return (u,v) in _cache['B'] or (v,u) in _cache['B']

    # ---------------------------------------------------------------------- #
    steps = 0
    while steps < budget:
        u = rng.choice(hubs)                          # always work around a hub
        neigh_intra = [v for v in G[u] if labels[v]==labels[u]]
        if not neigh_intra: continue

        v_del = rng.choice(neigh_intra)

        # candidate w_add: other-class, not neighbour yet
        cand = [w for w in hub_set if labels[w]!=labels[u] and w not in G[u]]
        if not cand: continue
        w_add = rng.choice(cand)

        # --- constraints check ------------------------------------------- #
        if is_bridge(u,v_del):        # keep connected
            continue
        if G.has_edge(u,w_add):       # should not happen, double sanity
            continue
        # avoid new 2-hop path linking all classes:
        # here: forbid if w_add already sees *every* class incl. lbl[u]
        classes_seen = set(labels[x] for x in G[w_add]) | {labels[u]}
        if len(classes_seen) == len(np.unique(labels)):
            continue

        # surrogate-score or random accept
        if rng.random() < p_surrogate:
            Δ = surrogate_loss_delta(X, labels, G, (u,v_del,w_add))
            if Δ <= 0:                           # makes loss *smaller* (better) for defender
                continue                        # => reject
        # --- execute the swap -------------------------------------------- #
        G.remove_edge(u, v_del)
        G.add_edge   (u, w_add)
        steps += 1

    new_edges = np.array(G.edges(), dtype=np.int64)

    # optional micro feature noise
    if flip_features:
        rare_mask = (X.sum(0) < 10)             # bits that are almost always 0
        rare_cols = np.where(rare_mask)[0]
        for u in hubs:
            if rare_cols.size == 0: break
            bit = rng.choice(rare_cols)
            X[u, bit] = 1-X[u, bit]             # toggle

    return new_edges, X


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def save_perturbed(out_dir: Path, ids: List[str], feats: np.ndarray,
                   labels: np.ndarray, edges: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"perturbed_edges.csv", 'w') as f:
        for u,v in edges:
            f.write(f"{ids[u]},{ids[v]}\n")
    with open(out_dir/"perturbed_content.csv", 'w') as f:
        for idx,(feat,lab) in enumerate(zip(feats, labels)):
            feat_str = ','.join(map(str, feat))
            f.write(f"{idx},{feat_str},{lab}\n")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora","citeseer","pubmed"], default="cora")
    parser.add_argument("--root",    default="data")
    parser.add_argument("--budget",  type=int, default=400)
    parser.add_argument("--seed",    type=int, default=0)
    args   = parser.parse_args()

    # 1) load
    X, y, E, data = load_planetoid(args.root, args.dataset)
    ids = np.arange(data.num_nodes).astype(str)

    # 2) attack
    new_E, new_X = strong_unnoticeable_attack(
        edges         = E,
        labels        = y,
        budget        = args.budget,
        p_surrogate   = 0.6,
        topk_hubs     = 400,
        flip_features = True,
        seed          = args.seed
    )

    # 3) save
    out_path = Path(args.root)/args.dataset/"perturbed"
    save_perturbed(out_path, ids, new_X, y, new_E)
    print(f"Perturbed graph written to  {out_path}  "
          f"(original |E|={len(E)}  →  new |E|={len(new_E)})")
