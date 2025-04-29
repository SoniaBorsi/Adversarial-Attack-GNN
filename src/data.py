# ───────────────────────────── data.py ──────────────────────────────
"""
Unified data-loader for Planetoid-style benchmarks and their perturbed
CSV variants.  See the docstring of `load_data` for returned objects.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp          #  ← NEW: needed for helpers below

# -------------------------------------------------------------------- #
# ----------- Helper functions you asked to include ------------------ #
# -------------------------------------------------------------------- #
def _row_norm(mx: sp.spmatrix) -> sp.spmatrix:
    """Row-normalise a SciPy sparse matrix (D^-1 · A)."""
    rowsum = np.array(mx.sum(1)).flatten()
    inv    = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
    return sp.diags(inv) @ mx


def _sp_to_torch(mx: sp.spmatrix) -> torch.Tensor:
    """Convert SciPy sparse matrix to `torch.sparse.FloatTensor`."""
    mx  = mx.tocoo().astype(np.float32)
    idx = torch.LongTensor([mx.row, mx.col])
    val = torch.FloatTensor(mx.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(mx.shape))

# -------------------------------------------------------------------- #
# ---------------------------- Public API ---------------------------- #
# -------------------------------------------------------------------- #
def load_data(
    root:   str,
    dataset:str,
    use_perturbed: bool = False,
):
    """Return (adj, x, y, idx_train, idx_test)."""
    if use_perturbed:
        return _load_perturbed(root)
    return _load_processed(root, dataset)

# -------------------------------------------------------------------- #
# ----------- Standard (un-perturbed) datasets via PyG --------------- #
# -------------------------------------------------------------------- #
def _load_processed(root: str, dataset: str):
    processed_dir = os.path.join(root, "processed")
    data_pt       = os.path.join(processed_dir, "data.pt")
    if not os.path.isfile(data_pt):
        raise FileNotFoundError(f"Missing {data_pt}")

    # Lazy import so we don't need PyG when only using CSV datasets
    from torch_geometric.data import Data                                      # noqa: E402
    data, slices = torch.load(data_pt)

    # PyG ≥3.x stores a dict+tensors; ≤2.x stores Data directly
    g = data if isinstance(data, Data) else Data()
    if not isinstance(data, Data):
        for key in data.keys:
            g[key] = data[key][slices[key][0] : slices[key][1]]

    num_nodes = g.num_nodes
    # ----- adjacency ---------------------------------------------------
    row, col = g.edge_index.cpu().numpy()
    adj_sp   = sp.coo_matrix((np.ones(len(row)), (row, col)),
                             shape=(num_nodes, num_nodes),
                             dtype=np.float32)
    adj_sp   = adj_sp + adj_sp.T
    adj_sp   = adj_sp + sp.eye(num_nodes, dtype=np.float32)
    adj      = _sp_to_torch(adj_sp)

    # ----- features & labels ------------------------------------------
    x  = torch.tensor(g.x, dtype=torch.float32)
    y  = torch.tensor(g.y, dtype=torch.long)

    # ----- train/test split -------------------------------------------
    if hasattr(g, "train_mask") and hasattr(g, "test_mask"):
        idx_train = g.train_mask.nonzero(as_tuple=False).view(-1)
        idx_test  = g.test_mask .nonzero(as_tuple=False).view(-1)
    else:
        # Fallback (Planetoid default or 80/20 split)
        if dataset.lower() in {"cora", "citeseer", "pubmed"}:
            idx_train = torch.arange(0, 140,  dtype=torch.long)
            idx_test  = torch.arange(1708, 2708, dtype=torch.long)
        else:
            perm      = torch.randperm(num_nodes)
            split     = int(0.8 * num_nodes)
            idx_test  = perm[: num_nodes - split]
            idx_train = perm[num_nodes - split :]

    return adj, x, y, idx_train, idx_test

# -------------------------------------------------------------------- #
# ----------------- Perturbed CSV datasets loader -------------------- #
# -------------------------------------------------------------------- #
def _load_perturbed(root: str):
    pert_dir = os.path.join(root, "perturbed")
    content  = os.path.join(pert_dir, "perturbed_content.csv")
    edges    = os.path.join(pert_dir, "perturbed_edges.csv")
    if not (os.path.isfile(content) and os.path.isfile(edges)):
        raise FileNotFoundError(f"CSV files not found in {pert_dir}")

    # ----- nodes, features, labels ------------------------------------
    df       = pd.read_csv(content, header=None)
    node_ids = df.iloc[:, 0].astype(str).tolist()
    id_map   = {nid: i for i, nid in enumerate(node_ids)}

    # sparse feature matrix → row-normalised → dense torch tensor
    feat_sp  = sp.csr_matrix(df.iloc[:, 1:-1].values, dtype=np.float32)
    feat_sp  = _row_norm(feat_sp)
    x        = torch.tensor(feat_sp.toarray(), dtype=torch.float32)

    y_raw    = df.iloc[:, -1].astype("category").cat.codes
    y        = torch.tensor(y_raw.values, dtype=torch.long)

    num_nodes = x.shape[0]

    # ----- edges -------------------------------------------------------
    e_df   = pd.read_csv(edges, header=None).astype(str)
    rows, cols = [], []
    for src, dst in e_df.itertuples(index=False):
        if src in id_map and dst in id_map:
            rows.append(id_map[src])
            cols.append(id_map[dst])

    if not rows:
        raise ValueError("No valid edges after filtering – check CSVs")

    adj_sp = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=(num_nodes, num_nodes),
                           dtype=np.float32)
    adj_sp = adj_sp + adj_sp.T
    adj_sp = adj_sp + sp.eye(num_nodes, dtype=np.float32)
    adj    = _sp_to_torch(adj_sp)

    # ----- train/test split -------------------------------------------
    tr_np, te_np = (os.path.join(pert_dir, f"idx_{s}.npy") for s in ("train", "test"))
    if os.path.isfile(tr_np) and os.path.isfile(te_np):
        idx_train = torch.from_numpy(np.load(tr_np)).long()
        idx_test  = torch.from_numpy(np.load(te_np)).long()
    else:
        split     = int(0.8 * num_nodes)
        perm      = torch.randperm(num_nodes)
        idx_test  = perm[: num_nodes - split]
        idx_train = perm[num_nodes - split :]

    return adj, x, y, idx_train, idx_test

