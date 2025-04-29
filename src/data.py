from __future__ import annotations
import os
from pathlib import Path
from typing  import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid, WebKB, PolBlogs
from torch_geometric.utils    import to_scipy_sparse_matrix

# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def load_data(
    root: str,
    *,
    dataset: str,
    use_perturbed: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Unified loader for citation-type graphs (Planetoid, WebKB Texas, PolBlogs).

    Parameters
    ----------
    root          Folder that contains     <dataset>/raw/   <dataset>/processed/
                  (and optionally      <dataset>/perturbed/)
    dataset       "cora" | "citeseer" | "pubmed" | "texas" | "polblogs"
    use_perturbed When *True* and the sub-folder ``perturbed/`` exists, the
                  CSVs produced by an attack script are read instead of the
                  clean PyG graph.

    Returns
    -------
    adj      (N,N)  torch.sparse.FloatTensor      row-normalised adjacency 𝐃⁻¹(𝐀+𝐈)
    features (N,F)  torch.FloatTensor             row-normalised features 𝐃⁻¹𝐗
    labels   (N,)   torch.LongTensor              class indices 0 … C-1
    idx_tr   (n_train,)  torch.LongTensor         training nodes (PyG canonical
                                                   mask if present, otherwise
                                                   stratified random split)
    idx_te   (n_test,)   torch.LongTensor         test nodes
    """
    name     = dataset.lower()
    base_dir = (Path(root) / name).expanduser().resolve()
    pert_dir = base_dir / "perturbed"

    print(f"Loading dataset '{name}' from {base_dir}")

    if use_perturbed and pert_dir.is_dir():
        print(" → Loading **PERTURBED** CSVs")
        adj_sp, feats_sp, labels = _read_perturbed(base_dir, pert_dir)
        pyg_data, idx_tr, idx_te = _pyg_split(base_dir, name)
    else:
        if pert_dir.is_dir() and not use_perturbed:
            print(" → Perturbed folder exists but flag is False – IGNORED")
        adj_sp, feats_sp, labels, idx_tr, idx_te = _read_clean_pyg(base_dir, name)

    # ---------- to torch ---------------------------------------------------
    features = torch.FloatTensor(feats_sp.todense())
    labels   = torch.LongTensor(np.where(labels)[1])  # one-hot → int
    adj      = _sp_to_torch(adj_sp)
    return adj, features, labels, idx_tr, idx_te

# ────────────────────────────────────────────────────────────────────
# PERTURBED CSVs (unchanged, dataset-agnostic)
# ────────────────────────────────────────────────────────────────────

def _read_perturbed(base: Path, pert: Path):
    """Load CSVs produced by the *strong-unnoticeable-attack* script."""
    edge_csv = pert / "perturbed_edges.csv"
    feat_csv = pert / "perturbed_content.csv"

    # ---------- edges ------------------------------------------------------
    edges_raw = np.genfromtxt(edge_csv, dtype=str, delimiter=",")
    src_ids, dst_ids = edges_raw[:, 0], edges_raw[:, 1]

    # ---------- node table -------------------------------------------------
    if feat_csv.exists():
        raw         = np.genfromtxt(feat_csv, dtype=str, delimiter=",")
        node_ids    = raw[:, 0]
        node_feats  = raw[:, 1:-1].astype(np.float32)
        node_labels = raw[:, -1]
    else:
        print("   (perturbed_content.csv missing – falling back to PyG raw files)")
        pyg_data, *_ = _pyg_split(base, base.name)
        node_ids    = np.array(pyg_data.node_ids if hasattr(pyg_data, "node_ids") else np.arange(pyg_data.num_nodes))
        node_feats  = pyg_data.x.cpu().numpy()
        node_labels = pyg_data.y.cpu().numpy()

    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    edges  = np.array([(id2idx[s], id2idx[d])
                       for s, d in zip(src_ids, dst_ids)
                       if s in id2idx and d in id2idx], dtype=np.int32)

    feats_sp = sp.csr_matrix(node_feats)
    labels   = _onehot(node_labels)

    adj_sp   = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                             shape=(len(node_ids), len(node_ids)),
                             dtype=np.float32)
    adj_sp   = adj_sp + adj_sp.T.multiply(adj_sp.T > adj_sp)
    feats_sp = _row_norm(feats_sp)
    adj_sp   = _row_norm(adj_sp + sp.eye(adj_sp.shape[0]))
    return adj_sp, feats_sp, labels

# ────────────────────────────────────────────────────────────────────
# CLEAN graphs via PyG
# ────────────────────────────────────────────────────────────────────

def _read_clean_pyg(base: Path, name: str):
    """Load the *clean* graph exactly as stored by PyG (raw/processed)."""
    pyg_data, idx_tr, idx_te = _pyg_split(base, name)

    # -------- adjacency ----------------------------------------------------
    edge_index = pyg_data.edge_index
    adj_sp = to_scipy_sparse_matrix(edge_index, num_nodes=pyg_data.num_nodes)
    adj_sp = adj_sp.astype(np.float32)
    adj_sp = adj_sp + adj_sp.T.multiply(adj_sp.T > adj_sp)
    adj_sp = _row_norm(adj_sp + sp.eye(adj_sp.shape[0]))

    # -------- features & labels -------------------------------------------
    feats_sp = sp.csr_matrix(pyg_data.x.cpu().numpy())
    feats_sp = _row_norm(feats_sp)
    labels   = _onehot(pyg_data.y.cpu().numpy())
    return adj_sp, feats_sp, labels, idx_tr, idx_te

# ────────────────────────────────────────────────────────────────────
# Generic PyG dataset loader + split helper
# ────────────────────────────────────────────────────────────────────

def _pyg_split(base: Path, name: str):
    """Return *(data, idx_train, idx_test)* for any of the supported datasets."""
    name = name.lower()

    # -------- pick the correct PyG dataset class --------------------------
    if name in {"cora", "citeseer", "pubmed"}:
        pyg_ds = Planetoid(str(base), name=name.capitalize(), split="public")
    elif name == "texas":
        pyg_ds = WebKB(str(base), name="Texas")
    elif name == "polblogs":
        pyg_ds = PolBlogs(str(base))
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = pyg_ds[0]

    # -------- obtain (possibly create) splits -----------------------------
    if hasattr(data, "train_mask") and data.train_mask is not None and int(data.train_mask.sum()) > 0:
        idx_tr = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_te = data.test_mask .nonzero(as_tuple=False).view(-1)
    else:   # datasets without canonical masks → make 60/40 stratified split
        rng      = np.random.default_rng(0)
        labels   = data.y.cpu().numpy()
        idx_tr_l, idx_te_l = [], []
        for c in np.unique(labels):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            k = max(1, int(0.6 * len(idx_c)))   # at least 1 per class
            idx_tr_l.extend(idx_c[:k])
            idx_te_l.extend(idx_c[k:])
        idx_tr = torch.as_tensor(idx_tr_l, dtype=torch.long)
        idx_te = torch.as_tensor(idx_te_l, dtype=torch.long)
    return data, idx_tr, idx_te

# ────────────────────────── misc utils ─────────────────────────────

def _onehot(labels):
    classes = sorted(set(labels))
    eye     = np.eye(len(classes), dtype=np.int32)
    table   = {c: eye[i] for i, c in enumerate(classes)}
    return np.array([table[l] for l in labels])


def _row_norm(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    inv    = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
    return sp.diags(inv).dot(mx)


def _sp_to_torch(mx):
    mx = mx.tocoo().astype(np.float32)
    idx = torch.LongTensor([mx.row, mx.col])
    val = torch.FloatTensor(mx.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(mx.shape))

