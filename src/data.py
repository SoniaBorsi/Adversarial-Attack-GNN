# import numpy as np
# import scipy.sparse as sp
# import torch

# def encode_onehot(labels):
#     classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
#     labels_onehot = np.array([classes_dict[label] for label in labels], dtype=np.int32)
#     return labels_onehot


# def load_data(path):
#     # Remove trailing slash and extract dataset name
#     dataset = path.strip("/").split("/")[-1]
#     print(f"Loading dataset '{dataset}' from: {path}")

#     # Load the content file: rows: node id, features, label
#     idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels_raw = idx_features_labels[:, -1]
#     labels = encode_onehot(labels_raw)
    
#     # Build an index map from node id to consecutive index
#     idx = np.array(idx_features_labels[:, 0])
#     idx_map = {j: i for i, j in enumerate(idx)}
    
#     # Load the cites file; each row is a pair of node ids
#     edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=str)
    
#     # Filter edges: include only edges where both endpoints are in our mapping
#     valid_edges = [(idx_map[src], idx_map[dst])
#                    for src, dst in edges_unordered if src in idx_map and dst in idx_map]
#     edges = np.array(valid_edges, dtype=np.int32)
    
#     # Build the adjacency matrix in COO format
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
#     # Symmetrize the adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
#     # Normalize features and adjacency (adding self-loops before normalizing)
#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
    
#     # Splits: 140 for training and nodes 500-1499 for testing 
#     idx_train = torch.LongTensor(range(140))
#     idx_test = torch.LongTensor(range(500, 1500))
    
#     # Convert to PyTorch tensors
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
    
#     return adj, features, labels, idx_train, idx_test


# def normalize(mx):
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     return sp.diags(r_inv).dot(mx)

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     return torch.sparse.FloatTensor(indices, values, torch.Size(sparse_mx.shape))


import os
import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = sorted(set(labels))
    mapping = {c: np.eye(len(classes), dtype=np.int32)[i] for i, c in enumerate(classes)}
    return np.array([mapping[l] for l in labels], dtype=np.int32)

def load_data(path):
    """
    path should be .../data/<dataset>/  OR  .../data/<dataset>/perturbed/
    If there's a perturbed/ subfolder, we'll read perturbed_edges.csv and
    perturbed_content.csv (if present). Otherwise we fall back to the
    original .cites/.content files in the parent <dataset> folder.
    """
    path = path.rstrip("/")
    last = os.path.basename(path)
    if last == "perturbed":
        pert_dir = path
        base_dir = os.path.dirname(path)
        dataset = os.path.basename(base_dir)
    else:
        pert_dir = os.path.join(path, "perturbed")
        base_dir = path
        dataset = last

    print(f"Loading dataset '{dataset}' from {base_dir}")
    # Try perturbed first
    if os.path.isdir(pert_dir):
        print(" -> Found perturbed folder, loading CSVs…")
        edge_csv = os.path.join(pert_dir, "perturbed_edges.csv")
        feat_csv = os.path.join(pert_dir, "perturbed_content.csv")
        edges_unordered = np.genfromtxt(edge_csv, dtype=str, delimiter=",")
        # If features CSV exists, load those; else fallback to clean
        if os.path.exists(feat_csv):
            raw = np.genfromtxt(feat_csv, dtype=str, delimiter=",")
            ids = raw[:, 0]
            feats = raw[:, 1:].astype(np.float32)
            idx_map = {nid: i for i, nid in enumerate(ids)}
            features = sp.csr_matrix(feats)
            # labels come from the clean .content
            clean_cont = os.path.join(base_dir, f"{dataset}.content")
            cont = np.genfromtxt(clean_cont, dtype=str, delimiter="\t")
            cont_map = {nid: lab for nid, *_, lab in cont}
            labels_raw = [cont_map[nid] for nid in ids]
            labels = encode_onehot(labels_raw)
        else:
            print("   (no perturbed_content.csv, loading clean content instead)")
            cont = np.genfromtxt(os.path.join(base_dir, f"{dataset}.content"),
                                 dtype=str, delimiter="\t")
            ids = cont[:, 0]
            feats = cont[:, 1:-1].astype(np.float32)
            features = sp.csr_matrix(feats)
            labels = encode_onehot(cont[:, -1])
            idx_map = {nid: i for i, nid in enumerate(ids)}

        # build edges
        valid = [(idx_map[s], idx_map[d]) for s, d in edges_unordered
                 if s in idx_map and d in idx_map]
        edges = np.array(valid, dtype=np.int32)

    else:
        print(" -> No perturbed data, loading clean files")
        # clean .content
        cont = np.genfromtxt(os.path.join(base_dir, f"{dataset}.content"),
                     dtype=str, delimiter="\t")
        ids = cont[:, 0]
        feats = cont[:, 1:-1].astype(np.float32)
        features = sp.csr_matrix(feats)
        labels = encode_onehot(cont[:, -1])
        idx_map = {nid: i for i, nid in enumerate(ids)}
        # clean .cites
        edges_unordered = np.genfromtxt(os.path.join(base_dir, f"{dataset}.cites"),
                                        dtype=str)
        valid = [(idx_map[s], idx_map[d]) for s, d in edges_unordered
                 if s in idx_map and d in idx_map]
        edges = np.array(valid, dtype=np.int32)

    # build adjacency
    n = len(idx_map)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])),
                        shape=(n,n), dtype=np.float32)
    # symmetrize
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # normalize
    features = normalize(features)
    adj = normalize(adj + sp.eye(n))

    # fixed splits
    idx_train = torch.LongTensor(range(140))
    idx_test  = torch.LongTensor(range(500, 1500))

    # to torch
    features = torch.FloatTensor(features.todense())
    labels   = torch.LongTensor(np.where(labels)[1])
    adj      = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    inv = np.where(rowsum>0, 1.0/rowsum, 0.0)
    return sp.diags(inv).dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor([sparse_mx.row, sparse_mx.col])
    values  = torch.FloatTensor(sparse_mx.data)
    shape   = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

