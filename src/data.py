import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([classes_dict[label] for label in labels], dtype=np.int32)
    return labels_onehot

def load_data(path):
    print(f"Loading dataset from: {path}")
    dataset = path.strip("/").split("/")[-1]
    
    idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    idx = np.array(idx_features_labels[:, 0])  
    idx_map = {j: i for i, j in enumerate(idx)} 
    
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32)
    #edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    edges_unordered = np.genfromtxt(f"{path}{dataset}.cites", dtype=str)

    # Filter edges to only include nodes present in the feature matrix
    valid_edges = [
        (idx_map[src], idx_map[dst])
        for src, dst in edges_unordered
        if src in idx_map and dst in idx_map
    ]

    edges = np.array(valid_edges, dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # for semi supervised 
    # idx_train = torch.LongTensor(range(140))
    # idx_val = torch.LongTensor(range(200, 500))
    # idx_test = torch.LongTensor(range(500, 1500))
    
    num_nodes = labels.shape[0]
    split = int(num_nodes * 0.8)
    
    idx_train = torch.LongTensor(range(split))
    idx_test = torch.LongTensor(range(split, num_nodes))
    #idx_val = torch.LongTensor([])  # Optional

    perm = torch.randperm(num_nodes)
    idx_train = perm[:split]
    idx_test = perm[split:]
    

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_test


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return sp.diags(r_inv).dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(sparse_mx.shape))






