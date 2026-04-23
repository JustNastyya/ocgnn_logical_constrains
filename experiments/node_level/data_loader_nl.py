import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WebKB
from torch.utils.data import random_split


DATASET_REFERENCE = {
    "Cora": Planetoid,
    "CiteSeer": Planetoid,
    "PubMed": Planetoid,
    "CS": Coauthor,
    "Photo": Amazon,
    "Texas": WebKB,
    "Cornell": WebKB,
    "Wisconsin": WebKB,
}

NORMAL_LABEL = 0

def get_data(dataset_name, batch_size):
    dataset = DATASET_REFERENCE[dataset_name](root=f'data/{dataset_name}', name=dataset_name)
    loader = DataLoader(dataset)
    return dataset, loader


def split_train_val_test(dataset, train_ratio=0.5, val_ratio=0.1, tree_ratio=0.2):
    data = dataset[0]
    
    normal_mask = (data.y == NORMAL_LABEL)
    anomaly_mask = ~normal_mask

    normal_idx = normal_mask.nonzero(as_tuple=False).view(-1)
    anomaly_idx = anomaly_mask.nonzero(as_tuple=False).view(-1)

    # shuffle it up
    normal_perm = torch.randperm(normal_idx.size(0))
    anomaly_perm = torch.randperm(anomaly_idx.size(0))
    normal_idx = normal_idx[normal_perm]
    anomaly_idx = anomaly_idx[anomaly_perm]

    n_normal = normal_idx.size(0)

    # sizes
    n_train = int(train_ratio * n_normal)
    n_val = int(val_ratio * n_normal)
    n_tree = int(tree_ratio * n_normal)

    train_idx = normal_idx[:n_train]

    val_normal_idx = normal_idx[n_train:n_train + n_val]
    val_anomaly_idx = anomaly_idx[:n_val]
    val_idx = torch.cat([val_normal_idx, val_anomaly_idx])

    tree_normal_idx = normal_idx[n_train + n_val:n_train + n_val + n_tree]
    tree_anomaly_idx = anomaly_idx[n_val:n_val + n_tree]

    min_tree = min(tree_normal_idx.size(0), tree_anomaly_idx.size(0))
    tree_idx = torch.cat([
        tree_normal_idx[:min_tree],
        tree_anomaly_idx[:min_tree]
    ])

    used_normal = n_train + n_val + n_tree
    used_anomaly = n_val + n_tree

    test_normal_idx = normal_idx[used_normal:]
    test_anomaly_idx = anomaly_idx[used_anomaly:]

    test_idx = torch.cat([test_normal_idx, test_anomaly_idx])

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    tree_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    tree_mask[tree_idx] = True

    return data, train_mask, val_mask, test_mask, tree_mask
