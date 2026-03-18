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
    "Physics": Coauthor,
    "Computers": Amazon,
    "Photo": Amazon,
    "Texas": WebKB,
    "Cornell": WebKB,
    "Wisconsin": WebKB,
}



TUDATASETS = ["MUTAG", "REDDIT-BINARY"]
NORMAL_LABEL = 0

def get_data(dataset_name, batch_size):
    if dataset_name in TUDATASETS:
        dataset = TUDataset(root=f"data/{dataset_name}", name=dataset_name)    
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset, loader
    else:
        dataset = DATASET_REFERENCE[dataset_name](root=f'data/{dataset_name}', name=dataset_name)
        loader = DataLoader(dataset)
        return dataset, loader


def split_train_val_test(dataset, train_ratio=0.7, val_ratio=0.1):
    data = dataset[0]

    # Identify normal vs anomaly
    normal_mask = (data.y == NORMAL_LABEL)
    anomaly_mask = ~normal_mask

    normal_idx = normal_mask.nonzero(as_tuple=False).view(-1)

    # Shuffle normal samples
    perm = torch.randperm(normal_idx.size(0))

    n_total = normal_idx.size(0)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Split indices
    train_idx = normal_idx[perm[:n_train]]
    val_idx = normal_idx[perm[n_train:n_train + n_val]]
    test_normal_idx = normal_idx[perm[n_train + n_val:]]

    # Test set = remaining normal + ALL anomalies
    test_idx = torch.cat([
        test_normal_idx,
        anomaly_mask.nonzero(as_tuple=False).view(-1)
    ])

    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return data, train_mask, val_mask, test_mask
