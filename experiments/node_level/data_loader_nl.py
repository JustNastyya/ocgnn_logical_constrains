import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

from torch.utils.data import random_split


TUDATASETS = ["MUTAG", "REDDIT-BINARY"]
NORMAL_LABEL = 0
DATASET_REFERENCE = {
    "Cora": Planetoid
}

def get_data(dataset_name, batch_size):
    if dataset_name in TUDATASETS:
        dataset = TUDataset(root=f"data/{dataset_name}", name=dataset_name)    
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset, loader
    else:
        dataset = DATASET_REFERENCE[dataset_name](root=f'data/{dataset_name}', name=dataset_name)
        loader = DataLoader(dataset)
        return dataset, loader
        

def split_test_train(dataset):
    data = dataset[0]
    normal_mask = (data.y == NORMAL_LABEL)
    anomaly_mask = ~normal_mask

    normal_idx = normal_mask.nonzero(as_tuple=False).view(-1)

    perm = torch.randperm(normal_idx.size(0))
    train_size = int(0.8 * normal_idx.size(0))

    train_idx = normal_idx[perm[:train_size]]
    test_normal_idx = normal_idx[perm[train_size:]]

    test_idx = torch.cat([
        test_normal_idx,
        anomaly_mask.nonzero(as_tuple=False).view(-1)
    ])

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return data, train_mask, test_mask