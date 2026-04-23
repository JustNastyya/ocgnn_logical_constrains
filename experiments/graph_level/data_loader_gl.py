import torch
import random
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from torch.utils.data import random_split


TUDATASETS = [
    "MUTAG", "AIDS", "DHFR", 
    "ENZYMES", "PROTEINS",                                                # bioinformatics
    "COIL-RAG", "MSRC_21",                                        # computer vision
    ]

NORMAL_LABEL = 0

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        data.idx = idx
        return data

def get_data(dataset_name, batch_size):
    dataset = TUDataset(root=f"data/{dataset_name}", name=dataset_name)    
    dataset = IndexedDataset(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader
        

def split_train_val_test(dataset, batch_size, train_ratio=0.5, val_ratio=0.1, tree_ratio=0.2):
    normal_indices = []
    anomaly_indices = []

    for i in range(len(dataset)):
        if dataset[i].y.item() == NORMAL_LABEL:
            normal_indices.append(i)
        else:
            anomaly_indices.append(i)

    random.shuffle(normal_indices)
    random.shuffle(anomaly_indices)

    n_normal = len(normal_indices)
    n_anomal = len(anomaly_indices)

    n_train = int(train_ratio * n_normal)
    n_val = int(val_ratio * n_normal)
    n_tree = int(tree_ratio * n_normal)
    
    n_val_anormal = int(val_ratio * n_anomal)
    n_tree_anormal = int(tree_ratio * n_anomal)
    
    # splitting
    train_idx = normal_indices[:n_train]

    val_normal_idx = normal_indices[n_train:n_train + n_val]
    val_anomaly_idx = anomaly_indices[:n_val_anormal]
    val_idx = val_normal_idx + val_anomaly_idx
    
    tree_normal_idx = normal_indices[n_train + n_val:n_train + n_val + n_tree]
    tree_anomaly_idx = anomaly_indices[n_val_anormal:n_val_anormal + n_tree_anormal]
    
    min_tree = min(len(tree_normal_idx), len(tree_anomaly_idx))
    tree_idx = tree_normal_idx[:min_tree] + tree_anomaly_idx[:min_tree]
    
    used_normal = n_train + n_val + n_tree
    used_anomaly = n_val_anormal + n_tree_anormal

    test_normal_idx = normal_indices[used_normal:]
    test_anomaly_idx = anomaly_indices[used_anomaly:]

    # test: remaining + all anomalies
    test_idx = test_normal_idx + test_anomaly_idx

    # subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    tree_dataset = Subset(dataset, tree_idx)

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    tree_loader = DataLoader(tree_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, tree_loader
