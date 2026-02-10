import torch
import random
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from torch.utils.data import random_split


TUDATASETS = ["MUTAG", "REDDIT-BINARY"]
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
        

def split_train_val_test(dataset, batch_size, train_ratio=0.7, val_ratio=0.1, seed=42):
    # Collect indices instead of Data objects
    normal_indices = []
    anomaly_indices = []

    for i in range(len(dataset)):
        if dataset[i].y.item() == NORMAL_LABEL:
            normal_indices.append(i)
        else:
            anomaly_indices.append(i)

    # Shuffle normal graphs
    random.shuffle(normal_indices)

    n_total = len(normal_indices)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Split
    train_idx = normal_indices[:n_train]
    val_idx = normal_indices[n_train:n_train + n_val]
    test_normal_idx = normal_indices[n_train + n_val:]

    # Test = remaining normals + ALL anomalies
    test_idx = test_normal_idx + anomaly_indices

    # Create Subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
