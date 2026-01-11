import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from torch.utils.data import random_split


TUDATASETS = ["MUTAG", "REDDIT-BINARY"]
NORMAL_LABEL = 0

def get_data(dataset_name, batch_size):
    dataset = TUDataset(root=f"data/{dataset_name}", name=dataset_name)    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader
        

def split_test_train(dataset, batch_size):
    normal_graphs = [d for d in dataset if d.y.item() == NORMAL_LABEL]
    anomalous_graphs = [d for d in dataset if d.y.item() != NORMAL_LABEL]

    num_normal = len(normal_graphs)
    train_size = int(0.8 * num_normal)
    test_size = num_normal - train_size

    train_dataset, test_normal = random_split(
        normal_graphs, [train_size, test_size]
    )

    test_dataset = test_normal + anomalous_graphs

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader
