from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

import networkx as nx
import matplotlib.pyplot as plt

from config import REDDIT_PATH


def get_data_reddit():
    dataset = TUDataset(root=REDDIT_PATH, name='REDDIT-BINARY')    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


def visualize_graph(graph_number):
    dataset = TUDataset(root=REDDIT_PATH, name='REDDIT-BINARY')

    data = dataset[graph_number]
    networkx_graph = to_networkx(data)

    plt.figure(figsize=(5, 5))
    nx.draw(networkx_graph, node_size=50)
    plt.show()


if __name__ == "__main__":
    visualize_graph(graph_number=0)
