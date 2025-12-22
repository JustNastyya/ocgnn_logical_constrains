import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


# small graph example

# every of 3 nodes have 2 features each:
x = torch.tensor([
    [1.0, 2.0],
    [0.5, 1.0],
    [3.0, 0.1]
])

# node in edge_index[i][0] connected to edge_index[i][1] 
edge_index = torch.tensor([
    [0, 1, 2, 0],
    [1, 0, 1, 2]
], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)


class GraphOneClass(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        graph_emb = global_mean_pool(x, batch)

        score = self.fc(graph_emb)
        return score


model = GraphOneClass()
node_embeddings = model(data.x, data.edge_index)
print(node_embeddings)
