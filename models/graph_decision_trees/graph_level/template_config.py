import torch
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected


class TemplateFeatureExtractor:
    def __init__(self, attribute_list):
        self.attribute_list = attribute_list
        self.translator = {
            "node_features": self.node_features,
            "node_degree": self.node_degree, # TODO: directed
            "clustering_coefficient": self.clustering_coefficient,
        }

    def node_features(self, data):
        if len(data.x.shape) == 1:
            return data.x
        # devide into list of features
        l_x = []
        for col in range(data.x.shape[1]):
            l_x.append(data.x[:,col])
        return l_x
    
    def node_degree(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        deg = degree(edge_index[0], num_nodes=num_nodes)
        return deg

    def clustering_coefficient(self, data):
        edge_index = to_undirected(data.edge_index)
        num_nodes = data.num_nodes

        adj = [[] for _ in range(num_nodes)]
        for u, v in edge_index.t().tolist():
            adj[u].append(v)

        cc = torch.zeros(num_nodes, dtype=torch.float)

        for v in range(num_nodes):
            neighbors = adj[v]
            k = len(neighbors)

            if k < 2:
                cc[v] = 0.0
                continue

            neighbor_set = set(neighbors)
            edges_between_neighbors = 0

            for u in neighbors:
                for w in adj[u]:
                    if w in neighbor_set:
                        edges_between_neighbors += 1

            edges_between_neighbors //= 2

            cc[v] = (2.0 * edges_between_neighbors) / (k * (k - 1))

        return cc