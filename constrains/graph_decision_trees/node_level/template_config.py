import torch
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from loguru import logger

class TemplateFeatureExtractor:
    def __init__(self, attribute_list):
        self.attribute_list = attribute_list
        self.translator = {
            "node_features": self.node_features, # have to remain the first ones!
            "node_degree": self.node_degree, # TODO: directed
            "clustering_coefficient": self.clustering_coefficient,
        }

    def node_features(self, data, devide_cat=True, cat_threshold=10):
        if len(data.x.shape) == 1:
            return data.x
        # devide into list of features
        l_x = []
        for col in range(data.x.shape[1]):
            feature = data.x[:, col]
            unique_vals = torch.unique(feature)
            n_unique = len(unique_vals)
            
            if devide_cat and 2 < n_unique <= cat_threshold:
                
                logger.info(f"Feature column {col} treated as categorical with {n_unique} unique values, one-hot encoded to {one_hot.shape[1]} features")
                _, encoded = torch.unique(feature, return_inverse=True)

                one_hot = F.one_hot(encoded, num_classes=n_unique).float()
                for k in range(one_hot.shape[1]):
                    l_x.append(one_hot[:, k])
            else:
                l_x.append(feature)    
    
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