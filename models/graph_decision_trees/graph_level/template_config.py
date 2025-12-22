import torch
from torch_geometric.utils import degree


class TemplateFeatureExtractor:
    def __init__(self, attribute_list):
        self.attribute_list = attribute_list
        self.translator = {
            "node_features": self.node_features,
            "node_degree": self.node_degree, # TODO: directed
        }

    def node_features(self, data):
        return data.x
    
    def node_degree(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        deg = degree(edge_index[0], num_nodes=num_nodes)
        return deg
