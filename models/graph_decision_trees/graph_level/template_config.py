import torch
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected


class TemplateFeatureExtractor:
    def __init__(self, attribute_list):
        self.attribute_list = attribute_list
        self.translator = {
            "mean_node_features": self.mean_node_features, # have to remain the first ones!
            "mean_node_degree": self.mean_node_degree, # TODO: directed
            "there_is_a_node": self.there_is_a_node
        }

    def mean_node_features(self, data):
        if len(data.x.shape) == 1:
            return data.x.mean()
        # devide into list of features
        l_x = []
        for col in range(data.x.shape[1]):
            l_x.append(data.x[:,col].mean())
        return l_x
    
    def mean_node_degree(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        deg = degree(edge_index[0], num_nodes=num_nodes).mean()
        return deg

    def there_is_a_node(self, data):
        pass
