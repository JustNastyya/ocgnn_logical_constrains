from template_config import TemplateFeatureExtractor
import torch
import random
from collections import defaultdict

class GraphLevelFeatureExtractor(TemplateFeatureExtractor):
    def __init__(self, attribute_list):
        super().__init__(attribute_list)
    
    def extract_features(self, dataset, balance=False):
        X_l = []
        y_l = []
        
        for graph_idx, data in enumerate(dataset):
            xs = []

            for feature_name in self.attribute_list:
                
                feature_l = self.translator[feature_name](data)
                if feature_l is not None:
                    if type(feature_l) is list:
                        xs.extend(feature_l)
                    else:    
                        xs.append(feature_l)
            
            if len(xs) > 1:
                X_l.append(torch.stack(xs, dim=1))
            else:
                X_l.append(xs[0]) # TODO len 0

            graph_label = data.y  
            num_nodes = data.num_nodes
            node_y = graph_label.repeat(num_nodes)
            y_l.append(node_y)
            
        X = torch.cat(tuple(X_l), dim=0)
        if len(X.shape) == 1:
            X = X.unsqueeze(1)  
        y = torch.cat(y_l, dim=0)
        
        if balance:
            return self.balance(X, y)
        return X, y
    
    def balance(self, X, y):
        label_to_idx = defaultdict(list)

        for i, label in enumerate(y.tolist()):
            label_to_idx[label].append(i)

        min_size = min(len(v) for v in label_to_idx.values())

        balanced_indices = []
        for label, idxs in label_to_idx.items():
            balanced_indices.extend(random.sample(idxs, min_size))

        balanced_indices = torch.tensor(balanced_indices)

        X_bal = X[balanced_indices]
        y_bal = y[balanced_indices]
        return X_bal, y_bal

