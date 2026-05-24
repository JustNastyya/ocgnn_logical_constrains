import torch
import random
from collections import defaultdict

from constrains.graph_decision_trees.graph_level.template_config import TemplateFeatureExtractor
from experiments.graph_level.data_loader_gl import get_data

class GraphLevelFeatureExtractor(TemplateFeatureExtractor):
    def __init__(self, attribute_list):
        super().__init__(attribute_list)
        self.index_mapping = {}
    
    def get_data(self, dataset_name):
        _, loader = get_data(dataset_name, batch_size=32)
        return loader
    
    def extract_features(self, loader, mask=None, balance=False):
        X_l = []
        y_l = []
        
        for batch in loader:
            if hasattr(batch, "to_data_list"):
                data_list = batch.to_data_list()
            else:
                data_list = [batch]

            for graph_idx, data in enumerate(data_list):
                xs = []

                for feature_name in self.attribute_list:
                    feature_l = self.translator[feature_name](data)

                    if feature_l is not None:
                        if isinstance(feature_l, list):
                            feature_new_ind_start = len(xs)
                            xs.extend(feature_l)
                            feature_new_ind_end = len(xs)

                            self.index_mapping[
                                f"{feature_new_ind_start}:{feature_new_ind_end}"
                            ] = feature_name
                        else:
                            feature_new_ind = len(xs)
                            xs.append(feature_l)

                            self.index_mapping[str(feature_new_ind)] = feature_name

                X_l.append(xs)
                y_l.append(data.y)
                
        X = torch.tensor(X_l)
        if len(X.shape) == 1:
            X = X.unsqueeze(1)  
        
        y = torch.tensor(y_l)
        
        if mask is not None:
            X = X[mask]
            y = y[mask]
            return X, y
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

