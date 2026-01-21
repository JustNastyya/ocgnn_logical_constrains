import torch
import random
from collections import defaultdict

from models.graph_decision_trees.template_config import TemplateFeatureExtractor
from experiments.node_level.data_loader_nl import get_data, split_test_train

class NodeLevelFeatureExtractor(TemplateFeatureExtractor):
    def __init__(self, attribute_list):
        super().__init__(attribute_list)
        self.index_mapping = {}
    
    def get_data(self, dataset_name):
        dataset, _ = get_data(dataset_name, batch_size=32)
        data, _, _ = split_test_train(dataset)
        return data
    
    def extract_features(self, data, balance=False):
        xs = []

        for feature_name in self.attribute_list:
            
            feature_l = self.translator[feature_name](data)
            if feature_l is not None:
                if type(feature_l) is list:
                    xs.extend(feature_l)
                else:    
                    feature_new_ind = len(xs)
                    
                    # is needed later for using in experiments
                    xs.append(feature_l)
                    self.index_mapping[feature_new_ind] = feature_name
        
        X = torch.stack(xs, dim=1)
        if len(X.shape) == 1:
            X = X.unsqueeze(1)  
        
        y = (data.y > 0).int()
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
