from template_config import TemplateFeatureExtractor
import torch

class GraphLevelFeatureExtractor(TemplateFeatureExtractor):
    def __init__(self, attribute_list):
        super().__init__(attribute_list)
    
    def extract_features(self, dataset):
        X_l = []
        y_l = []
        for data in dataset:
            xs = []

            for feature_name in self.attribute_list:
                
                feature_l = self.translator[feature_name](data)
                if feature_l is not None:
                    xs.append(feature_l)
            
            if len(xs) > 1:
                X_l.append(torch.cat(tuple(xs), dim=1))
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
        return X, y
