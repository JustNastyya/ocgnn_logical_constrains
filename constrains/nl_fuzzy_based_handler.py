import json
import torch
from loguru import logger

from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor
from constrains.fuzzy_handler import FuzzyBasedHandler

class NLFuzzyBasedHandler(FuzzyBasedHandler):
    def __init__(self, filename, l_factor, normal_label):
        super().__init__(filename, l_factor, normal_label)

    def get_constraint_value(self, data):
        """returns a vector of length of number of nodes"""
        # extend X by the additional features
        attribute_list = self.json_rules["additional_attributes"].values()
        config = NodeLevelFeatureExtractor(attribute_list)
        X, _ = config.extract_features(data, balance=False)
        
        # fail save
        self._test_attribute_mapping(self.json_rules["additional_attributes"], config.index_mapping)

        rule_values = []
        for rule in self.anomaly_rules:
            rule_values.append(self.rule_satisfaction(rule, X))
        
        if len(rule_values) == 0:
            return torch.zeros(X.shape[0])
        return self.l_factor * self.soft_or_prob(torch.stack(rule_values))
