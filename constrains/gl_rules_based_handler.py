import json
import torch
from loguru import logger

from models.graph_decision_trees.graph_level.config import GraphLevelFeatureExtractor
from constrains.constrains_handler import RuleBasedHandler

class GLRuleBasedHandler(RuleBasedHandler):
    def __init__(self, filename, l_factor, normal_label):
        super().__init__(filename, l_factor, normal_label)

    def get_constraint_value(self, loader):
        """returns a vector of length of number of graphs"""
        # extend X by the additional features
        attribute_list = self.json_rules["additional_attributes"].values()
        config = GraphLevelFeatureExtractor(attribute_list)
        X, _ = config.extract_features(loader, balance=False)
        # fail save
        self._test_attribute_mapping(self.json_rules["additional_attributes"], config.index_mapping)

        rule_values = []
        for rule in self.anomaly_rules:
            rule_values.append(self.rule_satisfaction(rule, X))
        
        return self.l_factor * self.soft_or_prob(torch.stack(rule_values))
