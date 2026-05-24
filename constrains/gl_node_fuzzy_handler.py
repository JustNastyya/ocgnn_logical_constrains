import json
import torch
import numpy as np
from loguru import logger

from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor
from constrains.nl_fuzzy_based_handler import FuzzyBasedHandler

class GLNodeFuzzyHandler(FuzzyBasedHandler):
    def __init__(self, filename, l_factor, normal_label, aggregation='mean'):
        super().__init__(filename, l_factor, normal_label)
        self.aggregation = aggregation

    def _evaluate_fuzzy(self, X):
        rule_values = []

        for rule in self.anomaly_rules:
            rule_values.append(self.rule_satisfaction(rule, X))

        if len(rule_values) == 0:
            return torch.zeros(X.shape[0])

        return self.l_factor * self.soft_or_prob(torch.stack(rule_values))

    def get_constraint_value(self, loader):
        """returns a vector of length of number of graphs"""
        # extend X by the additional features
        attribute_list = self.json_rules["additional_attributes"].values()
        config = NodeLevelFeatureExtractor(attribute_list)

        constraint_values = []
        mapping_checked = False

        for batch in loader:
            if hasattr(batch, "to_data_list"):
                data_list = batch.to_data_list()
            else:
                data_list = [batch]

            for graph_data in data_list:
                X, _ = config.extract_features(graph_data, balance=False)

                if not mapping_checked:
                    self._test_attribute_mapping(self.json_rules["additional_attributes"], config.index_mapping)
                    mapping_checked = True

                node_constraints = self._evaluate_fuzzy(X)
            
                if self.aggregation == 'mean':
                    graph_constraint = node_constraints.mean()
                else:
                    graph_constraint = node_constraints.max()

                constraint_values.append(graph_constraint)

        return torch.tensor(constraint_values)
