import json
import torch
from loguru import logger

from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor


class NLRuleBasedHandler:
    def __init__(self, filename, l_factor, normal_label):
        self.filename = filename
        self.json_rules = self._load_rules(filename)
        self.l_factor = l_factor
        self.anomaly_rules = []
        self.normal_label = normal_label
        self._load_anomaly_rules()

    def _load_rules(self, filename):
        with open(filename, "r") as f:
            return json.load(f)
    
    def _load_anomaly_rules(self):
        for rule in self.json_rules["constrains"]:
            if rule["predicted_class"] != self.normal_label:
                self.anomaly_rules.append(rule)
    
    def soft_leq(self, x, threshold):
        return torch.sigmoid(self.l_factor * (threshold - x))

    def soft_gt(self, x, threshold):
        return torch.sigmoid(self.l_factor * (x - threshold))

    def soft_and(self, values):
        return torch.prod(values, dim=0)

    def soft_or_prob(self, values):
        return 1.0 - torch.prod(1.0 - values, dim=0)

    def rule_satisfaction(self, rule, x):
        cond_values = []

        for cond in rule["conditions"]:
            idx = cond["feature_index"]
            if cond["op"] == "<=":
                v = self.soft_leq(x[:, idx], cond["threshold"])
            else:
                v = self.soft_gt(x[:, idx], cond["threshold"])
            cond_values.append(v)

        return self.soft_and(torch.stack(cond_values))

    def get_constraint_value(self, data):
        # extend X by the additional features
        attribute_list = self.json_rules["additional_attributes"].values()
        config = NodeLevelFeatureExtractor(attribute_list)
        X, _ = config.extract_features(data, balance=True)
        
        # fail save
        self._test_attribute_mapping(self.json_rules["additional_attributes"], config.index_mapping)

        rule_values = []
        for rule in self.anomaly_rules:
            rule_values.append(self.rule_satisfaction(rule, X))
        
        return self.l_factor * self.soft_or_prob(torch.stack(rule_values))

    def _test_attribute_mapping(self, old_attribute_mapper, new_attribute_mapper):
        try:
            assert len(old_attribute_mapper) == len(new_attribute_mapper)
            for index in sorted(old_attribute_mapper.keys()):
                assert old_attribute_mapper[index] == new_attribute_mapper[index]

        except Exception as e:
            logger.info("attribute mapping error: The new and old attribute lists do not match")
            logger.info(f"new attribute_list: {new_attribute_mapper}")
            logger.info(f"old attribute_list: {old_attribute_mapper}")
            
            raise Exception