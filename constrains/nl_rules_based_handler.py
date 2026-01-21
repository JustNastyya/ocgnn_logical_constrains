import json
import torch

from experiments.node_level.run_experiment_nl import NORMAL_LABEL

class NLRuleBasedHandler:
    def __init__(self, filename, l_factor):
        self.filename = filename
        self.json_rules = self._load_rules(filename)
        self.l_factor = l_factor
        self.anomaly_rules = []
        self._load_anomaly_rules()

    def _load_rules(self, filename):
        with open(filename, "r") as f:
            return json.load(f)
    
    def _load_anomaly_rules(self):
        for rule in self.json_rules:
            if rule["predicted_class"] != NORMAL_LABEL:
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

    def get_constraint_value(self, x):
        rule_values = []
        for rule in self.anomaly_rules:
            rule_values.append(self.rule_satisfaction(rule, x))
        
        return self.l_factor * self.soft_or(torch.stack(rule_values))

