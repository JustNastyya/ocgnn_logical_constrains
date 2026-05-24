import json
import torch.nn
import numpy as np
from loguru import logger
from models.graph_decision_trees.graph_level.config import GraphLevelFeatureExtractor
from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor


class ConstraintScoreBasedHandler:
    def __init__(self, filename, l_factor, normal_label, device="cpu"):
        self.filename = filename
        self.json_rules = self._load_rules(filename)
        self.l_factor = l_factor
        self.normal_label = normal_label
        self.anomaly_rules = []
        self.normality_rules = []
        self.device = device
        self._load_anomaly_rules()

    def _load_rules(self, filename):
        with open(filename, "r") as f:
            return json.load(f)
    
    def _load_anomaly_rules(self):
        for rule in self.json_rules["constrains"]:
            if rule["predicted_class"] == self.normal_label:
                self.anomaly_rules.append(rule)
            else:
                self.normality_rules.append(rule)

    def _get_distance(self, x, constraint):
        """x - vector and condition taken from json"""
        distances = []
        for condition in constraint["conditions"]:
            distances.append(abs(x[condition["feature_index"]] - condition["threshold"]))
        return min(distances)
    
    def _get_weighted_group_distance(self, x, constrain_group):
        """weighted_group_distance = lambda / n_constrains * SUM(distance(x, constrain_boundary))"""
        distance_sum = 0
        for constrain in constrain_group:
            distance_sum += self._get_distance(x, constrain)
        
        if len(constrain_group) == 0:
            return 0
        result = distance_sum * self.l_factor / len(constrain_group)
        return result

    def rule_satisfaction(self, x, constrain_group):
        """is x in this group of constrains and if yes - get distance to the decision boundary"""
        for constraint in constrain_group:
            satisfies = True
            for cond in constraint["conditions"]:
                idx = cond["feature_index"]
                if not(cond["op"] == "<=" and x[idx] <= cond["threshold"]):
                    satisfies = False
                    continue        
                if not(cond["op"] == ">" and x[idx] > cond["threshold"]):
                    satisfies = False
                    continue
            if satisfies:
                return self._get_distance(x, constraint)
        return None 
    
    def _get_min_distant(self, x, constrain_group):
        distant_l = []
        for constraint in constrain_group:
            distant_l.append(self._get_distance(x, constraint))
        
        return min(distant_l)

    def get_constraint_score(self, x):
        group_sum_0 = self._get_weighted_group_distance(x, self.normality_rules)
        group_sum_1 = self._get_weighted_group_distance(x, self.anomaly_rules)
        
        group_dis_diff = group_sum_0 - group_sum_1
        
        normality_satisfaction = self.rule_satisfaction(x, self.normality_rules)
        anomaly_satisfaction = self.rule_satisfaction(x, self.anomaly_rules)

        if normality_satisfaction is not None:
            # x is in a normal region
            constraint_distance_mul = -1 * normality_satisfaction * self._get_min_distant(x, self.anomaly_rules)
            constraint_distance = constraint_distance_mul + group_dis_diff
        elif anomaly_satisfaction is not None:
            # x is in a anormal region
            constraint_distance_mul = anomaly_satisfaction * self._get_min_distant(x, self.normality_rules)
            constraint_distance = constraint_distance_mul + group_dis_diff
        else:
            # "grey zone"
            constraint_distance = group_dis_diff
        return constraint_distance
        
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



class GLConstraintScoreBasedHandler(ConstraintScoreBasedHandler):
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

        scores = [self.get_constraint_score(x) for x in X]
        
        # apply sigmoid
        scores_st = 1 / (1 + np.exp(np.array(scores)))
        L_constrains = torch.tensor(scores_st, dtype=torch.float32, device=self.device)
        
        return L_constrains


class NLConstraintScoreBasedHandler(ConstraintScoreBasedHandler):
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

        scores = [self.get_constraint_score(x) for x in X]
        
        # apply sigmoid
        scores_st = 1 / (1 + np.exp(np.array(scores)))
        L_constrains = torch.tensor(scores_st, dtype=torch.float32, device=self.device)
        
        return L_constrains
