# Import necessary libraries
import torch
import numpy as np
from anytree import Node, RenderTree
from loguru import logger


class GraphDecisionTreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None  # Left child
        self.right = None  # Right child
        self.is_leaf = False
        self.predicted_class = None
    
    def fit(self, X, y):
        # If all labels are the same, make a leaf node
        if y.unique().numel() == 1:
            self.is_leaf = True
            self.predicted_class = y[0].item()
            return

        # Check if maximum depth is reached
        if self.max_depth is not None and self.depth >= self.max_depth:
            self.is_leaf = True
            self.predicted_class = y.mode()[0].item()
            return

        # Find the best split
        best_gini = float('inf')
        logger.info(f"Nu features: {X.shape}")
        num_features = X.shape[1]
        for feature in range(num_features):
            thresholds = torch.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                y_left = y[left_mask]
                y_right = y[right_mask]
                if y_left.numel() == 0 or y_right.numel() == 0:
                    continue
                gini = self._gini(y_left, y_right)
                if gini < best_gini:
                    best_gini = gini
                    self.feature_index = feature
                    self.threshold = threshold.item()
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        logger.info(f"found thresholf for level: {self.depth}")
        
        # If no valid split is found, make a leaf node
        if self.feature_index is None:
            self.is_leaf = True
            self.predicted_class = y.mode()[0].item()
            return

        # Recursively build the left and right subtrees
        self.left = GraphDecisionTreeNode(
            depth=self.depth + 1, max_depth=self.max_depth)
        self.right = GraphDecisionTreeNode(
            depth=self.depth + 1, max_depth=self.max_depth)
        self.left.fit(X[best_left_mask], y[best_left_mask])
        self.right.fit(X[best_right_mask], y[best_right_mask])

    def predict(self, X):
        if self.is_leaf:
            return torch.full((X.shape[0],), self.predicted_class, dtype=torch.long)
        else:
            left_mask = X[:, self.feature_index] <= self.threshold
            right_mask = X[:, self.feature_index] > self.threshold
            y_pred = torch.empty(X.shape[0], dtype=torch.long)
            y_pred[left_mask] = self.left.predict(X[left_mask])
            y_pred[right_mask] = self.right.predict(X[right_mask])
            return y_pred

    def _gini(self, y_left, y_right):
        # Compute Gini impurity
        def gini_impurity(group):
            if group.numel() == 0:
                return 0.0
            classes, counts = torch.unique(group, return_counts=True)
            probabilities = counts.float() / counts.sum()
            return 1.0 - torch.sum(probabilities ** 2).item()

        total_samples = y_left.numel() + y_right.numel()
        gini_left = gini_impurity(y_left)
        gini_right = gini_impurity(y_right)
        weighted_gini = (
            (y_left.numel() / total_samples) * gini_left
            + (y_right.numel() / total_samples) * gini_right
        )
        return weighted_gini
    
    def print_tree(self, root=None, is_left=False):
        if root is None:
            root_for_text_repr = Node(f"Nr.{self.feature_index} <= {self.threshold}")
        else:
            if is_left:
                answer_text = "yes?:"
            else:
                answer_text = "no?:"
            
            root_for_text_repr = Node(f"{answer_text} Nr.{self.feature_index} <= {self.threshold}", root)            
        
        if not(self.left.is_leaf):
            self.left.print_tree(root_for_text_repr, is_left=True)    
        else: # class labeling
            Node(f"yes?: it is a class {self.left.predicted_class}!", parent=root_for_text_repr)
        if not(self.right.is_leaf):
            self.right.print_tree(root_for_text_repr, is_left=False)    
        else: # class labeling
            Node(f"no?: it is a class {self.left.predicted_class}!", parent=root_for_text_repr)
        
        if root is None: # actual printing
            for pre, _, node in RenderTree(root_for_text_repr):
                print(f"{pre}{node.name}")
            
