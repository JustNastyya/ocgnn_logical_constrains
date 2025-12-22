from loguru import logger
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from anytree import Node, RenderTree
from model import GraphDecisionTreeNode
from config import GraphLevelFeatureExtractor
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# TODO you got a decision tree with only 0s cuz you dataset has mostly ones!
# do a equal split


logger.info("Loading Config...")
attribute_list = ["node_features", "node_degree"]
config = GraphLevelFeatureExtractor(attribute_list)

logger.info("Loading Dataset...")
dataset = TUDataset(root="data/TUDataset", name='REDDIT-BINARY')    
loader = DataLoader(dataset, batch_size=32, shuffle=True)
NORMAL_LABEL = 0
dataset.edge_index

max_depth = 3
logger.info(f"Starting training with max depth: {max_depth}")
X, y = config.extract_features(dataset)
decision_tree = GraphDecisionTreeNode(max_depth=max_depth)
decision_tree.fit(X=X, y=y)

logger.info("Training completed!")

decision_tree.print_tree()

"""
predictions = decision_tree.predict(X_test)

# Calculate accuracy
accuracy = (predictions == y_test).float().mean()
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

"""

