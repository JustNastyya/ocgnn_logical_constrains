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
import random


def train_and_print(attribute_list, max_depth, dataset_name):
    logger.info("Loading Config...")
    config = GraphLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    dataset = TUDataset(root="data/MUTAG", name='MUTAG')    

    logger.info(f"Starting training with max depth: {max_depth}")

    X, y = config.extract_features(dataset, balance=True)
    decision_tree = GraphDecisionTreeNode(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()


if __name__ == "__main__":
    attribute_list = ["node_features", "node_degree", "clustering_coefficient"]
    max_depth = 2
    dataset_name = "MUTAG"
    train_and_print(attribute_list, max_depth, dataset_name)
    