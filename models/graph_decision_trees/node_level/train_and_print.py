from loguru import logger
import torch

from torch_geometric.datasets import TUDataset

from models.graph_decision_trees.node_level.model import NodeLevelGraphDecisionTree
from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor


def train_and_print(attribute_list, max_depth, dataset_name):
    logger.info("Loading Config...")
    config = NodeLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    dataset = TUDataset(root="data/MUTAG", name='MUTAG')    

    logger.info(f"Starting training with max depth: {max_depth}")

    X, y = config.extract_features(dataset, balance=True)
    decision_tree = NodeLevelGraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()


if __name__ == "__main__":
    attribute_list = ["node_features", "node_degree", "clustering_coefficient"]
    max_depth = 2
    dataset_name = "MUTAG"
    train_and_print(attribute_list, max_depth, dataset_name)
    