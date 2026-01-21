from loguru import logger
import torch

from torch_geometric.datasets import TUDataset

from models.graph_decision_trees.node_level.model import NodeLevelGraphDecisionTree
from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor
from models.graph_decision_trees.node_level.filename_utils import get_filename


def train_and_print(attribute_list, max_depth, dataset_name, save=True):
    logger.info("Loading Config...")
    config = NodeLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    X, y = config.extract_features(dataset_name, balance=True)

    logger.info(f"Starting training with max depth: {max_depth}")

    decision_tree = NodeLevelGraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()
    
    filepath = get_filename(dataset_name, attribute_list, max_depth)
    
    logger.info(f"Saving as JSON under {filepath}")
    decision_tree.save_tree_decisions_as_json(filepath)


if __name__ == "__main__":
    attribute_list = ["node_features", "node_degree", "clustering_coefficient"]
    max_depth = 2
    dataset_name = "Cora"
    save = True
    
    train_and_print(
        attribute_list=attribute_list,
        max_depth=max_depth,
        dataset_name=dataset_name,
        save=save
    )
    