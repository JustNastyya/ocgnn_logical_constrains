from loguru import logger
import torch

from torch_geometric.datasets import TUDataset

from models.graph_decision_trees.model import GraphDecisionTree
from models.graph_decision_trees.graph_level.config import GraphLevelFeatureExtractor
from models.graph_decision_trees.graph_level.filename_utils import get_filename


def train_and_print(attribute_list, max_depth, dataset_name, save=True):
    logger.info("Loading Config...")
    config = GraphLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    loader = config.get_data(dataset_name)
    X, y = config.extract_features(loader, balance=True)
    
    logger.info(f"Starting training with max depth: {max_depth}")

    decision_tree = GraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()
    
    filepath = get_filename(dataset_name, attribute_list, max_depth)
    
    logger.info(f"Saving as JSON under {filepath}")
    additional_attributes = config.index_mapping
    decision_tree.save_tree_decisions_as_json(filepath, additional_attributes)


def train_for_model(tree_loader, attribute_list, max_depth, dataset_name):
    logger.info("Loading Config...")
    config = GraphLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    X, y = config.extract_features(tree_loader, balance=True)
    
    logger.info(f"Starting training with max depth: {max_depth}")

    decision_tree = GraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()
    
    filepath = get_filename(dataset_name, attribute_list, max_depth)
    
    logger.info(f"Saving as JSON under {filepath}")
    additional_attributes = config.index_mapping
    decision_tree.save_tree_decisions_as_json(filepath, additional_attributes)

    return filepath

if __name__ == "__main__":
    attribute_list = ["mean_node_features", "mean_node_degree"]
    max_depth = 2
    dataset_name = "MUTAG"
    save = True
    
    train_and_print(
        attribute_list=attribute_list,
        max_depth=max_depth,
        dataset_name=dataset_name,
        save=save
    )
    