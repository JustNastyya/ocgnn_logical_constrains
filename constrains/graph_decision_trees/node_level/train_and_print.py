from loguru import logger
import torch

from constrains.graph_decision_trees.model import GraphDecisionTree
from constrains.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor
from constrains.graph_decision_trees.node_level.filename_utils import get_filename


def train_and_print(attribute_list, max_depth, dataset_name, save=True):
    logger.info("Loading Config...")
    config = NodeLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    data = config.get_data(dataset_name)
    X, y = config.extract_features(data, balance=True)
    
    logger.info(f"Starting training with max depth: {max_depth}")

    decision_tree = GraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()
    
    filepath = get_filename(dataset_name, attribute_list, max_depth)
    
    logger.info(f"Saving as JSON under {filepath}")
    additional_attributes = config.index_mapping
    decision_tree.save_tree_decisions_as_json(filepath, additional_attributes)


def train_for_model(data, tree_mask, attribute_list, max_depth, dataset_name):
    logger.info("Loading Config...")
    config = NodeLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")
    X, y = config.extract_features(data, mask=tree_mask, balance=True)
    
    logger.info(f"Starting training with max depth: {max_depth}")

    decision_tree = GraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X, y=y)

    logger.info("Training completed!")

    decision_tree.print_tree()
    
    filepath = get_filename(dataset_name, attribute_list, max_depth)
    
    logger.info(f"Saving as JSON under {filepath}")
    additional_attributes = config.index_mapping
    rules = decision_tree.save_tree_decisions_as_json(filepath, additional_attributes)

    return filepath, rules


if __name__ == "__main__":
    attribute_list = ["node_features", "node_degree", "clustering_coefficient"]
    max_depth = 3
    dataset_name = "Physics"
    save = True
    
    train_and_print(
        attribute_list=attribute_list,
        max_depth=max_depth,
        dataset_name=dataset_name,
        save=save
    )
    