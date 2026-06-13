from loguru import logger
import torch

from constraints.graph_decision_trees.model import GraphDecisionTree
from constraints.graph_decision_trees.config import NodeLevelFeatureExtractor
from constraints.graph_decision_trees.filename_utils import get_filename


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


def train_for_model_nl(data, tree_mask, attribute_list, max_depth, dataset_name):
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


def train_for_model_gl(tree_loader, attribute_list, max_depth, dataset_name):
    logger.info("Loading Config...")
    config = NodeLevelFeatureExtractor(attribute_list)

    logger.info("Loading Dataset...")

    all_X = []
    all_y = []

    for batch in tree_loader:
        if hasattr(batch, "to_data_list"):
            data_list = batch.to_data_list()
        else:
            data_list = [batch]

        for graph_data in data_list:
            X_i, y_i = config.extract_features(graph_data, balance=False)
            all_X.append(X_i)
            all_y.append(y_i)

    X = torch.cat(all_X, dim=0)
    y = torch.cat(all_y, dim=0)

    X_bal, y_bal = config.balance(X, y)

    logger.info(f"Starting training with max depth: {max_depth} "
                f"on node-level features ({X_bal.shape[1]} dims, {X_bal.shape[0]} nodes after balancing)")

    decision_tree = GraphDecisionTree(max_depth=max_depth)
    decision_tree.fit(X=X_bal, y=y_bal)

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
    