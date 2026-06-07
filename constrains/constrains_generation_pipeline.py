import json
import torch
from loguru import logger

from constrains.graph_decision_trees.model import GraphDecisionTree
from constrains.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor
from constrains.graph_decision_trees.node_level.train_and_print import train_for_model
from constrains.graph_decision_trees.node_level.filename_utils import get_filename as get_nl_filename
from constrains.graph_decision_trees.graph_level.train_and_print import train_from_node_level_features
from constrains.logix_gates.train_model import train_logix_model, extract_logix_rules


def _detect_categorical_columns(x):
    return torch.all((x == 0) | (x == 1), dim=0)


def _merge_and_save(filepath, tree_rules, logix_rules):
    if not logix_rules:
        return filepath

    merged = tree_rules.copy()
    merged["constrains"] = merged["constrains"] + logix_rules

    with open(filepath, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(f"Merged {len(tree_rules['constrains'])} tree rules + {len(logix_rules)} logix rules into {filepath}")
    return filepath


def generate_constrains_nl(data, tree_mask, dataset_name, config):
    decision_tree_att_list = config["decision_tree"]["attribute_list"]
    decision_tree_max_depth = config["decision_tree"]["max_depth"]
    
    # Extract and balance features (includes all node_features + structural features)
    config_ext = NodeLevelFeatureExtractor(decision_tree_att_list)
    X, y = config_ext.extract_features(data, mask=tree_mask, balance=True)

    n_x_cols = data.x.shape[1]

    # Auto-detect categorical columns in data.x
    cat_col_mask = _detect_categorical_columns(data.x)
    cat_indices = torch.where(cat_col_mask)[0].tolist()
    # Keep only indices within the first n_x_cols of X (the node_features region)
    cat_in_x = [i for i in cat_indices if i < n_x_cols]

    logger.info(f"Detected {len(cat_in_x)} categorical columns out of {n_x_cols} node features, max_depth={decision_tree_max_depth}")

    # Train tree on ALL features (indices align with handler's X)
    tree = GraphDecisionTree(max_depth=decision_tree_max_depth)
    tree.fit(X, y)
    tree.print_tree()

    filepath = get_nl_filename(dataset_name, decision_tree_att_list, decision_tree_max_depth)
    tree_rules = tree.save_tree_decisions_as_json(filepath, config_ext.index_mapping)

    # Train logix on categorical columns if any
    logix_rules = []
    if cat_in_x:
        X_cat = X[:, cat_in_x].float()
        if X_cat.shape[1] > 0:
            logix_model = train_logix_model(X_cat, y, config)
            logix_rules = extract_logix_rules(logix_model, cat_in_x)
        else:
            logger.info("No categorical features or no logix config, skipping logix training")

    return _merge_and_save(filepath, tree_rules, logix_rules)


def generate_constrains_gl(tree_loader, dataset_name, config):
    decision_tree_att_list = config["decision_tree"]["attribute_list"]
    decision_tree_max_depth = config["decision_tree"]["max_depth"]
    
    # Train graph-level tree on node-level features (existing approach — uses balance internally)
    filepath, tree_rules = train_from_node_level_features(
        tree_loader,
        decision_tree_att_list,
        decision_tree_max_depth,
        dataset_name,
    )

    # Collect all nodes from the tree set for categorical detection and logix training
    all_X = []
    all_y = []
    n_x_cols = None

    for batch in tree_loader:
        if hasattr(batch, "to_data_list"):
            data_list = batch.to_data_list()
        else:
            data_list = [batch]

        for graph_data in data_list:
            if n_x_cols is None:
                n_x_cols = graph_data.x.shape[1]

            # Extract node-level features (same as train_from_node_level_features does)
            config_ext = NodeLevelFeatureExtractor(decision_tree_att_list)
            X_i, y_i = config_ext.extract_features(graph_data, balance=False)
            all_X.append(X_i)
            all_y.append(y_i)

    if not all_X or n_x_cols is None:
        return filepath

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)

    # Auto-detect categorical — use first graph's data.x to determine column types
    first_graph = next(iter(tree_loader))
    if hasattr(first_graph, "to_data_list"):
        first_data = first_graph.to_data_list()[0]
    else:
        first_data = first_graph
    cat_col_mask = _detect_categorical_columns(first_data.x)
    cat_indices = torch.where(cat_col_mask)[0].tolist()
    cat_in_x = [i for i in cat_indices if i < n_x_cols]

    logger.info(f"Detected {len(cat_in_x)} categorical columns out of {n_x_cols} node features")

    # Train logix on categorical columns
    logix_rules = []
    if cat_in_x:
        X_cat = X_all[:, cat_in_x].float()
        if X_cat.shape[1] > 0:
            logix_model = train_logix_model(X_cat, y_all, config)
            logix_rules = extract_logix_rules(logix_model, cat_in_x)
        else:
            logger.info("No categorical features or no logix config, skipping logix training")

    return _merge_and_save(filepath, tree_rules, logix_rules)
