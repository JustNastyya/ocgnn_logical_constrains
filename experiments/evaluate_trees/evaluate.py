from loguru import logger

import torch

from models.graph_decision_trees.model import GraphDecisionTree


def compute_metrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(name, train_metrics, test_metrics):
    header = f"{'':>20} {'Train Acc':>10} {'Test Acc':>10} {'Bal Acc':>10} {'Prec':>10} {'Recall':>10} {'F1':>10}"
    logger.info(f"--- {name} ---")
    logger.info(header)
    logger.info(f"{'train':>20} {train_metrics['accuracy']:>10.4f} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
    logger.info(f"{'test':>20} {'-':>10} {test_metrics['accuracy']:>10.4f} {test_metrics['balanced_accuracy']:>10.4f} {test_metrics['precision']:>10.4f} {test_metrics['recall']:>10.4f} {test_metrics['f1']:>10.4f}")
    logger.info("")


def evaluate_node_level(dataset_name, attribute_list, max_depth):
    from experiments.node_level.data_loader_nl import get_data, split_train_val_test
    from models.graph_decision_trees.node_level.config import NodeLevelFeatureExtractor

    logger.info(f"Loading node-level dataset '{dataset_name}'...")
    dataset, _ = get_data(dataset_name, batch_size=32)
    data, _, _, test_mask, tree_mask = split_train_val_test(dataset)

    config = NodeLevelFeatureExtractor(attribute_list)

    logger.info(f"Training tree on tree set (max_depth={max_depth})...")
    X_train, y_train = config.extract_features(data, mask=tree_mask, balance=True)
    tree = GraphDecisionTree(max_depth=max_depth)
    tree.fit(X_train, y_train)

    y_pred_train = tree.predict(X_train)
    train_metrics = compute_metrics(y_train, y_pred_train)

    logger.info(f"Evaluating on test set...")
    config_test = NodeLevelFeatureExtractor(attribute_list)
    X_test, y_test = config_test.extract_features(data, mask=test_mask, balance=False)
    y_pred_test = tree.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print_metrics(f"Node-Level {dataset_name} (depth={max_depth})", train_metrics, test_metrics)
    return train_metrics, test_metrics


def evaluate_graph_level(dataset_name, attribute_list, max_depth, batch_size=32):
    from experiments.graph_level.data_loader_gl import get_data, split_train_val_test
    from models.graph_decision_trees.graph_level.config import GraphLevelFeatureExtractor

    logger.info(f"Loading graph-level dataset '{dataset_name}'...")
    dataset, _ = get_data(dataset_name, batch_size)
    _, _, test_loader, tree_loader = split_train_val_test(dataset, batch_size)

    config = GraphLevelFeatureExtractor(attribute_list)

    logger.info(f"Training tree on tree set (max_depth={max_depth})...")
    X_train, y_train = config.extract_features(tree_loader, balance=True)
    tree = GraphDecisionTree(max_depth=max_depth)
    tree.fit(X_train, y_train)

    y_pred_train = tree.predict(X_train)
    train_metrics = compute_metrics(y_train, y_pred_train)

    logger.info(f"Evaluating on test set...")
    config_test = GraphLevelFeatureExtractor(attribute_list)
    X_test, y_test = config_test.extract_features(test_loader, balance=False)
    y_pred_test = tree.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print_metrics(f"Graph-Level {dataset_name} (depth={max_depth})", train_metrics, test_metrics)
    return train_metrics, test_metrics


if __name__ == "__main__":
    NODE_EVALS = [
        ("Cora", ["node_features", "node_degree", "clustering_coefficient"], 3),
        ("CiteSeer", ["node_features", "node_degree", "clustering_coefficient"], 3),
        ("PubMed", ["node_features", "node_degree", "clustering_coefficient"], 3),
    ]

    GRAPH_EVALS = [
        ("MUTAG", ["node_features", "node_degree", "clustering_coefficient"], 3),
        ("AIDS", ["node_features", "node_degree", "clustering_coefficient"], 3),
        ("DHFR", ["node_features", "node_degree", "clustering_coefficient"], 3),
        ]

    logger.info("=" * 90)
    logger.info("Node-Level Decision Tree Evaluation")
    logger.info("=" * 90)

    for dataset_name, attrs, depth in NODE_EVALS:
        try:
            evaluate_node_level(dataset_name, attrs, depth)
        except Exception as e:
            logger.error(f"Failed to evaluate node-level '{dataset_name}': {e}")

    logger.info("=" * 90)
    logger.info("Graph-Level Decision Tree Evaluation")
    logger.info("=" * 90)

    for dataset_name, attrs, depth in GRAPH_EVALS:
        try:
            evaluate_graph_level(dataset_name, attrs, depth)
        except Exception as e:
            logger.error(f"Failed to evaluate graph-level '{dataset_name}': {e}")
