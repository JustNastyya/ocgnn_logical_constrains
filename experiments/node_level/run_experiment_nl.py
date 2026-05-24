from loguru import logger

import torch

from experiments.model_registry import NodeModels
from experiments.node_level.data_loader_nl import get_data, split_train_val_test
from experiments.logging_utils import print_config_params, get_filename, init_logging

from constrains.nl_fuzzy_based_handler import NLFuzzyBasedHandler
from constrains.distance_handler import NLDistanceBasedHandler
from models.utils import compute_anomaly_scores_node_level, get_ratios_nl, get_decision_boundary_nl

from models.graph_decision_trees.node_level.train_and_print import train_for_model
NORMAL_LABEL = 0

def experiment_logging_wrapper(config):
    """A wrapper for logging occuring errors"""
    if config["save_logs"]:
        log_filename = get_filename(config, level="node")
        init_logging(log_filename, level="node")
    
    try:
        return run_experiment(config)
    except Exception as e:
        logger.exception("Failure")


def run_experiment(config):
    logger.info("##################### Loading config parameters")
    model_ref = config["model_train"]
    ModelClass = model_ref.value.model_class
    train_loop_func = model_ref.value.train_loop
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    device = config["device"]
    epochs = config["epochs"]
    lr = config["lr"]
    
    print_config_params(config)
    
    logger.info("##################### Loading data") 
    dataset, _ = get_data(dataset_name, batch_size)
    data, train_mask, val_mask, test_mask, tree_mask = split_train_val_test(dataset)

    logger.info("##################### creating model")

    
    dim_features = data.x.shape[1] # train_dataset.num_node_features
    logger.info(f"dim features: {dim_features}")
    model = ModelClass(dim_features, hidden_dim, num_layers, device).to(device)

    logger.info("##################### staring training")

    if config["is_logical"]:
        logger.info("##################### training a descision tree")
        
        decision_tree_att_list = config["decision_tree"]["attribute_list"]
        decision_tree_max_depth = config["decision_tree"]["max_depth"]
        ConstrainHandler = config["constrains_handler"]
        l_factor = config["l_factor"]
        
        constrains_filepath = train_for_model(
            data,
            tree_mask,
            decision_tree_att_list, 
            decision_tree_max_depth, 
            dataset_name
        )

        logger.info("setting up the logical constrains")
        constrains_handler_obj = ConstrainHandler(constrains_filepath, l_factor, normal_label=NORMAL_LABEL)
    
        train_loop_func(model, data, train_mask, test_mask, epochs, lr, constrains_handler_obj)
    else:
        train_loop_func(model, data, train_mask, test_mask, epochs, lr)

    logger.info("##################### computing scores")
    test_scores = compute_anomaly_scores_node_level(model, data, test_mask)
    val_scores = compute_anomaly_scores_node_level(model, data, val_mask)
    
    logger.info("computed decision boundary on anomaly scores as 95% quantile:")
    
    R = get_decision_boundary_nl(val_scores, data, val_mask)
    logger.info("Test anomaly scores:")
    logger.info(val_scores[:10])

    logger.info("")
    logger.info("##################### testing: ")
    
    results = get_ratios_nl(data, test_mask, test_scores, R)
    results["decision_boundary"] = R.item()

    for name, value in results.items():
        logger.info(f"{name}: {round(value, 3)}")
    return results 

if __name__ == "__main__":
    config = {
        "hidden_dim": 32,
        "num_layers": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "model_train": NodeModels.LOGIC_FORECAST_ADD_NL_OCGIN, # constrains_attribute_nl_ocgin
        "is_logical": True,
        "constrains_handler": NLDistanceBasedHandler,# NLFuzzyBasedHandler,
        "l_factor": 0.1,
        "save_logs": False,
        "decision_tree": {
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 3
        }
    }

    experiment_logging_wrapper(config=config)
    