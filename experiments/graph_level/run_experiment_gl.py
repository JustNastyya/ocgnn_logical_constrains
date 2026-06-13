from loguru import logger

import torch

from models.model_registry import GraphModels
from experiments.graph_level.data_loader_gl import get_data, split_train_val_test
from experiments.logging_utils import print_config_params, get_filename, init_logging

from constraints.constraints_handlers.gl_node_fuzzy_handler import GLNodeFuzzyHandler

from constraints.graph_decision_trees.train_and_print import train_for_model_gl

from models.utils import (
    compute_anomaly_scores_graph_level,
    get_ratios_gl,
    get_decision_boundary_gl
)
NORMAL_LABEL = 0

def experiment_logging_wrapper(config):
    """A wrapper for logging occuring errors"""
    if config["save_logs"]:
        log_filename = get_filename(config, level="graph")
        init_logging(log_filename, level="graph")

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
    train_loader, val_loader, test_loader, tree_loader = split_train_val_test(dataset, batch_size)
    
    logger.info("##################### creating model")

    dim_features = dataset[0].x.shape[1]
    logger.info(f"creating model with number of node features: {dim_features}")
    model = ModelClass(dim_features, hidden_dim, num_layers, device).to(device)

    logger.info("##################### staring training")
    if config["is_logical"]:
        logger.info("##################### training a descision tree")
        
        decision_tree_att_list = config["decision_tree"]["attribute_list"]
        decision_tree_max_depth = config["decision_tree"]["max_depth"]
        ConstrainHandler = config["constrains_handler"]
        l_factor = config["l_factor"]
        
        constrains_filepath, _ = train_for_model_gl(
            tree_loader,
            decision_tree_att_list, 
            decision_tree_max_depth, 
            dataset_name
        )
        logger.info("setting up the logical constraints")

        ConstrainHandler = config["constrains_handler"]
        l_factor = config["l_factor"]
        aggregation = config.get("aggregation", "mean")
        constrains_handler_obj = ConstrainHandler(
            constrains_filepath, l_factor,
            normal_label=NORMAL_LABEL,
            aggregation=aggregation
        )
    
        train_loop_func(model, train_loader, epochs, lr, constrains_handler_obj, dataset)
    else:
        train_loop_func(model, train_loader, epochs, lr)


    logger.info("##################### computing scores")
    train_scores = compute_anomaly_scores_graph_level(model, train_loader)
    test_scores = compute_anomaly_scores_graph_level(model, test_loader)
    val_scores = compute_anomaly_scores_graph_level(model, val_loader)
    
    logger.info("computed decision boundary on anomaly scores as 95% quantile:")
    
    R = get_decision_boundary_gl(val_loader, val_scores)

    logger.info("Test anomaly scores:")
    logger.info(test_scores[:10])

    logger.info("")
    logger.info("##################### testing: ")
    
    results = get_ratios_gl(test_loader, test_scores, R)
    results["decision_boundary"] = R.item()

    for name, value in results.items():
        logger.info(f"{name}: {round(value, 3)}")
    return results 



if __name__ == "__main__":
    config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "AIDS",
        "model_train": GraphModels.LOGIC_IGNORE_SUS_GL_OCGIN,
        "is_logical": True,
        "constrains_handler": GLNodeFuzzyHandler,
        "l_factor": 0.1,
        "save_logs": False,
        "decision_tree": {
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 2,
        },
        "aggregation": "mean"
    }

    experiment_logging_wrapper(config)
    