from loguru import logger

import torch

from experiments.node_level.model_reference_nl import model_reference
from experiments.node_level.data_loader_nl import get_data, split_train_val_test
from experiments.logging_utils import print_config_params, get_filename, init_logging

from constrains.nl_rules_based_handler import NLRuleBasedHandler

from models.utils import compute_anomaly_scores_node_level, get_ratios_nl
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
    ModelClass = config["model_train"]["model"] 
    train_loop_func = config["model_train"]["train_loop"]
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
    data, train_mask, val_mask, test_mask = split_train_val_test(dataset)
    data.x    
    logger.info("##################### creating model")

    
    dim_features = data.x.shape[1] # train_dataset.num_node_features
    model = ModelClass(dim_features, hidden_dim, num_layers, device).to(device)

    logger.info("##################### staring training")

    if config["is_logical"]:
        logger.info("setting up the logical constrains")

        ConstrainHandler = config["constrains_handler"]
        constrains_filepath = config["constrains_filepath"]
        l_factor = config["l_factor"]
        constrains_handler_obj = ConstrainHandler(constrains_filepath, l_factor, normal_label=NORMAL_LABEL)
    
        train_loop_func(model, data, train_mask, test_mask, epochs, lr, constrains_handler_obj)
    else:
        train_loop_func(model, data, train_mask, test_mask, epochs, lr)

    logger.info("##################### computing scores")
    train_scores = compute_anomaly_scores_node_level(model, data, train_mask)
    test_scores = compute_anomaly_scores_node_level(model, data, test_mask)
    
    logger.info("computed decision boundary on anomaly scores as 95% quantile:")
    R = torch.quantile(train_scores, 0.95)
    logger.info("Test anomaly scores:")
    logger.info(test_scores[:10])

    logger.info("")
    logger.info("##################### testing: ")
    
    results = get_ratios_nl(data, test_mask, test_scores, R)
    results["decision_boundary"] = R.item()

    for name, value in results.items():
        logger.info(f"{name}: {round(value, 3)}")
    return results 

if __name__ == "__main__":
    """
    config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "model_train": model_reference["simple_node_ocgin"],
        "is_logical": False,
    }

    experiment_logging_wrapper(config=config)
    
    """
    
    config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "model_train": model_reference["node_loss_logic_rule_based"],
        "is_logical": True,
        "constrains_filepath": "constrains/data/Cora_auto_generated_2_101_102_103.json",
        "constrains_handler": NLRuleBasedHandler,
        "l_factor": 0.1,
        "save_logs": True,
    }

    experiment_logging_wrapper(config=config)
    