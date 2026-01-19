from loguru import logger

import torch

from experiments.node_level.model_reference_nl import model_reference
from experiments.node_level.data_loader_nl import get_data, split_test_train
from experiments.logging_utils import print_config_params, get_filename, init_logging

from models.utils import compute_anomaly_scores_node_level
NORMAL_LABEL = 0

def experiment_logging_wrapper(config):
    """A wrapper for logging occuring errors"""
    log_filename = get_filename(config, level="node")
    init_logging(log_filename, level="node")
    
    try:
        run_experiment(config)
    except Exception as e:
        logger.exception("Failure")


def run_experiment(config, constrains=None):
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
    dataset, loader = get_data(dataset_name, batch_size)
    data, train_mask, test_mask = split_test_train(dataset)
    data.x    
    logger.info("##################### creating model")

    dim_features = data.x.shape[1] # train_dataset.num_node_features
    model = ModelClass(dim_features, hidden_dim, num_layers, device).to(device)

    logger.info("##################### staring training")
    train_loop_func(model, data, train_mask, test_mask, epochs, lr)

    logger.info("##################### computing scores")
    train_scores = compute_anomaly_scores_node_level(model, data, train_mask)
    test_scores = compute_anomaly_scores_node_level(model, data, test_mask)
    
    logger.info("computed decision boundary on anomaly scores as 95% quantile:")
    R = torch.quantile(train_scores, 0.95)
    logger.info(R)

    logger.info("Test anomaly scores:")
    logger.info(test_scores[:10])

    logger.info("##################### test rate: ")
    
    pred = (test_scores > R).int()
    # all labels except of 0 is an anomaly
    y = (data.y[test_mask] > 0).int()
    compare = pred == y

    logger.info(compare.sum().item() / len(compare))
    logger.info(f"right classified: {compare.sum().item()} out of {len(compare)}")


if __name__ == "__main__":
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