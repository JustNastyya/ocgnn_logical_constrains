from loguru import logger

import torch

from experiments.graph_level.model_reference_gl import model_reference
from experiments.graph_level.data_loader_gl import get_data, split_test_train
from experiments.logging_utils import print_config_params, get_filename, init_logging

from models.utils import compute_anomaly_scores_graph_level


def experiment_logging_wrapper(config):
    """A wrapper for logging occuring errors"""
    log_filename = get_filename(config)
    init_logging(log_filename)
    
    try:
        run_experiment(config)
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
    train_loader, test_loader = split_test_train(dataset, batch_size)
    
    logger.info("##################### creating model")

    dim_features = dataset[0].x.shape[1]
    logger.info(f"creating model with number of node features: {dim_features}")
    model = ModelClass(dim_features, hidden_dim, num_layers, device).to(device)

    logger.info("##################### staring training")
    train_loop_func(model, train_loader, epochs, lr)

    logger.info("##################### computing scores")
    train_scores = compute_anomaly_scores_graph_level(model, train_loader)
    test_scores = compute_anomaly_scores_graph_level(model, test_loader)
    
    logger.info("computed decision boundary on anomaly scores as 95% quantile:")
    R = torch.quantile(train_scores, 0.95)
    logger.info(R)

    logger.info("Test anomaly scores:")
    logger.info(test_scores[:10])

    logger.info("##################### test rate: ")
    
    pred = (test_scores > R).int()
    y = []
    for data in test_loader:
        y.append(data.y)
    
    y_vec = torch.cat(y, dim=0)
    compare = pred == y_vec

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
        "dataset": "MUTAG",
        "model_train": model_reference["simple_graph_ocgin"],
        "is_logical": False
    }

    experiment_logging_wrapper(config)
    