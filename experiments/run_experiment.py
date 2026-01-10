from loguru import logger

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from experiments.model_reference import model_reference
from models.utils import compute_anomaly_scores
from experiments.logging_utils import print_config_params, get_filename, init_logging

NORMAL_LABEL = 0


def run_experiment(config, constrains=None):
    log_filename = get_filename(config)
    init_logging(log_filename)
    
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
    
    dataset = TUDataset(root=f"data/{dataset_name}", name=dataset_name)    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    normal_graphs = [d for d in dataset if d.y.item() == NORMAL_LABEL]
    anomalous_graphs = [d for d in dataset if d.y.item() != NORMAL_LABEL]

    num_normal = len(normal_graphs)
    train_size = int(0.8 * num_normal)
    test_size = num_normal - train_size

    train_dataset, test_normal = random_split(
        normal_graphs, [train_size, test_size]
    )

    test_dataset = test_normal + anomalous_graphs

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    logger.info("##################### creating model")

    dim_features = 7 # train_dataset.num_node_features
    model = ModelClass(dim_features, hidden_dim, num_layers, device).to(device)

    logger.info("##################### staring training")
    train_loop_func(model, train_loader, epochs, lr)

    logger.info("##################### computing scores")
    train_scores = compute_anomaly_scores(model, train_loader)
    test_scores = compute_anomaly_scores(model, test_loader)
    
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
        "model_train": model_reference["simple_ocgin"],
        "is_logical": False
    }

    run_experiment(config)
    