from loguru import logger

import torch
import plotly.express as px
from models.model_registry import NodeModels
from experiments.node_level.data_loader_nl import get_data, split_train_val_test
from experiments.logging_utils import print_config_params, get_filename, init_logging

from constrains.constrains_handlers.nl_fuzzy_based_handler import NLFuzzyBasedHandler
from constrains.constrains_handlers.distance_handler import NLDistanceBasedHandler

NORMAL_LABEL = 0


def debug(config):
    logger.info("##################### Loading config parameters")
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    
    print_config_params(config)
    
    logger.info("##################### Loading data") 
    dataset, _ = get_data(dataset_name, batch_size)
    data, train_mask, val_mask, test_mask, tree_mask = split_train_val_test(dataset)
    
    logger.info("##################### creating model")

    
    dim_features = data.x.shape[1] # train_dataset.num_node_features
    logger.info(f"dim features: {dim_features}")
    
    ConstrainHandler = config["constrains_handler"]
    constrains_filepath = config["constrains_filepath"]
    l_factor = config["l_factor"]
    constrains_handler_obj = ConstrainHandler(constrains_filepath, l_factor, normal_label=NORMAL_LABEL)

    L_constrains = constrains_handler_obj.get_constraint_value(data)
    
    # vektor_metrics(L_constrains)
    # print(data.y)
    fig = px.histogram(L_constrains[train_mask])
    fig.show()
    
    fig = px.histogram(L_constrains[val_mask])
    fig.show()
    


def vektor_metrics(vek):
    print(vek.max())
    print(vek.min())
    print(vek)

    
if __name__ == "__main__":
    config = {
        "hidden_dim": 32,
        "num_layers": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "CiteSeer",
        "model_train": NodeModels.LOGIC_ADD_NL_OCGIN,
        "is_logical": True,
        "constrains_filepath": "constrains/data/CiteSeer_auto_generated_3_101_102_103.json",
        "constrains_handler": NLDistanceBasedHandler,# NLFuzzyBasedHandler,
        "l_factor": 0.1,
        "save_logs": False,
    }

    debug(config=config)
    