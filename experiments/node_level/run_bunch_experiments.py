from loguru import logger

import torch
import json
import os
import tempfile
from itertools import product

from experiments.node_level.model_reference_nl import model_reference
from experiments.node_level.run_experiment_nl import experiment_logging_wrapper

from constrains.nl_rules_based_handler import NLRuleBasedHandler

FILEPATH = "experiments/node_level/bunch_json_results/"


def json_dump(data, path):
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def experiment_wrapper(*args, **kwargs):
    config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "model_train": model_reference["loss_logic_rule_based"],
        "is_logical": True,
        "constrains_filepath": "constrains/data/Cora_auto_generated_2_101_102_103.json",
        "constrains_handler": NLRuleBasedHandler,
        "l_factor": 0.1,
        "save_logs": False,
    }
    
    for config_key in kwargs.keys():
        config[config_key] = kwargs[config_key]

    model_results = experiment_logging_wrapper(config=config)
    
    results = {
        "model_config": {
            **config,
            "model_train": {
                "model": config["model_train"]["model"].__name__,
                "train_loop": config["model_train"]["train_loop"].__name__,
            },
            "constrains_handler": config["constrains_handler"].__name__,
        },
        "model_results": model_results
    }
    
    # would be better to save in between results
    return results


def run_bunch_experiments():
    # gonna iterate over a bunch of settings and train
    # every time 1 + constrains_n models,
    # and save the results into the results folder as json
    
    hidden_dim_l = [16, 32, 64, 128, 256]
    num_layers_l = [2, 3, 4, 5]
    l_factor_l = [0.1, 0.01, 0.005, 0.001]
    
    constrains_l = [
        "constrains/data/Cora_auto_generated_2_101_102_103.json",
        "constrains/data/Cora_auto_generated_3_101_102_103.json"
    ]
    file_full_path = FILEPATH + "TODO.json"
    
    results_l = []

    for hidden_dim, num_layers in product(
        hidden_dim_l,
        num_layers_l
    ):
        results_l.append(experiment_wrapper(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            is_logical=False,
            model_train=model_reference["simple_node_ocgin"]
        ))
        for constraint_path, l_factor in product(
                constrains_l,
                l_factor_l
            ):
            results_l.append(experiment_wrapper(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                l_factor=l_factor,
                is_logical=True,
                constrains_filepath=constraint_path
            ))
        
        json_dump(results_l, file_full_path)


if __name__ == "__main__":
    run_bunch_experiments()