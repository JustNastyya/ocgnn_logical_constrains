from loguru import logger

import torch
import json
import os
import tempfile
from itertools import product

from models.model_registry import NodeModels
from experiments.node_level.run_experiment_nl import experiment_logging_wrapper

from constraints.constraints_handlers.fuzzy_handler import NLFuzzyBasedHandler
from constraints.constraints_handlers.distance_handler import NLDistanceBasedHandler

FILEPATH = "experiments/node_level/bunch_json_results/"
TRAIN_NORMAL = True
TRAIN_ANORMAL = True

def json_dump(data, path):
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def experiment_wrapper(
    default_config,
    config_updates,
    is_logical,
    ):
    
    for config_key in config_updates.keys():
        default_config[config_key] = config_updates[config_key]
    default_config["is_logical"] = is_logical

    model_results = experiment_logging_wrapper(config=default_config)
    
    results = {
        "model_config": {
            **default_config,
            "model_train": {
                "model": default_config["model_train"].value.model_class.__name__,
                "train_loop": default_config["model_train"].value.train_loop.__name__,
            },
        },
        "model_results": model_results
    }
    if default_config["is_logical"]:
        results["model_config"]["constrains_handler"] = default_config["constrains_handler"].__name__
    else:
        results["model_config"]["constrains_handler"] = None

    return results


def run_bunch_experiments(
        default_config: dict, 
        baseline_var_pars: dict,
        const_var_pars: dict,
        baseline_model: str,
        filepath: str
    ):
    # iterate over a bunch of settings and train
    # every time 1 + constrains_n models,
    # and save the results into the results folder as json
    
    file_full_path = FILEPATH + filepath
    
    results_l = []
    
    for combo in product(*baseline_var_pars.values()):
        config_updates = dict(zip(baseline_var_pars.keys(), combo))
        config_updates["model_train"] = baseline_model
        print(config_updates)
        if TRAIN_NORMAL:
            results_l.append(experiment_wrapper(
                default_config=default_config,
                config_updates=config_updates,
                is_logical=False,
            ))
        if not(TRAIN_ANORMAL):
            continue
        
        for const_combo in product(*const_var_pars.values()):
            config_updates_const = dict(zip(const_var_pars.keys(), const_combo))
            config_update = config_updates | config_updates_const

            results_l.append(experiment_wrapper(
                default_config=default_config,
                config_updates=config_update,
                is_logical=True,
                ))
            
            json_dump(results_l, file_full_path)

