from loguru import logger

import torch
import json
import os
import tempfile
from itertools import product

from experiments.model_registry import GraphModels
from experiments.graph_level.run_experiment_gl import experiment_logging_wrapper

from constrains.gl_rules_based_handler import GLRuleBasedHandler
from constrains.constrains_score_handler import GLConstraintScoreBasedHandler

FILEPATH = "experiments/graph_level/bunch_json_results/"
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
    # gonna iterate over a bunch of settings and train
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



if __name__ == "__main__":
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [4, 8, 16, 32, 64, 128, 256, 512],
        "num_layers": [2, 3, 4, 5]
    }
    const_var_pars = {
        "l_factor": [1, 0.1, 0.5, 0.01, 0.001],
        "decision_tree": [{
            "attribute_list": ["mean_node_features", "mean_node_degree"],
            "max_depth": 3,
        },
        {
            "attribute_list": ["mean_node_features", "mean_node_degree"],
            "max_depth": 2,
        }],
        "model_train": [
            GraphModels.LOGIC_ADD_GL_OCGIN,
            GraphModels.LOGIC_WEIGHTING_GL_OCGIN,
            GraphModels.LOGIC_IGNORE_SUS_GL_OCGIN
        ],
        "constrains_handler": [GLConstraintScoreBasedHandler, GLRuleBasedHandler]
    }
    baseline_model=GraphModels.SIMPLE_GRAPH_OCGIN
    default_config["dataset"] = "MUTAG"
    result_name = "test_gl.json"
    run_bunch_experiments(
            default_config, 
            baseline_var_pars, 
            const_var_pars,
            baseline_model,
            result_name
            )
