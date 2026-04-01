from loguru import logger

import torch
from experiments.node_level.model_reference_nl import model_reference
from experiments.node_level.run_bunch_experiments import run_bunch_experiments

from constrains.nl_rules_based_handler import NLRuleBasedHandler


def the_pipeline():
    
    default_config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "model_train": model_reference["logic_add_nl_ocgin"],
        "is_logical": True,
        "constrains_filepath": "constrains/data/Cora_auto_generated_2_101_102_103.json",
        "constrains_handler": NLRuleBasedHandler,
        "l_factor": 0.1,
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [32],
        "num_layers": [2, 3, 4, 5]
    }
    const_var_pars = {
        "l_factor": [0.1, 0.5, 0.01, 0.001],
        "constrains_filepath": [
            "constrains/data/Cora_auto_generated_2_101_102_103.json",
            "constrains/data/Cora_auto_generated_3_101_102_103.json"
        ],
        "model_train": [
            model_reference["logic_add_nl_ocgin"],
            model_reference["logic_weight_nl_ocgin"],
            model_reference["logic_ignore_sus_nl_ocgin"]
        ]
    }
    baseline_model="simple_node_ocgin"
    run_bunch_experiments(
        default_config, 
        baseline_var_pars, 
        const_var_pars,
        baseline_model,
        "test_nl_all.json"
        )


if __name__ == "__main__":
    the_pipeline()