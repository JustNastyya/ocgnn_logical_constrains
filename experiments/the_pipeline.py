from loguru import logger

import torch
from experiments.node_level.model_reference_nl import model_reference as model_reference_nl
from experiments.node_level.run_bunch_experiments import run_bunch_experiments as run_bunch_experiments_nl
from experiments.node_level.data_loader_nl import DATASET_REFERENCE

from experiments.graph_level.model_reference_gl import model_reference as model_reference_gl
from experiments.graph_level.run_bunch_experiments import run_bunch_experiments as run_bunch_experiments_gl
from experiments.graph_level.data_loader_gl import TUDATASETS

from constrains.nl_rules_based_handler import NLRuleBasedHandler
from constrains.constrains_score_handler import NLConstraintScoreBasedHandler
from constrains.gl_rules_based_handler import GLRuleBasedHandler
from constrains.constrains_score_handler import GLConstraintScoreBasedHandler

def the_pipeline():
    
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [4, 8, 16, 32, 64, 128, 256, 512],
        "num_layers": [2, 3, 4, 5]
    }
    const_var_pars = {
        "l_factor": [1, 0.1, 0.5, 0.01, 0.001],
        "decision_tree": [{
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 3,
        },
        {
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 2,
        }],
        "model_train": [
            model_reference_nl["logic_add_nl_ocgin"],
            model_reference_nl["logic_weight_nl_ocgin"],
            model_reference_nl["logic_ignore_sus_nl_ocgin"]
        ],
        "constrains_handler": [NLRuleBasedHandler, NLConstraintScoreBasedHandler]
    }
    baseline_model="simple_node_ocgin"
    
    # the actual datasets
    datasets = DATASET_REFERENCE.keys()
    
    for dataset_name in datasets:
        result_name = f"all_loss_nl_{dataset_name}.json"
        default_config["dataset"] = dataset_name
        run_bunch_experiments_nl(
            default_config, 
            baseline_var_pars, 
            const_var_pars,
            baseline_model,
            result_name
            )


def the_pipeline_gl():
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [4, 8, 16, 32, 64, 128, 256],
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
            model_reference_gl["logic_add_gl_ocgin"],
            model_reference_gl["logic_weighting_gl_ocgin"],
            model_reference_gl["logic_ignore_sus_gl_ocgin"]
        ],
        "constrains_handler": [GLRuleBasedHandler, GLConstraintScoreBasedHandler]
    }
    baseline_model="simple_graph_ocgin"
    
    # the actual datasets
    datasets = TUDATASETS
    
    for dataset_name in datasets:
        result_name = f"all_loss_gl_{dataset_name}.json"
        default_config["dataset"] = dataset_name
        run_bunch_experiments_gl(
            default_config, 
            baseline_var_pars, 
            const_var_pars,
            baseline_model,
            result_name
            )

if __name__ == "__main__":
    the_pipeline_gl()
