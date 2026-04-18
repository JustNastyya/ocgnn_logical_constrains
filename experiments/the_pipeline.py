from loguru import logger

import torch
from experiments.model_registry import NodeModels
from experiments.node_level.run_bunch_experiments import run_bunch_experiments as run_bunch_experiments_nl
from experiments.node_level.data_loader_nl import DATASET_REFERENCE

from experiments.model_registry import GraphModels
from experiments.graph_level.run_bunch_experiments import run_bunch_experiments as run_bunch_experiments_gl
from experiments.graph_level.data_loader_gl import TUDATASETS

from constrains.nl_rules_based_handler import NLRuleBasedHandler
from constrains.constrains_score_handler import NLConstraintScoreBasedHandler
from constrains.gl_rules_based_handler import GLRuleBasedHandler
from constrains.constrains_score_handler import GLConstraintScoreBasedHandler

def the_pipeline_nl():
    
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [4, 8, 16, 32, 64, 128, 256],
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
            NodeModels.LOGIC_FORECAST_ADD_NL_OCGIN,
            NodeModels.LOGIC_FORECAST_WEIGHT_NL_OCGIN,
            NodeModels.LOGIC_FORECAST_IGNORE_SUS_NL_OCGIN
        ],
        "constrains_handler": [NLRuleBasedHandler, NLConstraintScoreBasedHandler]
    }
    # the actual datasets
    datasets = ["Cora", "PubMed"]
    
    for dataset_name in datasets:
        result_name = f"loss_forecasting_{dataset_name}.json"
        default_config["dataset"] = dataset_name
        run_bunch_experiments_nl(
            default_config, 
            baseline_var_pars, 
            const_var_pars,
            NodeModels.SIMPLE_NODE_OCGIN,
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
            GraphModels.LOGIC_ADD_GL_OCGIN,
            GraphModels.LOGIC_WEIGHTING_GL_OCGIN,
            GraphModels.LOGIC_IGNORE_SUS_GL_OCGIN
        ],
        "constrains_handler": [GLRuleBasedHandler, GLConstraintScoreBasedHandler]
    }
    
    # the actual datasets
    datasets = TUDATASETS
    
    for dataset_name in datasets:
        result_name = f"loss_forecasting_{dataset_name}.json"
        default_config["dataset"] = dataset_name
        run_bunch_experiments_gl(
            default_config, 
            baseline_var_pars, 
            const_var_pars,
            GraphModels.SIMPLE_GRAPH_OCGIN,
            result_name
            )

if __name__ == "__main__":
    the_pipeline_nl()
