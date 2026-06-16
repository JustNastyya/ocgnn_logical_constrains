from loguru import logger

import torch
from models.model_registry import NodeModels
from experiments.node_level.run_bunch_experiments import run_bunch_experiments as run_bunch_experiments_nl
from experiments.node_level.data_loader_nl import DATASET_REFERENCE

from models.model_registry import GraphModels
from experiments.graph_level.run_bunch_experiments import run_bunch_experiments as run_bunch_experiments_gl
from experiments.graph_level.data_loader_gl import TUDATASETS

from constraints.constraints_handlers.fuzzy_handler import NLFuzzyBasedHandler, GLNodeFuzzyHandler
from constraints.constraints_handlers.distance_handler import NLDistanceBasedHandler, GLNodeDistanceBasedHandler


def the_pipeline_nl():
    
    # ------------- datasets for the logic in forecasting
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [4, 8, 16, 32, 64, 256],
        "num_layers": [2, 3, 5]
    }
    const_var_pars = {
        "l_factor": [1, 0.1, 0.5, 0.01, 0.001],
        "decision_tree": [{
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 3,
        },
        {
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 4,
        }],
        "model_train": [
            NodeModels.LOGIC_ADD_NL_OCGIN,
            NodeModels.LOGIC_WEIGHT_NL_OCGIN,
            NodeModels.LOGIC_IGNORE_SUS_NL_OCGIN,
            NodeModels.LOGIC_INFERENCE_ADD_NL_OCGIN,
            NodeModels.LOGIC_INFERENCE_WEIGHT_NL_OCGIN,
        ],
        "constrains_handler": [NLFuzzyBasedHandler, NLDistanceBasedHandler]
    }

    datasets = DATASET_REFERENCE.keys()
    
    for dataset_name in datasets:
        result_name = f"nl_decision_trees_{dataset_name}.json"
        default_config["dataset"] = dataset_name
        run_bunch_experiments_nl(
            default_config, 
            baseline_var_pars, 
            const_var_pars,
            NodeModels.SIMPLE_NODE_OCGIN,
            result_name
            )


def the_pipeline_gl():
    # ------------- datasets for the logic in forecasting
    
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "Cora",
        "save_logs": False,
    }
    
    baseline_var_pars = {
        "hidden_dim": [4, 8, 16, 32, 64, 256],
        "num_layers": [2, 3, 5]
    }
    const_var_pars = {
        "l_factor": [1, 0.1, 0.5, 0.01, 0.001],
        "decision_tree": [{
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 3,
        },
        {
            "attribute_list": ["node_features", "node_degree", "clustering_coefficient"],
            "max_depth": 4,
        }],
        "model_train": [
            GraphModels.LOGIC_ADD_GL_OCGIN,
            GraphModels.LOGIC_WEIGHTING_GL_OCGIN,
            GraphModels.LOGIC_IGNORE_SUS_GL_OCGIN,
            GraphModels.LOGIC_INFERENCE_ADD_GL_OCGIN,
            GraphModels.LOGIC_INFERENCE_WEIGHT_GL_OCGIN,
        ],
        "constrains_handler": [GLNodeFuzzyHandler, GLNodeDistanceBasedHandler]
    }
    
    # the actual datasets
    datasets = TUDATASETS
    for dataset_name in datasets:
        result_name = f"gl_decision_trees_{dataset_name}.json"
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
    the_pipeline_gl()
