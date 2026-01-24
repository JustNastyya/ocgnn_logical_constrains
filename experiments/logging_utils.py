from loguru import logger
import os
import sys
from pathlib import Path

FILENAME_CONFIG = [
    "model_train",
    "dataset",
    "num_layers",
    "hidden_dim",
    "logical_num"
]
LOG_RESULTS_PATH_GL = "/home/ana/stuff/1uni/bachlor/experiments/graph_level/log_results"
LOG_RESULTS_PATH_NL = "/home/ana/stuff/1uni/bachlor/experiments/node_level/log_results"

BUNCH_RESULTS_PATH_GL = "/home/ana/stuff/1uni/bachlor/experiments/graph_level/bunch_results"
BUNCH_RESULTS_PATH_NL = "/home/ana/stuff/1uni/bachlor/experiments/node_level/bunch_results"

def print_config_params(config):
    logger.info("Config parameters:")
    logger.info("")
    
    for param in config.keys():
        if param == "constrains_handler":
            logger.info(f"{param}: {config[param].__name__}")
        elif param != "model_train":
            logger.info(f"{param}: {config[param]}")
        
    logger.info("")
    logger.info(f"Training on model: {config["model_train"]["model"].__name__}")
    logger.info("")


def get_filename(config, level):
    file_name_list = []
    for filename_entity in FILENAME_CONFIG:
        if filename_entity == "model_train":
            file_name_list.append(config["model_train"]["model"].__name__)
        elif filename_entity == "logical_num":
            if config["is_logical"]:
                file_name_list.append(config["constrains_handler"].__name__)
        else:
            file_name_list.append(config[filename_entity])
    
    filename = "_".join([str(entity) for entity in file_name_list])
    
    if level == "node":
        existing = os.listdir(LOG_RESULTS_PATH_NL)
    else:
        existing = os.listdir(LOG_RESULTS_PATH_GL)        
    
    # versioning
    version = 1
    for file in existing:
        if file.startswith(filename):
            version += 1
    
    result_filename = f"{filename}_v{version:02d}.log"
    return result_filename


def init_logging(filename, level):
    from loguru import logger
    from pathlib import Path

    if level == "node":
        log_dir = Path(LOG_RESULTS_PATH_NL)
    else:
        log_dir = Path(LOG_RESULTS_PATH_GL)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    
    logger.add(
        log_dir / filename,
        level="TRACE",
        format=log_format,
        backtrace=True
    )
