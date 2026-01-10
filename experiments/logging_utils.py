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
RESULTS_PATH = "/home/ana/stuff/1uni/bachlor/experiments/results"

def print_config_params(config):
    logger.info("Config parameters:")
    logger.info("")
    
    for param in config.keys():
        if param != "model_train":
            logger.info(f"{param}: {config[param]}")
    
    logger.info("")
    logger.info(f"Training on model: {config["model_train"]["model"].__name__}")
    logger.info("")


def get_filename(config):
    file_name_list = []
    for filename_entity in FILENAME_CONFIG:
        if filename_entity == "model_train":
            file_name_list.append(config["model_train"]["model"].__name__)
        elif filename_entity == "logical_num":
            if config["is_logical"]:
                file_name_list.append(config["logical_constraint_ref"]) # TODO
        else:
            file_name_list.append(config[filename_entity])
    
    filename = "_".join([str(entity) for entity in file_name_list])
    existing = os.listdir(RESULTS_PATH)
    
    # versioning
    version = 1
    for file in existing:
        if file.startswith(filename):
            version += 1
    
    result_filename = f"{filename}_v{version:02d}.log"
    return result_filename


def init_logging(filename):
    from loguru import logger
    from pathlib import Path

    log_dir = Path(RESULTS_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    
    logger.add(
        log_dir / filename,
        level="TRACE",
        format=log_format
    )
