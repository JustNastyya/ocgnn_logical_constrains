from loguru import logger

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
    pass