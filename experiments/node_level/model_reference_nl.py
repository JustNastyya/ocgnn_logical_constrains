# this file contains a json like form of all models and their training loops

from models.simple_ocgin.node_level_ocgin import NodeOCGIN, train_node_ocgin
from models.constrains_in_loss.nl_ocgin import (
    NodeOCGINLossConstrains, 
    train_node_ocgin_add_loss_constrains, 
    train_node_ocgin_weighting,
    train_node_ocgin_irnoring_sus
)
from models.constrains_loss_forecasting.nl_ocgin import (
    NodeOCGINLossConstrains as NLNodeOCGINForecasting,
    train_node_ocgin_add_loss_constrains as nl_train_add,
    train_node_ocgin_weighting as nl_train_weight,
    train_node_ocgin_irnoring_sus as nl_train_ignore
)
from models.constrains_in_model.constrains_attribute_nl import train_node_ocgin_constrains_attribute, NodeOCGINAdditionalArgument

model_reference = {
    "simple_node_ocgin": {
        "model": NodeOCGIN,
        "train_loop": train_node_ocgin
    },
    "logic_add_nl_ocgin": {
        "model": NodeOCGINLossConstrains,
        "train_loop": train_node_ocgin_add_loss_constrains
    },
    "logic_weight_nl_ocgin": {
        "model": NodeOCGINLossConstrains,
        "train_loop": train_node_ocgin_weighting
    },
    "logic_ignore_sus_nl_ocgin": {
        "model": NodeOCGINLossConstrains,
        "train_loop": train_node_ocgin_irnoring_sus
    },
    "constrains_attribute_nl_ocgin": {
        "model": NodeOCGINAdditionalArgument,
        "train_loop": train_node_ocgin_constrains_attribute
    },
    "logic_forecast_add_nl_ocgin": {
        "model": NLNodeOCGINForecasting,
        "train_loop": nl_train_add
    },
    "logic_forecast_weight_nl_ocgin": {
        "model": NLNodeOCGINForecasting,
        "train_loop": nl_train_weight
    },
    "logic_forecast_ignore_sus_nl_ocgin": {
        "model": NLNodeOCGINForecasting,
        "train_loop": nl_train_ignore
    }
}
