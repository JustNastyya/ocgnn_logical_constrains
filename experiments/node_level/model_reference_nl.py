# this file contains a json like form of all models and their training loops

from models.simple_ocgin.node_level_ocgin import NodeOCGIN, train_node_ocgin
from models.constrains_in_loss.nl_ocgin import (
    NodeOCGINLossConstrains, 
    train_node_ocgin_add_loss_constrains, 
    train_node_ocgin_weighting,
    train_node_ocgin_irnoring_sus
)
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
    }
}
