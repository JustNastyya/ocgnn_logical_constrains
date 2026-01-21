# this file contains a json like form of all models and their training loops

from models.simple_ocgin.node_level_ocgin import NodeOCGIN, train_node_ocgin
from models.constrains_in_loss.nl_ocgin import NodeOCGINLossConstrains, train_node_ocgin_loss_constrains

model_reference = {
    "simple_node_ocgin": {
        "model": NodeOCGIN,
        "train_loop": train_node_ocgin
    },
    "loss_logic_rule_based": {
        "model": NodeOCGINLossConstrains,
        "train_loop": train_node_ocgin_loss_constrains
    }
}
