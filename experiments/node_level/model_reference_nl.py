# this file contains a json like form of all models and their training loops

from models.simple_ocgin.node_level_ocgin import NodeOCGIN, train_node_ocgin

model_reference = {
    "simple_node_ocgin": {
        "model": NodeOCGIN,
        "train_loop": train_node_ocgin
    }
}
