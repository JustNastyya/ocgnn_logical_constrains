# this file contains a json like form of all models and their training loops

from models.simple_ocgin.graph_level_ocgin import GraphOCGIN, train_graph_ocgin

model_reference = {
    "simple_graph_ocgin": {
        "model": GraphOCGIN,
        "train_loop": train_graph_ocgin
    },
}
