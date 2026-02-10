# this file contains a json like form of all models and their training loops

from models.simple_ocgin.graph_level_ocgin import GraphOCGIN, train_graph_ocgin
from models.constrains_in_loss.gl_ocgin import GraphOCGINLossConstrains, train_graph_ocgin_add_loss_constrains, train_graph_ocgin_graph_defined_loss_constrains

model_reference = {
    "simple_graph_ocgin": {
        "model": GraphOCGIN,
        "train_loop": train_graph_ocgin
    },
    "loss_logic_graph_ocgin": {
        "model": GraphOCGINLossConstrains,
        "train_loop": train_graph_ocgin_add_loss_constrains
    },
    "loss_specific_logic_graph_ocgin": {
        "model": GraphOCGINLossConstrains,
        "train_loop": train_graph_ocgin_graph_defined_loss_constrains
    }
}
