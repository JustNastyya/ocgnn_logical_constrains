# this file contains a json like form of all models and their training loops

from models.simple_ocgin.graph_level_ocgin import GraphOCGIN, train_graph_ocgin
from models.constrains_in_loss.gl_ocgin import GraphOCGINLossConstrains, train_graph_ocgin_add_loss_constrains, train_graph_ocgin_irnoring_sus, train_graph_ocgin_weighting

model_reference = {
    "simple_graph_ocgin": {
        "model": GraphOCGIN,
        "train_loop": train_graph_ocgin
    },
    "logic_add_gl_ocgin": {
        "model": GraphOCGINLossConstrains,
        "train_loop": train_graph_ocgin_add_loss_constrains
    },
    "logic_weighting_gl_ocgin": {
        "model": GraphOCGINLossConstrains,
        "train_loop": train_graph_ocgin_weighting
    },
    "logic_ignore_sus_gl_ocgin": {
        "model": GraphOCGINLossConstrains,
        "train_loop": train_graph_ocgin_irnoring_sus
    }
}
