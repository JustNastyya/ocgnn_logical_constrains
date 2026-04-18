# this file contains a json like form of all models and their training loops

from models.simple_ocgin.graph_level_ocgin import GraphOCGIN, train_graph_ocgin
from models.constrains_in_loss.gl_ocgin import GraphOCGINLossConstrains, train_graph_ocgin_add_loss_constrains, train_graph_ocgin_irnoring_sus, train_graph_ocgin_weighting
from models.constrains_loss_forecasting.gl_ocgin import (
    GraphOCGINLossConstrains as GLGraphOCGINForecasting,
    train_graph_ocgin_add_loss_constrains as gl_train_add,
    train_graph_ocgin_weighting as gl_train_weight,
    train_graph_ocgin_irnoring_sus as gl_train_ignore
)

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
    },
    "logic_forecast_add_gl_ocgin": {
        "model": GLGraphOCGINForecasting,
        "train_loop": gl_train_add
    },
    "logic_forecast_weight_gl_ocgin": {
        "model": GLGraphOCGINForecasting,
        "train_loop": gl_train_weight
    },
    "logic_forecast_ignore_sus_gl_ocgin": {
        "model": GLGraphOCGINForecasting,
        "train_loop": gl_train_ignore
    }
}
