from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum

from models.simple_ocgin.node_level_ocgin import NodeOCGIN, train_node_ocgin
from models.simple_ocgin.graph_level_ocgin import GraphOCGIN, train_graph_ocgin
from models.constrains_in_loss.nl_ocgin import (
    NodeOCGINLossConstrains, 
    train_node_ocgin_add_loss_constrains, 
    train_node_ocgin_weighting,
    train_node_ocgin_supression
)
from models.constrains_in_loss.gl_ocgin import (
    GraphOCGINLossConstrains, 
    train_graph_ocgin_add_loss_constrains, 
    train_graph_ocgin_weighting,
    train_graph_ocgin_supression
)
from models.constrains_inference.nl_ocgin import (
        NodeOCGINLossConstrainsInference,
    train_node_ocgin_add_loss_constrains_inference,
    train_node_ocgin_weighting_inference,
)
from models.constrains_inference.gl_ocgin import (
    GraphOCGINLossConstrainsInference,
    train_graph_ocgin_add_loss_constrains_inference,
    train_graph_ocgin_weighting_inference,
)


@dataclass
class ModelReference:
    name: str
    model_class: Any
    train_loop: Callable
    description: str = ""
    level: str = ""


class NodeModels(Enum):
    SIMPLE_NODE_OCGIN = ModelReference(
        name="simple_node_ocgin",
        model_class=NodeOCGIN,
        train_loop=train_node_ocgin,
        description="Baseline OCGIN without constraints",
        level="node"
    )
    LOGIC_ADD_NL_OCGIN = ModelReference(
        name="logic_add_nl_ocgin",
        model_class=NodeOCGINLossConstrains,
        train_loop=train_node_ocgin_add_loss_constrains,
        description="OCGIN with logical constraints added to loss",
        level="node"
    )
    LOGIC_WEIGHT_NL_OCGIN = ModelReference(
        name="logic_weight_nl_ocgin",
        model_class=NodeOCGINLossConstrains,
        train_loop=train_node_ocgin_weighting,
        description="OCGIN with weighted logical constraints",
        level="node"
    )
    LOGIC_IGNORE_SUS_NL_OCGIN = ModelReference(
        name="logic_ignore_sus_nl_ocgin",
        model_class=NodeOCGINLossConstrains,
        train_loop=train_node_ocgin_supression,
        description="OCGIN ignoring suspicious constraints",
        level="node"
    )
    LOGIC_INFERENCE_ADD_NL_OCGIN = ModelReference(
        name="LOGIC_INFERENCE_ADD_NL_OCGIN",
        model_class=NodeOCGINLossConstrainsInference,
        train_loop=train_node_ocgin_add_loss_constrains_inference,
        description="OCGIN with loss forecasting constraints (add)",
        level="node"
    )
    LOGIC_INFERENCE_WEIGHT_NL_OCGIN = ModelReference(
        name="LOGIC_INFERENCE_WEIGHT_NL_OCGIN",
        model_class=NodeOCGINLossConstrainsInference,
        train_loop=train_node_ocgin_weighting_inference,
        description="OCGIN with loss forecasting constraints (weight)",
        level="node"
    )

class GraphModels(Enum):
    SIMPLE_GRAPH_OCGIN = ModelReference(
        name="simple_graph_ocgin",
        model_class=GraphOCGIN,
        train_loop=train_graph_ocgin,
        description="Baseline OCGIN without constraints",
        level="graph"
    )
    LOGIC_ADD_GL_OCGIN = ModelReference(
        name="logic_add_gl_ocgin",
        model_class=GraphOCGINLossConstrains,
        train_loop=train_graph_ocgin_add_loss_constrains,
        description="OCGIN with logical constraints added to loss",
        level="graph"
    )
    LOGIC_WEIGHTING_GL_OCGIN = ModelReference(
        name="logic_weighting_gl_ocgin",
        model_class=GraphOCGINLossConstrains,
        train_loop=train_graph_ocgin_weighting,
        description="OCGIN with weighted logical constraints",
        level="graph"
    )
    LOGIC_IGNORE_SUS_GL_OCGIN = ModelReference(
        name="logic_ignore_sus_gl_ocgin",
        model_class=GraphOCGINLossConstrains,
        train_loop=train_graph_ocgin_supression,
        description="OCGIN ignoring suspicious constraints",
        level="graph"
    )
    LOGIC_INFERENCE_ADD_GL_OCGIN = ModelReference(
        name="LOGIC_INFERENCE_ADD_GL_OCGIN",
        model_class=GraphOCGINLossConstrainsInference,
        train_loop=train_graph_ocgin_add_loss_constrains_inference,
        description="OCGIN with loss forecasting constraints (add)",
        level="graph"
    )
    LOGIC_INFERENCE_WEIGHT_GL_OCGIN = ModelReference(
        name="LOGIC_INFERENCE_WEIGHT_GL_OCGIN",
        model_class=GraphOCGINLossConstrainsInference,
        train_loop=train_graph_ocgin_weighting_inference,
        description="OCGIN with loss forecasting constraints (weight)",
        level="graph"
    )
