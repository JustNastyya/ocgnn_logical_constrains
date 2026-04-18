from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum

from models.simple_ocgin.node_level_ocgin import NodeOCGIN, train_node_ocgin
from models.simple_ocgin.graph_level_ocgin import GraphOCGIN, train_graph_ocgin
from models.constrains_in_loss.nl_ocgin import (
    NodeOCGINLossConstrains, 
    train_node_ocgin_add_loss_constrains, 
    train_node_ocgin_weighting,
    train_node_ocgin_irnoring_sus
)
from models.constrains_in_loss.gl_ocgin import (
    GraphOCGINLossConstrains, 
    train_graph_ocgin_add_loss_constrains, 
    train_graph_ocgin_weighting,
    train_graph_ocgin_irnoring_sus
)
from models.constrains_loss_forecasting.nl_ocgin import (
    NodeOCGINLossConstrains as NLNodeOCGINForecasting,
    train_node_ocgin_add_loss_constrains as nl_train_add,
    train_node_ocgin_weighting as nl_train_weight,
    train_node_ocgin_irnoring_sus as nl_train_ignore
)
from models.constrains_loss_forecasting.gl_ocgin import (
    GraphOCGINLossConstrains as GLGraphOCGINForecasting,
    train_graph_ocgin_add_loss_constrains as gl_train_add,
    train_graph_ocgin_weighting as gl_train_weight,
    train_graph_ocgin_irnoring_sus as gl_train_ignore
)
from models.constrains_in_model.constrains_attribute_nl import train_node_ocgin_constrains_attribute, NodeOCGINAdditionalArgument


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
        train_loop=train_node_ocgin_irnoring_sus,
        description="OCGIN ignoring suspicious constraints",
        level="node"
    )
    CONSTRAINS_ATTRIBUTE_NL_OCGIN = ModelReference(
        name="constrains_attribute_nl_ocgin",
        model_class=NodeOCGINAdditionalArgument,
        train_loop=train_node_ocgin_constrains_attribute,
        description="OCGIN with constraints as additional attributes",
        level="node"
    )
    LOGIC_FORECAST_ADD_NL_OCGIN = ModelReference(
        name="logic_forecast_add_nl_ocgin",
        model_class=NLNodeOCGINForecasting,
        train_loop=nl_train_add,
        description="OCGIN with loss forecasting constraints (add)",
        level="node"
    )
    LOGIC_FORECAST_WEIGHT_NL_OCGIN = ModelReference(
        name="logic_forecast_weight_nl_ocgin",
        model_class=NLNodeOCGINForecasting,
        train_loop=nl_train_weight,
        description="OCGIN with loss forecasting constraints (weight)",
        level="node"
    )
    LOGIC_FORECAST_IGNORE_SUS_NL_OCGIN = ModelReference(
        name="logic_forecast_ignore_sus_nl_ocgin",
        model_class=NLNodeOCGINForecasting,
        train_loop=nl_train_ignore,
        description="OCGIN with loss forecasting constraints (ignore sus)",
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
        train_loop=train_graph_ocgin_irnoring_sus,
        description="OCGIN ignoring suspicious constraints",
        level="graph"
    )
    LOGIC_FORECAST_ADD_GL_OCGIN = ModelReference(
        name="logic_forecast_add_gl_ocgin",
        model_class=GLGraphOCGINForecasting,
        train_loop=gl_train_add,
        description="OCGIN with loss forecasting constraints (add)",
        level="graph"
    )
    LOGIC_FORECAST_WEIGHT_GL_OCGIN = ModelReference(
        name="logic_forecast_weight_gl_ocgin",
        model_class=GLGraphOCGINForecasting,
        train_loop=gl_train_weight,
        description="OCGIN with loss forecasting constraints (weight)",
        level="graph"
    )
    LOGIC_FORECAST_IGNORE_SUS_GL_OCGIN = ModelReference(
        name="logic_forecast_ignore_sus_gl_ocgin",
        model_class=GLGraphOCGINForecasting,
        train_loop=gl_train_ignore,
        description="OCGIN with loss forecasting constraints (ignore sus)",
        level="graph"
    )