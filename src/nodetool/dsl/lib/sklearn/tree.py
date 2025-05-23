from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.sklearn.tree


class DecisionTreeClassifierNode(GraphNode):
    """
    Decision Tree Classifier.
    machine learning, classification, tree

    Use cases:
    - Classification with interpretable results
    - Feature importance analysis
    - Handling both numerical and categorical data
    """

    DecisionTreeCriterion: typing.ClassVar[type] = (
        nodetool.nodes.lib.sklearn.tree.DecisionTreeCriterion
    )
    X_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training target values",
    )
    max_depth: int | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Maximum depth of the tree"
    )
    min_samples_split: int | GraphNode | tuple[GraphNode, str] = Field(
        default=2, description="Minimum samples required to split a node"
    )
    min_samples_leaf: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Minimum samples required at a leaf node"
    )
    criterion: nodetool.nodes.lib.sklearn.tree.DecisionTreeCriterion = Field(
        default=nodetool.nodes.lib.sklearn.tree.DecisionTreeCriterion.GINI,
        description="Function to measure quality of split ('gini' or 'entropy')",
    )
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random state for reproducibility"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.sklearn.tree.DecisionTreeClassifier"


import nodetool.nodes.lib.sklearn.tree


class DecisionTreeRegressorNode(GraphNode):
    """
    Decision Tree Regressor.
    machine learning, regression, tree

    Use cases:
    - Regression with interpretable results
    - Non-linear relationships
    - Feature importance analysis
    """

    DecisionTreeRegressorCriterion: typing.ClassVar[type] = (
        nodetool.nodes.lib.sklearn.tree.DecisionTreeRegressorCriterion
    )
    X_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training features",
    )
    y_train: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training target values",
    )
    max_depth: int | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Maximum depth of the tree"
    )
    min_samples_split: int | GraphNode | tuple[GraphNode, str] = Field(
        default=2, description="Minimum samples required to split a node"
    )
    min_samples_leaf: int | GraphNode | tuple[GraphNode, str] = Field(
        default=1, description="Minimum samples required at a leaf node"
    )
    criterion: nodetool.nodes.lib.sklearn.tree.DecisionTreeRegressorCriterion = Field(
        default=nodetool.nodes.lib.sklearn.tree.DecisionTreeRegressorCriterion.SQUARED_ERROR,
        description="Function to measure quality of split ('squared_error', 'friedman_mse', 'absolute_error', 'poisson')",
    )
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random state for reproducibility"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.sklearn.tree.DecisionTreeRegressor"
