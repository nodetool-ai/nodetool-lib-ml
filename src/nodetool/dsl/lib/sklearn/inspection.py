from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.sklearn.inspection


class PartialDependenceDisplayNode(GraphNode):
    """
    Create Partial Dependence Plot (PDP) visualization data.
    machine learning, model inspection, visualization

    Use cases:
    - Visualizing feature effects
    - Model interpretation
    - Feature relationship analysis
    """

    PartialDependenceKind: typing.ClassVar[type] = (
        nodetool.nodes.lib.sklearn.inspection.PartialDependenceKind
    )
    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Fitted sklearn model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training data",
    )
    features: tuple[int | tuple[int, int]] | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined,
        description="Features for which to create PDP. Can be indices for 1D or tuples for 2D",
    )
    feature_names: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Comma separated names of features"
    )
    grid_resolution: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="Number of points in the grid"
    )
    lower_percentile: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.05, description="Lower percentile to compute the feature values range"
    )
    upper_percentile: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.95, description="Upper percentile to compute the feature values range"
    )
    kind: nodetool.nodes.lib.sklearn.inspection.PartialDependenceKind = Field(
        default=nodetool.nodes.lib.sklearn.inspection.PartialDependenceKind.AVERAGE,
        description="Kind of partial dependence result",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.sklearn.inspection.PartialDependenceDisplay"


import nodetool.nodes.lib.sklearn.inspection


class PartialDependenceNode(GraphNode):
    """
    Calculate Partial Dependence for features.
    machine learning, model inspection, feature effects

    Use cases:
    - Feature impact visualization
    - Model interpretation
    - Understanding feature relationships
    """

    PartialDependenceKind: typing.ClassVar[type] = (
        nodetool.nodes.lib.sklearn.inspection.PartialDependenceKind
    )
    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Fitted sklearn model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Training data",
    )
    features: tuple[int] | GraphNode | tuple[GraphNode, str] = Field(
        default=PydanticUndefined,
        description="List of features for which to calculate PD. Each element can be an int for 1D PD or a list of 2 ints for 2D",
    )
    kind: nodetool.nodes.lib.sklearn.inspection.PartialDependenceKind = Field(
        default=nodetool.nodes.lib.sklearn.inspection.PartialDependenceKind.AVERAGE,
        description="Kind of partial dependence result: 'average' or 'individual'",
    )
    grid_resolution: int | GraphNode | tuple[GraphNode, str] = Field(
        default=100, description="Number of equally spaced points in the grid"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.sklearn.inspection.PartialDependence"


class PermutationImportanceNode(GraphNode):
    """
    Calculate Permutation Feature Importance.
    machine learning, model inspection, feature importance

    Use cases:
    - Feature selection
    - Model interpretation
    - Identifying key predictors
    """

    model: types.SKLearnModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.SKLearnModel(type="sklearn_model", model=None),
        description="Fitted sklearn model",
    )
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Validation data",
    )
    y: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="True labels/values",
    )
    n_repeats: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Number of times to permute each feature"
    )
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(
        default=42, description="Random state for reproducibility"
    )
    scoring: str | GraphNode | tuple[GraphNode, str] = Field(
        default="accuracy",
        description="Scoring metric (if None, uses estimator's default scorer)",
    )
    n_jobs: int | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Number of jobs to run in parallel"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.sklearn.inspection.PermutationImportance"
