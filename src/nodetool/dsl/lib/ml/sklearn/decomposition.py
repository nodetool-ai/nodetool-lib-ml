from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.ml.sklearn.decomposition

class NMFNode(GraphNode):
    """
    Non-Negative Matrix Factorization.
    machine learning, dimensionality reduction, feature extraction

    Use cases:
    - Topic modeling
    - Source separation
    - Feature extraction for non-negative data
    """

    NMFInit: typing.ClassVar[type] = nodetool.nodes.lib.ml.sklearn.decomposition.NMFInit
    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(default=types.NPArray(type='np_array', value=None, dtype='<i8', shape=(1,)), description='Non-negative features for decomposition')
    n_components: int | GraphNode | tuple[GraphNode, str] = Field(default=2, description='Number of components')
    init: nodetool.nodes.lib.ml.sklearn.decomposition.NMFInit = Field(default=nodetool.nodes.lib.ml.sklearn.decomposition.NMFInit.RANDOM, description='Method for initialization')
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(default=0, description='Random state for reproducibility')
    max_iter: int | GraphNode | tuple[GraphNode, str] = Field(default=200, description='Maximum number of iterations')

    @classmethod
    def get_node_type(cls): return "lib.ml.sklearn.decomposition.NMF"



class PCANode(GraphNode):
    """
    Principal Component Analysis for dimensionality reduction.
    machine learning, dimensionality reduction, feature extraction

    Use cases:
    - Dimensionality reduction
    - Feature extraction
    - Data visualization
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(default=types.NPArray(type='np_array', value=None, dtype='<i8', shape=(1,)), description='Features for decomposition')
    n_components: int | GraphNode | tuple[GraphNode, str] = Field(default=2, description='Number of components to keep')
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(default=42, description='Random state for reproducibility')

    @classmethod
    def get_node_type(cls): return "lib.ml.sklearn.decomposition.PCA"



class TruncatedSVDNode(GraphNode):
    """
    Truncated Singular Value Decomposition (LSA).
    machine learning, dimensionality reduction, feature extraction

    Use cases:
    - Text processing (LSA/LSI)
    - Dimensionality reduction for sparse data
    - Feature extraction
    """

    X: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(default=types.NPArray(type='np_array', value=None, dtype='<i8', shape=(1,)), description='Features for decomposition')
    n_components: int | GraphNode | tuple[GraphNode, str] = Field(default=2, description='Number of components')
    random_state: int | GraphNode | tuple[GraphNode, str] = Field(default=42, description='Random state for reproducibility')
    n_iter: int | GraphNode | tuple[GraphNode, str] = Field(default=5, description='Number of iterations for randomized SVD')

    @classmethod
    def get_node_type(cls): return "lib.ml.sklearn.decomposition.TruncatedSVD"


