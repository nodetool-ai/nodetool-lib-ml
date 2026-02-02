import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.tree import (
    DecisionTreeClassifierNode,
    DecisionTreeRegressorNode,
    DecisionTreeCriterion,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_decision_tree_classifier():
    ctx = ProcessingContext()
    # Simple classification dataset
    X_train = NPArray.from_numpy(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    )
    y_train = NPArray.from_numpy(np.array([0, 0, 1, 1, 1]))
    
    node = DecisionTreeClassifierNode(
        X_train=X_train,
        y_train=y_train,
        max_depth=3,
        criterion=DecisionTreeCriterion.GINI,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None
    assert result["feature_importances"] is not None
    
    importances = result["feature_importances"].to_numpy()
    assert len(importances) == 2  # 2 features


@pytest.mark.asyncio
async def test_decision_tree_classifier_entropy():
    ctx = ProcessingContext()
    X_train = NPArray.from_numpy(np.array([[1], [2], [3], [4]], dtype=float))
    y_train = NPArray.from_numpy(np.array([0, 0, 1, 1]))
    
    node = DecisionTreeClassifierNode(
        X_train=X_train,
        y_train=y_train,
        criterion=DecisionTreeCriterion.ENTROPY,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None


@pytest.mark.asyncio
async def test_decision_tree_regressor():
    ctx = ProcessingContext()
    X_train = NPArray.from_numpy(np.array([[1], [2], [3], [4], [5]], dtype=float))
    y_train = NPArray.from_numpy(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    
    node = DecisionTreeRegressorNode(
        X_train=X_train,
        y_train=y_train,
        max_depth=5,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None
    assert result["feature_importances"] is not None
    
    importances = result["feature_importances"].to_numpy()
    assert len(importances) == 1  # 1 feature
