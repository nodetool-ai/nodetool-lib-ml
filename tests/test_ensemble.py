import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.ensemble import (
    RandomForestClassifierNode,
    RandomForestRegressorNode,
    GradientBoostingClassifierNode,
    GradientBoostingRegressorNode,
    RandomForestCriterion,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_random_forest_classifier():
    ctx = ProcessingContext()
    # Simple classification dataset
    X_train = NPArray.from_numpy(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=float)
    )
    y_train = NPArray.from_numpy(np.array([0, 0, 0, 1, 1, 1]))
    
    node = RandomForestClassifierNode(
        X_train=X_train,
        y_train=y_train,
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None
    assert result["feature_importances"] is not None
    
    importances = result["feature_importances"].to_numpy()
    assert len(importances) == 2  # 2 features


@pytest.mark.asyncio
async def test_random_forest_regressor():
    ctx = ProcessingContext()
    X_train = NPArray.from_numpy(
        np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    )
    y_train = NPArray.from_numpy(np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]))
    
    node = RandomForestRegressorNode(
        X_train=X_train,
        y_train=y_train,
        n_estimators=10,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None
    assert result["feature_importances"] is not None


@pytest.mark.asyncio
async def test_gradient_boosting_classifier():
    ctx = ProcessingContext()
    X_train = NPArray.from_numpy(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    )
    y_train = NPArray.from_numpy(np.array([0, 0, 1, 1]))
    
    node = GradientBoostingClassifierNode(
        X_train=X_train,
        y_train=y_train,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None
    assert result["feature_importances"] is not None
    
    importances = result["feature_importances"].to_numpy()
    assert len(importances) == 2


@pytest.mark.asyncio
async def test_gradient_boosting_regressor():
    ctx = ProcessingContext()
    X_train = NPArray.from_numpy(np.array([[1], [2], [3], [4], [5]], dtype=float))
    y_train = NPArray.from_numpy(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    
    node = GradientBoostingRegressorNode(
        X_train=X_train,
        y_train=y_train,
        n_estimators=10,
        learning_rate=0.1,
        random_state=42,
    )
    result = await node.process(ctx)
    
    assert result["model"] is not None
    assert result["feature_importances"] is not None
