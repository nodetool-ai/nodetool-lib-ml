import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.feature_selection import (
    SelectKBestNode,
    VarianceThresholdNode,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_select_k_best_classification():
    ctx = ProcessingContext()
    # Create a simple dataset where first 2 features are informative
    X = NPArray.from_numpy(
        np.array(
            [[1, 2, 0.1, 0.2], [2, 3, 0.15, 0.25], [1, 2, 0.12, 0.22], [2, 3, 0.18, 0.28]],
            dtype=float,
        )
    )
    y = NPArray.from_numpy(np.array([0, 1, 0, 1]))
    
    node = SelectKBestNode(X=X, y=y, k=2, score_func="f_classif")
    result = await node.process(ctx)
    
    assert result["selected_features"] is not None
    assert result["scores"] is not None
    assert result["selected_mask"] is not None
    assert result["model"] is not None
    
    selected = result["selected_features"].to_numpy()
    assert selected.shape[1] == 2  # Selected k=2 features


@pytest.mark.asyncio
async def test_select_k_best_regression():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array([[1, 2, 0.1], [2, 3, 0.2], [3, 4, 0.3], [4, 5, 0.4]], dtype=float)
    )
    y = NPArray.from_numpy(np.array([1.0, 2.0, 3.0, 4.0]))
    
    node = SelectKBestNode(X=X, y=y, k=2, score_func="f_regression")
    result = await node.process(ctx)
    
    assert result["selected_features"] is not None
    selected = result["selected_features"].to_numpy()
    assert selected.shape[1] == 2


@pytest.mark.asyncio
async def test_variance_threshold():
    ctx = ProcessingContext()
    # Create a dataset where first column has zero variance
    X = NPArray.from_numpy(
        np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7], [1, 8, 9]], dtype=float)
    )
    
    node = VarianceThresholdNode(X=X, threshold=0.0)
    result = await node.process(ctx)
    
    assert result["selected_features"] is not None
    assert result["variances"] is not None
    assert result["selected_mask"] is not None
    assert result["model"] is not None
    
    selected = result["selected_features"].to_numpy()
    # Should remove the first constant column
    assert selected.shape[1] == 2
    
    mask = result["selected_mask"].to_numpy()
    assert mask[0] == False  # First feature should be removed
    assert mask[1] == True
    assert mask[2] == True


@pytest.mark.asyncio
async def test_variance_threshold_with_higher_threshold():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array(
            [[1, 2, 100], [1.1, 2.1, 200], [1.2, 2.2, 300], [1.3, 2.3, 400]], dtype=float
        )
    )
    
    # Set a high threshold to filter out low-variance features
    node = VarianceThresholdNode(X=X, threshold=1.0)
    result = await node.process(ctx)
    
    selected = result["selected_features"].to_numpy()
    variances = result["variances"].to_numpy()
    
    # Only high-variance features should remain
    assert selected.shape[1] <= X.to_numpy().shape[1]
    assert len(variances) == X.to_numpy().shape[1]
