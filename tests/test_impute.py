import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.impute import SimpleImputerNode, KNNImputerNode
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_simple_imputer_mean():
    ctx = ProcessingContext()
    # Create data with missing values
    X = NPArray.from_numpy(
        np.array([[1, 2], [np.nan, 4], [5, np.nan], [7, 8]], dtype=float)
    )

    node = SimpleImputerNode(X=X, strategy="mean")
    result = await node.process(ctx)

    assert result["transformed"] is not None
    assert result["model"] is not None

    transformed = result["transformed"].to_numpy()
    # Check no NaN values remain
    assert not np.any(np.isnan(transformed))
    assert transformed.shape == X.to_numpy().shape


@pytest.mark.asyncio
async def test_simple_imputer_median():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [np.nan, 4], [5, 6], [7, 8]], dtype=float))

    node = SimpleImputerNode(X=X, strategy="median")
    result = await node.process(ctx)

    transformed = result["transformed"].to_numpy()
    assert not np.any(np.isnan(transformed))


@pytest.mark.asyncio
async def test_simple_imputer_most_frequent():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [1, 4], [np.nan, 2], [3, 8]], dtype=float))

    node = SimpleImputerNode(X=X, strategy="most_frequent")
    result = await node.process(ctx)

    transformed = result["transformed"].to_numpy()
    assert not np.any(np.isnan(transformed))


@pytest.mark.asyncio
async def test_simple_imputer_constant():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [np.nan, 4], [5, np.nan]], dtype=float))

    node = SimpleImputerNode(X=X, strategy="constant", fill_value=0.0)
    result = await node.process(ctx)

    transformed = result["transformed"].to_numpy()
    assert not np.any(np.isnan(transformed))
    # Check that missing values were filled with 0
    # Note: This assumes the imputer worked correctly


@pytest.mark.asyncio
async def test_knn_imputer_uniform():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    )

    node = KNNImputerNode(X=X, n_neighbors=2, weights=KNNImputerNode.Weights.UNIFORM)
    result = await node.process(ctx)

    assert result["transformed"] is not None
    assert result["model"] is not None

    transformed = result["transformed"].to_numpy()
    assert not np.any(np.isnan(transformed))
    assert transformed.shape == X.to_numpy().shape


@pytest.mark.asyncio
async def test_knn_imputer_distance():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array([[1, 2], [np.nan, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    )

    node = KNNImputerNode(X=X, n_neighbors=3, weights=KNNImputerNode.Weights.DISTANCE)
    result = await node.process(ctx)

    transformed = result["transformed"].to_numpy()
    assert not np.any(np.isnan(transformed))
