import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.decomposition import (
    PCANode,
    NMFNode,
    TruncatedSVDNode,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_pca():
    ctx = ProcessingContext()
    # Create data with 3 features
    X = NPArray.from_numpy(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    )

    node = PCANode(X=X, n_components=2, random_state=42)
    result = await node.process(ctx)

    assert result["transformed"] is not None
    assert result["components"] is not None
    assert result["explained_variance_ratio"] is not None
    assert result["model"] is not None

    transformed = result["transformed"].to_numpy()
    assert transformed.shape == (4, 2)  # Reduced to 2 components

    components = result["components"].to_numpy()
    assert components.shape == (2, 3)  # 2 components, 3 original features

    variance_ratio = result["explained_variance_ratio"].to_numpy()
    assert len(variance_ratio) == 2
    assert np.all(variance_ratio >= 0)
    assert np.all(variance_ratio <= 1)


@pytest.mark.asyncio
async def test_nmf():
    ctx = ProcessingContext()
    # Create non-negative data
    X = NPArray.from_numpy(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    )

    node = NMFNode(X=X, n_components=2, random_state=0, max_iter=200)
    result = await node.process(ctx)

    assert result["transformed"] is not None
    assert result["components"] is not None
    assert result["model"] is not None

    transformed = result["transformed"].to_numpy()
    assert transformed.shape == (4, 2)
    # NMF produces non-negative outputs
    assert np.all(transformed >= 0)

    components = result["components"].to_numpy()
    assert components.shape == (2, 3)
    assert np.all(components >= 0)


@pytest.mark.asyncio
async def test_truncated_svd():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=float)
    )

    node = TruncatedSVDNode(X=X, n_components=2, random_state=42, n_iter=5)
    result = await node.process(ctx)

    assert result["transformed"] is not None
    assert result["components"] is not None
    assert result["explained_variance_ratio"] is not None
    assert result["model"] is not None

    transformed = result["transformed"].to_numpy()
    assert transformed.shape == (3, 2)

    components = result["components"].to_numpy()
    assert components.shape == (2, 4)

    variance_ratio = result["explained_variance_ratio"].to_numpy()
    assert len(variance_ratio) == 2
    assert np.all(variance_ratio >= 0)
    assert np.all(variance_ratio <= 1)


@pytest.mark.asyncio
async def test_pca_single_component():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=float))

    node = PCANode(X=X, n_components=1, random_state=42)
    result = await node.process(ctx)

    transformed = result["transformed"].to_numpy()
    assert transformed.shape == (3, 1)
