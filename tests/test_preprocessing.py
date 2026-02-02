import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.preprocessing import (
    StandardScalerNode,
    MinMaxScalerNode,
    RobustScalerNode,
    NormalizerNode,
    NormalizerNorm,
    TransformNode,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_standard_scaler():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=float))
    
    node = StandardScalerNode(X=X, with_mean=True, with_std=True)
    result = await node.process(ctx)
    
    assert result["transformed"] is not None
    assert result["model"] is not None
    transformed = result["transformed"].to_numpy()
    # Check that mean is approximately 0 and std is approximately 1
    assert np.allclose(np.mean(transformed, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(transformed, axis=0), 1, atol=1e-10)


@pytest.mark.asyncio
async def test_minmax_scaler():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=float))
    
    node = MinMaxScalerNode(X=X, feature_range=(0, 1))
    result = await node.process(ctx)
    
    assert result["transformed"] is not None
    assert result["model"] is not None
    transformed = result["transformed"].to_numpy()
    # Check that values are in range [0, 1]
    assert np.all(transformed >= 0)
    assert np.all(transformed <= 1)


@pytest.mark.asyncio
async def test_robust_scaler():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [3, 4], [5, 6], [100, 200]], dtype=float))
    
    node = RobustScalerNode(
        X=X, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
    )
    result = await node.process(ctx)
    
    assert result["transformed"] is not None
    assert result["model"] is not None
    transformed = result["transformed"].to_numpy()
    assert transformed.shape == X.to_numpy().shape


@pytest.mark.asyncio
async def test_normalizer_l2():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[3, 4], [6, 8]], dtype=float))
    
    node = NormalizerNode(X=X, norm=NormalizerNorm.L2)
    result = await node.process(ctx)
    
    assert result["transformed"] is not None
    assert result["model"] is not None
    transformed = result["transformed"].to_numpy()
    # Check that L2 norm of each row is approximately 1
    norms = np.linalg.norm(transformed, axis=1)
    assert np.allclose(norms, 1, atol=1e-10)


@pytest.mark.asyncio
async def test_normalizer_l1():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.array([[1, 2], [3, 6]], dtype=float))
    
    node = NormalizerNode(X=X, norm=NormalizerNorm.L1)
    result = await node.process(ctx)
    
    assert result["transformed"] is not None
    transformed = result["transformed"].to_numpy()
    # Check that L1 norm of each row is approximately 1
    l1_norms = np.sum(np.abs(transformed), axis=1)
    assert np.allclose(l1_norms, 1, atol=1e-10)


@pytest.mark.asyncio
async def test_transform_node():
    ctx = ProcessingContext()
    X_train = NPArray.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=float))
    X_test = NPArray.from_numpy(np.array([[7, 8]], dtype=float))
    
    # First fit a scaler
    scaler_node = StandardScalerNode(X=X_train)
    scaler_result = await scaler_node.process(ctx)
    
    # Then transform new data
    transform_node = TransformNode(model=scaler_result["model"], X=X_test)
    transform_result = await transform_node.process(ctx)
    
    assert transform_result["transformed"] is not None
    transformed = transform_result["transformed"].to_numpy()
    assert transformed.shape == X_test.to_numpy().shape
