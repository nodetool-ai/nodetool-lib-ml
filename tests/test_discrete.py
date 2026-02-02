import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.statsmodels.discrete import LogitNode, PoissonNode
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_logit_regression():
    ctx = ProcessingContext()
    # Create simple binary classification data
    X = NPArray.from_numpy(np.array([[1], [2], [3], [4], [5], [6]], dtype=float))
    y = NPArray.from_numpy(np.array([0, 0, 0, 1, 1, 1]))

    node = LogitNode(X=X, y=y)
    result = await node.process(ctx)

    assert result["model"] is not None
    assert result["summary"] is not None
    assert result["params"] is not None
    assert result["pvalues"] is not None
    assert isinstance(result["pseudo_rsquared"], float)

    params = result["params"].to_numpy()
    assert len(params) == 2  # Intercept + 1 feature


@pytest.mark.asyncio
async def test_poisson_regression():
    ctx = ProcessingContext()
    # Create simple count data
    X = NPArray.from_numpy(np.array([[1], [2], [3], [4]], dtype=float))
    y = NPArray.from_numpy(np.array([2, 3, 5, 7]))  # Count data

    node = PoissonNode(X=X, y=y)
    result = await node.process(ctx)

    assert result["model"] is not None
    assert result["summary"] is not None
    assert result["params"] is not None
    assert result["pvalues"] is not None

    params = result["params"].to_numpy()
    assert len(params) == 2  # Intercept + 1 feature
