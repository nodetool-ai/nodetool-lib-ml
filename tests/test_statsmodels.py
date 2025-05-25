import numpy as np
import statsmodels.api as sm
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.statsmodels.regression import OLSNode
from nodetool.nodes.lib.statsmodels import PredictNode as SMPredictNode
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_ols_regression_predict():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(np.arange(10).reshape(-1, 1))
    y = NPArray.from_numpy(2 * np.arange(10) + 1)

    ols = OLSNode(X=X, y=y)
    results = await ols.process(ctx)
    model = results["model"]
    X_const = NPArray.from_numpy(sm.add_constant(np.arange(10).reshape(-1, 1)))

    pred_node = SMPredictNode(model=model, X=X_const)
    preds = await pred_node.process(ctx)

    assert np.allclose(preds.to_numpy(), y.to_numpy(), atol=1e-8)
