import numpy as np
import pytest
from sklearn import metrics

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.metrics import (
    AccuracyNode,
    PrecisionNode,
    RecallNode,
    F1Node,
    MSENode,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_classification_metrics():
    ctx = ProcessingContext()
    y_true = NPArray.from_numpy(np.array([0, 1, 1, 0, 1]))
    y_pred = NPArray.from_numpy(np.array([0, 1, 0, 0, 1]))

    acc = await AccuracyNode(y_true=y_true, y_pred=y_pred).process(ctx)
    prec = await PrecisionNode(y_true=y_true, y_pred=y_pred).process(ctx)
    rec = await RecallNode(y_true=y_true, y_pred=y_pred).process(ctx)
    f1 = await F1Node(y_true=y_true, y_pred=y_pred).process(ctx)

    assert acc == metrics.accuracy_score(y_true.to_numpy(), y_pred.to_numpy())
    assert prec == metrics.precision_score(y_true.to_numpy(), y_pred.to_numpy())
    assert rec == metrics.recall_score(y_true.to_numpy(), y_pred.to_numpy())
    assert f1 == metrics.f1_score(y_true.to_numpy(), y_pred.to_numpy())


@pytest.mark.asyncio
async def test_regression_mse():
    ctx = ProcessingContext()
    y_true = NPArray.from_numpy(np.array([1.0, 2.0, 3.0]))
    y_pred = NPArray.from_numpy(np.array([1.1, 1.9, 3.2]))

    mse = await MSENode(y_true=y_true, y_pred=y_pred).process(ctx)
    assert pytest.approx(mse, rel=1e-6) == metrics.mean_squared_error(
        y_true.to_numpy(), y_pred.to_numpy()
    )
