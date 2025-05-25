import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.datasets import MakeClassificationDataset
from nodetool.nodes.lib.sklearn.linear_model import LogisticRegressionNode
from nodetool.nodes.lib.sklearn import PredictNode
from nodetool.nodes.lib.sklearn.metrics import AccuracyNode


@pytest.mark.asyncio
async def test_logistic_regression_prediction_accuracy():
    ctx = ProcessingContext()
    dataset_node = MakeClassificationDataset(
        n_samples=50, n_features=4, n_classes=2, random_state=0
    )
    data = await dataset_node.process(ctx)

    lr_node = LogisticRegressionNode(X_train=data["data"], y_train=data["target"])
    lr_result = await lr_node.process(ctx)

    predict_node = PredictNode(model=lr_result["model"], X=data["data"])
    preds = await predict_node.process(ctx)

    accuracy_node = AccuracyNode(y_true=data["target"], y_pred=preds)
    accuracy = await accuracy_node.process(ctx)
    assert 0.5 <= accuracy <= 1.0
