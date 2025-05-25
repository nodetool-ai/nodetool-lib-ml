import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.datasets import MakeClassificationDatasetDF


@pytest.mark.asyncio
async def test_make_classification_dataframe():
    ctx = ProcessingContext()
    node = MakeClassificationDatasetDF(
        n_samples=10, n_features=3, n_classes=2, n_redundant=0
    )
    df_ref = await node.process(ctx)

    assert df_ref.columns is not None
    assert len(df_ref.columns) == 4  # three features + target
    assert df_ref.data is not None
    assert len(df_ref.data) == 10
