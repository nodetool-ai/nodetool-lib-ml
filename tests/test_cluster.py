import numpy as np
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.sklearn.cluster import (
    KMeansNode,
    DBSCANNode,
    AgglomerativeClusteringNode,
    AgglomerativeClusteringLinkage,
)
from nodetool.metadata.types import NPArray


@pytest.mark.asyncio
async def test_kmeans():
    ctx = ProcessingContext()
    # Create simple clusters
    X = NPArray.from_numpy(
        np.array([[1, 1], [1, 2], [2, 1], [10, 10], [10, 11], [11, 10]], dtype=float)
    )
    
    node = KMeansNode(X=X, n_clusters=2, random_state=42)
    result = await node.process(ctx)
    
    assert result["labels"] is not None
    assert result["centroids"] is not None
    assert result["model"] is not None
    
    labels = result["labels"].to_numpy()
    assert len(labels) == 6
    assert len(np.unique(labels)) == 2  # Should have 2 clusters
    
    centroids = result["centroids"].to_numpy()
    assert centroids.shape == (2, 2)  # 2 clusters, 2 features


@pytest.mark.asyncio
async def test_dbscan():
    ctx = ProcessingContext()
    # Create points with a clear cluster structure
    X = NPArray.from_numpy(
        np.array(
            [
                [1, 1], [1, 2], [2, 1], [2, 2],
                [10, 10], [10, 11], [11, 10], [11, 11]
            ],
            dtype=float,
        )
    )
    
    node = DBSCANNode(X=X, eps=2.0, min_samples=2, metric="euclidean")
    result = await node.process(ctx)
    
    assert result["labels"] is not None
    assert result["model"] is not None
    
    labels = result["labels"].to_numpy()
    assert len(labels) == 8


@pytest.mark.asyncio
async def test_agglomerative_clustering():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array([[1, 1], [1, 2], [2, 1], [10, 10], [10, 11], [11, 10]], dtype=float)
    )
    
    node = AgglomerativeClusteringNode(
        X=X, n_clusters=2, linkage=AgglomerativeClusteringLinkage.WARD
    )
    result = await node.process(ctx)
    
    assert result["labels"] is not None
    assert result["model"] is not None
    
    labels = result["labels"].to_numpy()
    assert len(labels) == 6
    assert len(np.unique(labels)) == 2


@pytest.mark.asyncio
async def test_agglomerative_clustering_complete_linkage():
    ctx = ProcessingContext()
    X = NPArray.from_numpy(
        np.array([[1, 1], [2, 2], [3, 3], [10, 10]], dtype=float)
    )
    
    node = AgglomerativeClusteringNode(
        X=X, n_clusters=2, linkage=AgglomerativeClusteringLinkage.COMPLETE
    )
    result = await node.process(ctx)
    
    labels = result["labels"].to_numpy()
    assert len(labels) == 4
