import json
from pathlib import Path


def test_package_metadata_contains_nodes():
    meta_path = (
        Path(__file__).resolve().parents[1]
        / "src/nodetool/package_metadata/nodetool-lib-ml.json"
    )
    data = json.loads(meta_path.read_text())
    assert data.get("name") == "nodetool-lib-ml"
    assert data.get("nodes"), "metadata should list at least one node"
