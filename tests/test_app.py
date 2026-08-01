"""
Unit test for app imports and integrity verification.
"""

from pathlib import Path
from src.utils import BASE_DIR, MODELS_DIR, RAW_DATA_DIR


def test_project_structure_exists():
    assert (BASE_DIR / "app.py").exists()
    assert (BASE_DIR / "requirements.txt").exists()
    assert (BASE_DIR / "README.md").exists()
    assert (BASE_DIR / "src" / "preprocessing.py").exists()
    assert (BASE_DIR / "src" / "train.py").exists()
    assert (BASE_DIR / "src" / "predict.py").exists()
    assert (BASE_DIR / "src" / "visualization.py").exists()
    assert (BASE_DIR / "src" / "utils.py").exists()
