# tests/test_preprocess.py
from preprocess import run_preprocess
import os

def test_preprocess_creates_files(tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    os.environ["DATA_DIR"] = str(out)
    run_preprocess()
    assert (out / "X_train.pkl").exists()
    assert (out / "X_test.pkl").exists()
