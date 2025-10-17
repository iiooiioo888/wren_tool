from scripts.poc_run import load_data
import os


def test_load_data_columns(tmp_path):
    sample = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ohlc.csv")
    sample = os.path.normpath(sample)
    df = load_data(sample)
    assert "timestamp" in df.columns
    assert "close" in df.columns
    assert len(df) >= 1
