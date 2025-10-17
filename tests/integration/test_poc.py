from scripts.poc_run import run_poc
import os


def test_run_poc_creates_output(tmp_path):
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_ohlc.csv"))
    out_file = tmp_path / "poc_out.json"
    res = run_poc(csv_path, str(out_file))
    assert "total_value" in res
    assert out_file.exists()
