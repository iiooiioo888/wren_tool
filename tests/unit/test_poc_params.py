from scripts.poc_run import run_poc
import os


def test_poc_with_slippage(tmp_path):
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_ohlc.csv"))
    out_file = tmp_path / "poc_out.json"
    res = run_poc(csv_path, str(out_file), fee=0.001, slippage=0.001, delay_ms=0)
    assert "total_value" in res
    assert res["trades"] is not None
    assert out_file.exists()
